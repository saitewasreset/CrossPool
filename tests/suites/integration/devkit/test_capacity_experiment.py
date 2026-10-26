from pathlib import Path
from typing import Self

import pytest
from pydantic import ValidationError
from scripts.observability_capacity.drain import publish, verify
from scripts.observability_capacity.run import Point, prepare, report

from xbench.harness.serving.case import BenchCatalog, OwnedBenchCase
from xpool.devkit.capacity_counter import Histogram, write_histogram


def histogram(counts: list[int]) -> Histogram:
    return Histogram(
        pid=1,
        role="instance",
        device="GPU-test",
        origin_ns=100,
        last_ns=100 + max(0, len(counts) - 1) * 10_000_000,
        outside_window=0,
        arithmetic_overflow=0,
        allocation_bytes=1024,
        buckets=[(n, 0, 0, 0, 0, 0, 0, 0) for n in counts],
    )


def test_atomic_histogram_publication_and_invalid_clock(tmp_path: Path) -> None:
    path = tmp_path / "counter.json"
    value = histogram([1, 2])
    write_histogram(path, value)
    assert Histogram.model_validate_json(path.read_bytes()) == value
    assert not path.with_suffix(".partial").exists()
    with pytest.raises(ValidationError):
        Histogram.model_validate(value.model_dump() | {"last_ns": 0})


def test_synthetic_publication_checks_crc_extent_and_io_errors(tmp_path: Path) -> None:
    path = tmp_path / "chunk.bin"
    checksum, _, synced = publish(path, memoryview(b"data"), 1, sync=True)
    assert checksum > 0 and synced is not None
    verify(path)
    path.write_bytes(path.read_bytes() + b"extra")
    with pytest.raises(ValueError, match="invalid probe"):
        verify(path)
    with pytest.raises(OSError):
        publish(tmp_path / "missing" / "chunk.bin", memoryview(b"data"), 1, sync=False)


def test_generated_catalogue_preserves_deployment_resolution(tmp_path: Path) -> None:
    point = Point(model="Qwen/Qwen3-0.6B", input_tokens=128, output_tokens=1024, rate=4.0)
    runtime = tmp_path / "runtime.toml"
    runtime.write_text("")
    catalogue = prepare(tmp_path / "attempt", point, runtime)
    case = BenchCatalog.from_file(catalogue).cases[0]
    assert isinstance(case, OwnedBenchCase)
    assert case.deployment.is_file()
    assert case.runtime_config == runtime
    assert case.max_inflight == 128
    assert case.targets[0].output_tokens == 1024
    assert case.arrivals.duration_seconds == 30


def test_offline_report_retains_domains_without_inventing_service(tmp_path: Path) -> None:
    import json

    write_histogram(tmp_path / "counter.1.json", histogram([1, 2]))
    report(tmp_path)
    value = json.loads((tmp_path / "estimates.json").read_text())
    assert value["domains"]["counter.1.json"]["events"] == 3
    assert value["domains"]["counter.1.json"]["capacities"] == []
    assert (tmp_path / "checksums.json").is_file()


def test_instance_readout_follows_successful_native_detach(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    import torch

    import xpool.bootstrap
    import xpool.config
    import xpool.devkit.capacity_counter as counter
    import xpool.native
    from xpool.native import RuntimeRole
    from xpool.runtime.instance import InstanceRankRuntime

    class FakeRunner:
        def forward(self) -> None:
            pass

        def capture(self) -> None:
            pass

        def execute(self) -> None:
            pass

    for module_name, symbol in (
        ("sglang.srt.model_executor.model_runner", "ModelRunner"),
        ("sglang.srt.model_executor.runner.decode_cuda_graph_runner", "DecodeCudaGraphRunner"),
        ("sglang.srt.model_executor.runner.prefill_cuda_graph_runner", "PrefillCudaGraphRunner"),
    ):
        module = ModuleType(module_name)
        setattr(module, symbol, FakeRunner)
        monkeypatch.setitem(sys.modules, module_name, module)
    config = xpool.config.CapacityCounterDebugConfig(enable=True, outdir=tmp_path)
    monkeypatch.setattr(
        xpool.config, "get_global_config", lambda: SimpleNamespace(debug=SimpleNamespace(capacity_counter=config))
    )
    monkeypatch.setattr(xpool.bootstrap, "get_runtime_role", lambda: RuntimeRole.INSTANCE)
    order: list[str] = []

    def detach(runtime: InstanceRankRuntime) -> None:
        order.append("detach-synchronized")
        runtime.arena_handle = None

    def finish() -> SimpleNamespace:
        order.append("finish")
        return SimpleNamespace(
            origin_ns=1,
            last_ns=1,
            outside_window=0,
            arithmetic_overflow=0,
            allocation_bytes=64,
            buckets=[(1, 0, 0, 0, 0, 0, 0, 0)],
        )

    monkeypatch.setattr(InstanceRankRuntime, "detach_arena", detach)
    monkeypatch.setattr(xpool.native.devkit.capacity_counter, "finish", finish)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-test"))
    monkeypatch.setattr(counter, "installed", False)
    counter.install()
    counter.install()
    runtime = object.__new__(InstanceRankRuntime)
    runtime.__dict__["arena_handle"] = object()
    runtime.detach_arena()
    assert order == ["detach-synchronized", "finish"]
    assert len(list(tmp_path.glob("counter.*.json"))) == 1


def test_probe_header_corruption_and_report_errors_remain_explicit(tmp_path: Path) -> None:
    import json

    path = tmp_path / "chunk.bin"
    publish(path, memoryview(b"data"), 1, sync=False)
    contents = bytearray(path.read_bytes())
    contents[9] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="header CRC"):
        verify(path)
    (tmp_path / "counter.bad.json").write_text("{}")
    write_histogram(tmp_path / "counter.good.json", histogram([1]))
    report(tmp_path)
    result = json.loads((tmp_path / "estimates.json").read_text())
    assert "counter.bad.json" in result["report_errors"]
    assert "counter.good.json" in result["domains"]


def test_probe_writer_failure_propagates_with_bounded_host_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch
    from scripts.observability_capacity import drain

    class Tensor:
        def __init__(self, size: int) -> None:
            self.data = bytearray([165]) * size

        def copy_(self, source: Self, *, non_blocking: bool) -> Self:
            self.data[:] = source.data
            return self

        def numpy(self) -> bytearray:
            return self.data

    class Event:
        def record(self) -> None:
            pass

        def query(self) -> bool:
            return True

        def elapsed_time(self, other: Self) -> float:
            return 0.001

    monkeypatch.setattr(torch, "full", lambda shape, *args, **kwargs: Tensor(shape[0]))
    monkeypatch.setattr(torch, "empty_like", lambda source, **kwargs: Tensor(len(source.data)))
    monkeypatch.setattr(torch.cuda, "Stream", object)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: Event())
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-test"))
    settings = drain.DrainSettings(seconds=1, max_bytes=1024**2)
    result = drain.run(tmp_path / "complete", settings)
    assert result.verified and result.file_bytes == 1024**2
    assert len(result.transfers) == 1
    assert result.transfers[0].released_s >= result.transfers[0].published_s

    def broken_publish(*args: object, **kwargs: object) -> None:
        raise OSError("injected writer failure")

    monkeypatch.setattr(drain, "publish", broken_publish)
    with pytest.raises(OSError, match="injected writer failure"):
        drain.run(tmp_path / "failed", settings)
