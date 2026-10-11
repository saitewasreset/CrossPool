import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import xpool.native
import xtest
from xpool.devkit.timeline.format import RECORD, RECORD_BYTES
from xtest.harness.native.case import run_native_case


def test_named_record_preserves_version_one_wire_bytes() -> None:
    options = xpool.native.debug.TimelineOptions(True, 8 << 20, 32 << 20, 4096)
    recorder = xpool.native.devkit.timeline.Recorder(options)
    record = xpool.native.devkit.timeline.Record(
        sequence=999,
        timestamp=0x0807060504030201,
        kind=3,
        site=4,
        generation_high=2**63 + 5,
        generation_low=6,
        endpoint_creator=7,
        endpoint_index=8,
        operation=9,
        instance=10,
        layer=11,
        lane=12,
        lease=13,
        result=14,
        rows=15,
    )
    assert record.generation_high == 2**63 + 5
    assert record.reserved == 0
    assert recorder.record(record)
    recorder.stop()
    chunk = recorder.collect()
    assert chunk is not None
    with memoryview(chunk) as payload:
        assert len(payload) == RECORD_BYTES == 128
        # File version one owns this order; commit assigns the sequence even
        # when the caller supplies a different value in its immutable input.
        assert bytes(payload) == RECORD.pack(1, 0x0807060504030201, 3, 4, 2**63 + 5, *range(6, 16), 0)
    chunk.release()
    del chunk
    assert recorder.drained()
    recorder.close()


def test_host_pool_concurrency_and_receipt_lifetime() -> None:
    options = xpool.native.debug.TimelineOptions(True, 8 << 20, 32 << 20, 4096)
    recorder = xpool.native.devkit.timeline.Recorder(options)
    record = xpool.native.devkit.timeline.Record(timestamp=1, kind=1)
    with ThreadPoolExecutor(max_workers=8) as workers:
        accepted = list(workers.map(lambda _: recorder.record(record), range(1000)))
    counts = recorder.counters()
    assert counts.attempted == 1000
    assert counts.committed == sum(accepted)
    assert counts.committed + counts.dropped == counts.attempted
    lease = recorder.collect()
    assert lease is not None
    view = memoryview(lease)
    original = bytes(view)
    lease.release()
    del lease
    gc.collect()
    assert bytes(view) == original
    recorder.stop()
    assert not recorder.drained()
    view.release()
    del view
    while (chunk := recorder.collect()) is not None:
        chunk.release()
        del chunk
    assert recorder.drained()
    recorder.close()


def test_host_chunk_holds_pool_until_explicit_discard() -> None:
    options = xpool.native.debug.TimelineOptions(True, 8 << 20, 32 << 20, 4096)
    recorder = xpool.native.devkit.timeline.Recorder(options)
    assert recorder.record(xpool.native.devkit.timeline.Record(timestamp=time.monotonic_ns(), kind=1))
    chunk = recorder.collect()
    assert chunk is not None
    recorder.stop()
    assert not recorder.drained()
    chunk.release()
    del chunk
    assert recorder.drained()
    recorder.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_native_diagnostics_log_lock_edges_before_operations(enabled: bool, capfd: pytest.CaptureFixture[str]) -> None:
    options = xpool.native.debug.TimelineOptions(True, 8 << 20, 32 << 20, 4096, diagnostics=enabled)
    recorder = xpool.native.devkit.timeline.Recorder(options)
    assert recorder.record(xpool.native.devkit.timeline.Record(timestamp=1, kind=1))
    chunk = recorder.collect()
    assert chunk is not None
    chunk.release()
    del chunk
    recorder.stop()
    assert recorder.drained()
    assert recorder.counters().committed == 1
    recorder.close()
    messages = capfd.readouterr().err.splitlines()
    if enabled:
        assert any("operation=collect_lock edge=enter" in message for message in messages)
        assert any("operation=collect_lock edge=exit" in message for message in messages)
        assert any("operation=release_lock edge=exit" in message for message in messages)
        assert all("pid=" in message and "tid=" in message and "begin_ns=" in message for message in messages)
        entries = [
            message.split(" elapsed_ns=")[0].replace("edge=enter", "edge=exit")
            for message in messages
            if "edge=enter" in message
        ]
        exits = [message.split(" elapsed_ns=")[0] for message in messages if "edge=exit" in message]
        assert sorted(entries) == sorted(exits)
    else:
        assert messages == []


def isolated_device_diagnostics() -> None:
    xpool.native.initialize(xpool.native.RuntimeRole.INSTANCE, 0, None)
    options = xpool.native.debug.TimelineOptions(True, 262144, 262144, 4096, diagnostics=True)
    recorder = xpool.native.devkit.timeline.Recorder(options, 0)
    recorder.stop()
    deadline = time.monotonic() + 5
    while not recorder.drained() and time.monotonic() < deadline:
        assert recorder.collect() is None
        time.sleep(0.001)
    assert recorder.drained()
    recorder.close()


@xtest.requirements(device_count=1)
def test_device_diagnostics_expose_receipt_operations(tmp_path: Path) -> None:
    workdir = tmp_path / "case"
    run_native_case(isolated_device_diagnostics, workdir=workdir)
    messages = (workdir / "case.log").read_text()
    for name in ("snapshot_launch", "snapshot_copy", "event_record", "event_query", "device_guard"):
        assert f"operation={name} edge=enter" in messages
        assert f"operation={name} edge=exit" in messages
    assert "receipt pid=" in messages and "phase=" in messages and "result=0" in messages
