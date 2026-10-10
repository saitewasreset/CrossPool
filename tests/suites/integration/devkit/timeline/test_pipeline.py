import json
import os
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest

import xpool.config
import xpool.native
from xpool.cli import main
from xpool.config import init_global_config
from xpool.devkit.timeline.models import Grant, GrantRequest, ProducerRequest, Quality
from xpool.devkit.timeline.offline import verify
from xpool.devkit.timeline.runtime import Pipeline
from xpool.devkit.timeline.session import Session
from xtest.harness.support.config import reset_global_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_background_pipeline_and_artifact_only_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XPOOL_DEBUG_TIMELINE_ENABLE", "1")
    monkeypatch.setenv("XPOOL_DEBUG_TIMELINE_OUTDIR", str(tmp_path))
    config = init_global_config(cli={"config_path": "configs/xpool.example.toml"})
    owner = Session(tmp_path, config.debug.timeline.model_dump(mode="json"), ["daemon:host"])
    producer = owner.register(
        ProducerRequest(startup_id=uuid4(), pid=os.getpid(), create_time=1, role="daemon", slot="daemon", source="host")
    )
    recorder = xpool.native.devkit.timeline.Recorder(config.debug.native_options().timeline)
    pipeline = Pipeline(producer, lambda: recorder, owner.grant)
    for _ in range(100):
        assert recorder.record(xpool.native.devkit.timeline.Record(timestamp=time.monotonic_ns(), kind=1))
    pipeline.close(time.monotonic() + 5, True)
    assert pipeline.writer_done.is_set()
    assert pipeline.collector_done.is_set()
    owner.close()
    quality = Quality.model_validate_json((producer.directory / "summary.json").read_bytes())
    assert quality.closed
    assert quality.exported == 100
    assert quality.lost_after_commit == 0
    assert verify(owner.directory).complete
    monkeypatch.setattr(xpool.config, "global_config", None)
    assert main(["timeline", "verify", "--session", str(owner.directory)]) == 0
    assert "complete" in capsys.readouterr().out
    assert main(["timeline", "export", "--session", str(owner.directory), "--output", str(tmp_path / "derived")]) == 0
    assert json.loads((tmp_path / "derived" / "quality.json").read_text())["records"] == 100


def test_disk_limit_stops_new_records_and_keeps_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPOOL_DEBUG_TIMELINE_ENABLE", "1")
    monkeypatch.setenv("XPOOL_DEBUG_TIMELINE_OUTDIR", str(tmp_path))
    config = init_global_config(cli={"config_path": "configs/xpool.example.toml"})
    owner = Session(tmp_path, config.debug.timeline.model_dump(mode="json"), ["daemon:host"])
    producer = owner.register(
        ProducerRequest(startup_id=uuid4(), pid=os.getpid(), create_time=1, role="daemon", slot="daemon", source="host")
    )
    owner.maximum = owner.reserve
    recorder = xpool.native.devkit.timeline.Recorder(config.debug.native_options().timeline)
    pipeline = Pipeline(producer, lambda: recorder, owner.grant)
    assert recorder.record(xpool.native.devkit.timeline.Record(timestamp=1, kind=1))
    pipeline.close(time.monotonic() + 5, True)
    assert not recorder.record(xpool.native.devkit.timeline.Record(timestamp=2, kind=1))
    owner.close()
    quality = Quality.model_validate_json((producer.directory / "summary.json").read_bytes())
    assert quality.closed
    assert "disk_limit" in quality.reasons
    assert quality.lost_after_commit == 1
    assert not list(producer.directory.glob("*.bin"))
    assert not verify(owner.directory).complete


def test_slow_grant_retains_lease_after_bounded_shutdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPOOL_DEBUG_TIMELINE_ENABLE", "1")
    monkeypatch.setenv("XPOOL_DEBUG_TIMELINE_OUTDIR", str(tmp_path))
    config = init_global_config(cli={"config_path": "configs/xpool.example.toml"})
    owner = Session(tmp_path, config.debug.timeline.model_dump(mode="json"), ["daemon:host"])
    producer = owner.register(
        ProducerRequest(startup_id=uuid4(), pid=os.getpid(), create_time=1, role="daemon", slot="daemon", source="host")
    )
    entered, resume = threading.Event(), threading.Event()

    def slow_grant(request: GrantRequest) -> Grant:
        entered.set()
        if not resume.wait(5):
            raise TimeoutError("test did not release grant")
        return owner.grant(request)

    recorder = xpool.native.devkit.timeline.Recorder(config.debug.native_options().timeline)
    pipeline = Pipeline(producer, lambda: recorder, slow_grant)
    assert entered.wait(2)
    assert recorder.record(xpool.native.devkit.timeline.Record(timestamp=1, kind=1))
    started = time.monotonic()
    pipeline.close(started + 0.05, True)
    assert time.monotonic() - started < 1
    assert pipeline.writer_thread.is_alive()
    assert "flush_timeout" in pipeline.writer.quality.reasons
    with pytest.raises(RuntimeError, match="live reservations"):
        recorder.close()
    resume.set()
    pipeline.collector.join(2)
    pipeline.writer_thread.join(2)
    assert pipeline.writer_done.is_set()
    assert recorder.drained()
    recorder.close()
    owner.close()
    quality = Quality.model_validate_json((producer.directory / "summary.json").read_bytes())
    assert not quality.closed and quality.unknown_tail
    assert "flush_timeout" in quality.reasons
    assert not verify(owner.directory).complete


def test_io_failure_preserves_partial_and_marks_unknown_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPOOL_DEBUG_TIMELINE_ENABLE", "1")
    monkeypatch.setenv("XPOOL_DEBUG_TIMELINE_OUTDIR", str(tmp_path))
    config = init_global_config(cli={"config_path": "configs/xpool.example.toml"})
    owner = Session(tmp_path, config.debug.timeline.model_dump(mode="json"), ["daemon:host"])
    producer = owner.register(
        ProducerRequest(startup_id=uuid4(), pid=os.getpid(), create_time=1, role="daemon", slot="daemon", source="host")
    )

    # Fail precisely at closed-file publication; the real partial contents survive.
    def fail_rename(path: Path, target: Path) -> Path:
        raise OSError(28, f"injected publication failure: {target}")

    recorder = xpool.native.devkit.timeline.Recorder(config.debug.native_options().timeline)
    with monkeypatch.context() as fault:
        fault.setattr(Path, "rename", fail_rename)
        pipeline = Pipeline(producer, lambda: recorder, owner.grant)
        assert recorder.record(xpool.native.devkit.timeline.Record(timestamp=1, kind=1))
        pipeline.close(time.monotonic() + 5, True)
    owner.close()
    assert pipeline.writer.failed
    assert pipeline.writer.quality.lost_after_commit == 1
    assert "trace_io" in pipeline.writer.quality.reasons
    assert list(producer.directory.glob("*.partial"))
    assert not list(producer.directory.glob("*.bin"))
    report = verify(owner.directory)
    assert {issue.reason for issue in report.issues} >= {"unknown_tail", "unpublished_partial"}
