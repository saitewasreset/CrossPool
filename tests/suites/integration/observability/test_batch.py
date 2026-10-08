import json
import subprocess
import tarfile
from pathlib import Path

import pytest
from local_scripts.observability import remote

from xkit.device import DevicePool
from xtest.harness.report import TestRunManifest, TestRunResults


def test_actual_software_capture_identifies_native_build_and_uv() -> None:
    identity = remote.software_identity()
    assert identity["native_sha256"] is not None and len(identity["native_sha256"]) == 64
    assert identity["uv"] is not None and identity["uv"].startswith("uv ")


def test_failed_serving_batch_preserves_attempt_and_return_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "base.toml"
    config.write_text(
        '[scheduler]\nslo={ttft_ms=1000,tbt_ms=50}\n[atn]\ndevices=[0]\n[ffn]\ndevices=[1]\n[vendor]\nmodel_base_uri="/tmp/models"\n[[models]]\nid="Qwen/Qwen3-0.6B"\n'
    )
    output = tmp_path / "failed-run"
    monkeypatch.setattr(remote, "software_identity", lambda: {"native_sha256": "synthetic"})
    monkeypatch.setattr(
        DevicePool, "from_environment", lambda: DevicePool(("GPU-a", "GPU-b"), {"GPU-a": 0, "GPU-b": 1})
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str | bytes]:
        commands.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "synthetic-revision\n")
        if command[0] in {"git", "nvcc", "nvidia-smi"}:
            return subprocess.CompletedProcess(command, 0, "synthetic" if kwargs.get("text") else b"")
        assert command[:8] == ["uv", "run", "--no-sync", "xtest", "run", "--suite", "e2e", "--strict-requirements"]
        run = output / "xtest" / "retained"
        attempt = run / "task" / "pytest-tmp" / "test_serving" / "attempt-1"
        attempt.mkdir(parents=True)
        (attempt / "failure.log").write_text("synthetic startup failure")
        manifest = TestRunManifest(
            run_id="synthetic",
            selected_suites=("e2e",),
            strict_requirements=True,
            python_cases=(remote.SELECTOR,),
            tasks={"task": (remote.SELECTOR,)},
        )
        (run / "run.json").write_text(manifest.model_dump_json())
        (run / "results.json").write_text(
            TestRunResults(finished=True, cleanup_verified=True, overall_result_code=1).model_dump_json()
        )
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_stream(
        command: list[str], log_path: Path, *, environment: dict[str, str] | None = None, check: bool = False
    ) -> subprocess.CompletedProcess[str | bytes]:
        log_path.write_text("synthetic test output")
        return fake_run(command, env=environment, check=check)

    monkeypatch.setattr(remote, "stream_command", fake_stream)
    assert remote.run(config, ("GPU-a", "GPU-b"), output) == 1
    assert (output / "archives/attempt-1/raw/failure.log").read_text() == "synthetic startup failure"
    quality = json.loads((output / "derived/attempt-1/quality.json").read_text())
    assert not quality["representative"]
    assert any(issue["kind"] == "abnormal_or_unknown_exit" for issue in quality["issues"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["xtest_returncode"] == 1
    with tarfile.open(Path(f"{output}.tar.gz")) as bundle:
        assert f"{output.name}/summary.json" in bundle.getnames()
    assert not any("mps-control" in " ".join(command) for command in commands)
