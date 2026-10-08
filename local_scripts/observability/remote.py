"""Operator-invoked remote batching, preserving xtest process ownership."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

import tomli_w
from local_scripts.observability.archive import archive_attempt, digest, timestamp
from local_scripts.observability.models import Participant, RunContext
from local_scripts.observability.timeline import convert, query
from pydantic import JsonValue, TypeAdapter

import xpool.native
from xkit.device import DevicePool
from xkit.serving.readiness import ReadinessEvidence
from xpool.config import XpoolConfig
from xpool.service.wire import ReadinessSnapshot, ReadinessStatus
from xtest.harness.report import TestRunManifest, TestRunResults
from xtest.harness.sglang.serving.launch import SERVING_OBSERVER_RECORD_CAPACITY

SELECTOR = (
    "tests/suites/e2e/sglang/test_e2e_model_serving.py::"
    "test_e2e_model_serving[serving-001-decode-full-prefill-breakable]"
)


def stream_command(
    command: list[str],
    log_path: Path,
    *,
    environment: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    """Tee merged child output to a byte-preserving log and stdout.

    Flush chunks without waiting for a newline. A failed output destination is
    reported after draining the child, so a managed test runner can finish its
    own cleanup rather than blocking on a full pipe.
    """

    failure: OSError | None = None
    log_failed = False
    stdout_failed = False
    child_environment = dict(os.environ if environment is None else environment)
    child_environment["PYTHONUNBUFFERED"] = "1"
    with (
        log_path.open("wb") as log,
        subprocess.Popen(
            command, env=child_environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
        ) as process,
    ):
        if process.stdout is None:
            raise RuntimeError("merged child output pipe was not created")
        with process.stdout:
            while chunk := process.stdout.read(65536):
                if not log_failed:
                    try:
                        log.write(chunk)
                        log.flush()
                    except OSError as error:
                        log_failed = True
                        failure = error
                if not stdout_failed:
                    try:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    except OSError as error:
                        stdout_failed = True
                        failure = failure or error
        returncode = process.wait()
    if failure is not None:
        raise OSError(f"failed to mirror command output for {command!r}; log={log_path}") from failure
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)


def restricted_path(path: Path) -> Path:
    """Exclude the operator's forbidden directory, including symlink aliases."""

    resolved = path.expanduser().resolve()
    if resolved.is_relative_to("/home/incoming"):
        raise ValueError(f"forbidden path: {path}")
    return resolved


def prepare(output: Path, models: Path) -> None:
    output, models = restricted_path(output), restricted_path(models)
    checkpoint = restricted_path(models / "Qwen/Qwen3-0.6B/config.json")
    if not checkpoint.is_file():
        raise ValueError(f"model config is missing: {checkpoint}")
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "logging": {"color": False},
        "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
        "atn": {"devices": [0]},
        "ffn": {"devices": [1]},
        "vendor": {"model_base_uri": str(models)},
        "models": [{"id": "Qwen/Qwen3-0.6B"}],
    }
    config = output / "xpool.toml"
    config.write_text(tomli_w.dumps(payload), encoding="utf-8")
    XpoolConfig.from_file(config, env={})
    command = ["uv", "run", "--no-sync", "xtest", "list", "--suite", "e2e", SELECTOR]
    stream_command(command, output / "collection.log", check=True)
    print(f"prepared config: {config}")


def execution_environment(config: Path, devices: tuple[str, ...]) -> dict[str, str]:
    """Pin the batch to its explicit configuration and reserved device view."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("XPOOL_")
        and key not in {"UV_ENV_FILE", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"}
    }
    environment.update(
        {"XPOOL_CONFIG": str(config), "CUDA_VISIBLE_DEVICES": ",".join(devices), "SGLANG_PLUGINS": "xpool"}
    )
    return environment


def software_identity() -> dict[str, str | None]:
    values: dict[str, str | None] = {"python": sys.version, "platform": platform.platform()}
    for package in ("xpool", "xpool-dev", "torch", "sglang", "sglang-kernel", "nvidia-nvshmem-cu13", "pydantic"):
        values[package] = importlib.metadata.version(package)
    values["uv"] = subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    native_path = Path(xpool.native.__file__)
    values["native_path"] = str(native_path)
    values["native_sha256"] = digest(native_path)
    return values


def read_participants(attempt: Path, devices: tuple[str, ...]) -> list[Participant]:
    """Cross-check the actual registration against the launch and task UUID view."""

    config = XpoolConfig.from_file(attempt / "launch/xpool.toml", env={})
    if config.atn.devices != [0] or config.ffn.devices != [1]:
        raise ValueError("unexpected role placement for serving-001")
    evidence_path = attempt / "launch/readiness/agent-registration.json"
    evidence = TypeAdapter(ReadinessEvidence).validate_json(evidence_path.read_bytes(), strict=True)
    if evidence.last_status_code != 200 or evidence.last_response_excerpt is None:
        raise ValueError("successful Agent registration evidence is unavailable")
    readiness = ReadinessSnapshot.model_validate_json(evidence.last_response_excerpt)
    participants: list[Participant] = []
    for pe, entries, device in ((0, readiness.atnagents, 0), (1, readiness.ffnagents, 1)):
        matching = [entry for entry in entries if entry.device == device and entry.status is ReadinessStatus.ONLINE]
        if len(matching) != 1 or matching[0].pid is None:
            raise ValueError(f"registration does not confirm PE {pe} device {device}")
        participants.append(
            Participant(
                pe=pe,
                device_uuid=devices[device],
                pid=matching[0].pid,
                evidence="single-task UUID view; raw/launch/xpool.toml; raw/launch/readiness/agent-registration.json",
            )
        )
    return participants


def run(config: Path, devices: tuple[str, ...], output: Path) -> int:
    config, output = restricted_path(config), restricted_path(output)
    if len(devices) != 2 or len(set(devices)) != 2 or any(not device.startswith("GPU-") for device in devices):
        raise ValueError("provide exactly two distinct full reserved GPU UUIDs, attention first")
    base = XpoolConfig.from_file(config, env={})
    for model in base.models:
        if model.path is not None:
            restricted_path(model.path)
    if base.vendor.model_base_uri is not None:
        restricted_path(base.vendor.model_base_uri)
    environment = execution_environment(config, devices)
    # The tool schedules exactly one two-device task. Its first allocation preserves this view.
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    try:
        pool = DevicePool.from_environment()
        if pool.uuids != devices:
            raise ValueError("physical UUID inventory does not match the requested order")
        pool.close()
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    software = software_identity()
    software["nvcc"] = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True).stdout
    software["reserved_devices"] = subprocess.run(
        ["nvidia-smi", "-i", ",".join(devices), "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    patch = subprocess.run(["git", "diff", "--binary", "HEAD"], check=True, capture_output=True).stdout
    (output / "source.patch").write_bytes(patch)
    untracked = (
        subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], check=True, capture_output=True)
        .stdout.decode()
        .split("\0")
    )
    source_files: dict[str, JsonValue] = {
        name: digest(restricted_path(Path(name))) for name in untracked if name and Path(name).is_file()
    }
    shutil.copytree(Path(__file__).parent, output / "tool-source", ignore=shutil.ignore_patterns("__pycache__"))
    source_identity: dict[str, JsonValue] = {
        "revision": revision,
        "diff_sha256": digest(output / "source.patch"),
        "untracked_sha256": source_files,
    }
    command = [
        "uv",
        "run",
        "--no-sync",
        "xtest",
        "run",
        "--suite",
        "e2e",
        "--strict-requirements",
        "--result-root",
        str(output / "xtest"),
        SELECTOR,
    ]
    started = timestamp()
    (output / "execution.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_at": started,
                "command": command,
                "devices": devices,
                "software": software,
                "source": source_identity,
            },
            indent=2,
        )
    )
    completed = stream_command(command, output / "execution.log", environment=environment)
    finished = timestamp()
    metadata_issues: list[str] = []
    representative_attempts: list[str] = []
    cleanup_verified = True
    report_runs = list((output / "xtest").glob("*/run.json"))
    if len(report_runs) != 1:
        metadata_issues.append(f"expected one xtest run, found {len(report_runs)}")
    for run_path in report_runs:
        run_manifest = TestRunManifest.model_validate_json(run_path.read_bytes())
        results_path = run_path.parent / "results.json"
        results = (
            TestRunResults.model_validate_json(results_path.read_bytes())
            if results_path.is_file()
            else TestRunResults()
        )
        if run_manifest.python_cases != (SELECTOR,) or len(run_manifest.tasks) != 1:
            raise ValueError("xtest selection differed from the single accepted serving task")
        if not results.finished or results.cleanup_verified is not True:
            metadata_issues.append("xtest cleanup was not verified; allocation requires operator inspection")
            cleanup_verified = False
            continue
        if len(results.tasks) != 1 or results.tasks[0].key not in run_manifest.tasks:
            metadata_issues.append("the selected Serving task has no unique completed result")
        attempts = sorted(run_path.parent.glob("*/pytest-tmp/**/attempt-*"))
        attempts = [
            attempt for attempt in attempts if attempt.is_dir() and attempt.name.removeprefix("attempt-").isdigit()
        ]
        if not attempts:
            metadata_issues.append("no Serving Attempt directory was retained")
        for attempt in attempts:
            number = int(attempt.name.removeprefix("attempt-"))
            context_issues = list(metadata_issues)
            try:
                participants = read_participants(attempt, devices)
            except (ValueError, OSError) as error:
                participants = []
                context_issues.append(f"placement evidence unavailable: {error}")
            context = RunContext(
                run_id=run_id,
                attempt=number,
                command=command,
                started_at=started,
                finished_at=finished,
                time_scope="harness",
                record_capacity=SERVING_OBSERVER_RECORD_CAPACITY,
                returncode=completed.returncode,
                outcome="passed" if completed.returncode == 0 and attempt == attempts[-1] else "failed",
                participants=participants,
                software=software,
                source=source_identity,
                metadata_issues=context_issues,
            )
            archive = output / "archives" / attempt.name
            manifest = archive_attempt(attempt, archive, context)
            derived = output / "derived" / attempt.name
            try:
                report = convert(manifest, derived)
                for phase, invocation in (("prefill", report.prefill_invocation), ("decode", report.decode_invocation)):
                    if invocation is not None:
                        _, _, generation, instance, sequence = json.loads(invocation)
                        (derived / f"{phase}.query.json").write_text(
                            json.dumps(query(derived / "timeline.sqlite", generation, instance, sequence), indent=2)
                        )
                if report.representative:
                    representative_attempts.append(attempt.name)
            except (ValueError, OSError) as error:
                metadata_issues.append(f"{attempt.name}: conversion failed: {error}")
                (archive / "conversion-error.txt").write_text(str(error))
    summary = {
        "run_id": run_id,
        "started_at": started,
        "finished_at": finished,
        "xtest_returncode": completed.returncode,
        "representative_attempts": representative_attempts,
        "issues": metadata_issues,
        "real_trace_acceptance": "pending raw-file and Perfetto inspection",
        "clock_calibrated": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    bundle = Path(f"{output}.tar.gz")
    if bundle.exists():
        raise FileExistsError(f"result bundle already exists: {bundle}")

    def stable_members(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if not cleanup_verified and (
            member.name == f"{output.name}/xtest" or member.name.startswith(f"{output.name}/xtest/")
        ):
            return None
        return member

    with tarfile.open(bundle, "x:gz") as archive:
        archive.add(output, arcname=output.name, filter=stable_members)
    print(f"results: {output}")
    print(f"return bundle: {bundle}")
    return 0 if completed.returncode == 0 and representative_attempts and not metadata_issues else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="CrossPool operator-run Observer experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--output", required=True, type=Path)
    preparation.add_argument("--models", type=Path, default=Path("/home/LAB/yezr/models"))
    execution = commands.add_parser("run")
    execution.add_argument("--config", required=True, type=Path)
    execution.add_argument("--devices", nargs=2, required=True)
    execution.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    try:
        if options.command == "prepare":
            prepare(options.output, options.models)
            return 0
        return run(options.config, tuple(options.devices), options.output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"xpool observer batch failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
