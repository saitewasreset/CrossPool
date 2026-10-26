"""User-run two-Device capacity experiment with bounded scheduling and retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import tomli_w
from pydantic import BaseModel, ConfigDict, Field
from scripts.observability_capacity.analysis import Service, estimate
from scripts.observability_capacity.drain import DrainResult

from xpool.devkit.capacity_counter import Histogram

REPO = Path(__file__).resolve().parents[2]
MODELS = ("Qwen/Qwen3-0.6B", "deepseek-ai/DeepSeek-V2-Lite-Chat")
PROFILES = ((128, 1024), (1024, 128), (512, 512))


class Mode(StrEnum):
    """Serving perturbation controls."""

    DISABLED = "disabled"
    COUNTED = "counted"
    DRAIN = "drain"


class Point(BaseModel):
    """An independent attempt, with its complete replay inputs retained."""

    model_config = ConfigDict(extra="forbid")
    model: str
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    rate: float = Field(gt=0)
    inflight: int = Field(default=128, gt=0)
    seconds: float = Field(default=30, gt=0)
    mode: Mode = Mode.COUNTED
    probe_sync: bool = False


class Outcome(BaseModel):
    """Failures and incomplete cleanup are explicit, never silently skipped."""

    point: Point
    path: str
    command: list[str]
    returncode: int
    elapsed_s: float
    cleanup_verified: bool
    peak_bytes_s: float | None
    reason: str | None = None
    event_peak_bytes_s: float | None = None
    attachment_peak_bytes_s: float | None = None


def permitted(path: Path) -> Path:
    """Reject operator-forbidden paths before inspecting or creating them."""
    value = path.expanduser().resolve()
    if value.is_relative_to(Path("/home/incoming")):
        raise ValueError("operator-forbidden path")
    return value


def checkpoint(path: Path, value: BaseModel | list[Outcome]) -> None:
    """Publish a recoverable complete JSON checkpoint."""
    content = (
        value.model_dump_json(indent=2)
        if isinstance(value, BaseModel)
        else json.dumps([item.model_dump(mode="json") for item in value], indent=2)
    )
    temporary = path.with_suffix(".partial")
    temporary.write_text(content + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(directory: Path, point: Point, runtime: Path) -> Path:
    """Keep portable deployment resolution and the canonical xbench worker."""
    from xpool.model import ModelId

    root = directory / "inputs"
    benches = root / "benches"
    suites = benches / "suites" / "serving"
    suites.mkdir(parents=True)
    source = (REPO / "benches/suites/serving/multi_model.py").read_text()
    # Enforce an absolute request-count bound before the canonical worker starts serving.
    needle = "    system = None\n"
    if needle not in source:
        raise RuntimeError("canonical worker request-admission seam changed")
    source = source.replace(
        needle,
        "    if len(workload.requests) > 32768:\n"
        '        raise ValueError("capacity screen exceeds 32768 planned requests")\n' + needle,
        1,
    )
    (suites / "multi_model.py").write_text(source)
    identity = ModelId(point.model).uri_encode()
    deployment = root / "configs/deployments" / identity
    deployment.mkdir(parents=True)
    shutil.copyfile(
        REPO / "configs/deployments" / identity / "atn1-ffn1-lanes1.toml", deployment / "atn1-ffn1-lanes1.toml"
    )
    case = {
        "description": "Bounded observability capacity experiment",
        "module": "serving.multi_model",
        "mode": "owned",
        "deployment": "atn1-ffn1-lanes1",
        "runtime_config": str(runtime),
        "seed": 0,
        "max_inflight": point.inflight,
        "startup_timeout_seconds": 300.0,
        "request_timeout_seconds": 90.0,
        "warmup_requests_per_target": 1,
        "bucket_seconds": 0.1,
        "arrivals": {"kind": "poisson", "duration_seconds": point.seconds, "rates": {point.model: point.rate}},
        "targets": [
            {
                "model_id": point.model,
                "graph_mode": "decode-full-prefill-breakable",
                "output_tokens": point.output_tokens,
                "prompts": {"kind": "random", "input_tokens": point.input_tokens},
            }
        ],
    }
    catalogue = benches / "benches.toml"
    catalogue.write_text(tomli_w.dumps({"serving_cases": {str(uuid4()): case}}))
    return catalogue


def stop(process: subprocess.Popen[bytes]) -> None:
    """Ask a task owner to drain; hard termination leaves cleanup unverified."""
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def execute(point: Point, directory: Path, runtime: Path, devices: tuple[str, str], deadline: float) -> Outcome:
    """One supervised benchmark; concurrent probes start after measurement begins."""
    directory.mkdir()
    catalogue = prepare(directory, point, runtime)
    counters = directory / "counters"
    environment = dict(os.environ)
    # Independent observer destinations cannot leak in from an operator shell.
    for key in tuple(environment):
        if key.startswith("XPOOL_DEBUG_"):
            environment.pop(key)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if point.mode is not Mode.DISABLED:
        environment.update(XPOOL_DEBUG_CAPACITY_COUNTER_ENABLE="1", XPOOL_DEBUG_CAPACITY_COUNTER_OUTDIR=str(counters))
    command = [
        str(REPO / ".venv/bin/xbench"),
        "run",
        "--all",
        "--catalog",
        str(catalogue),
        "--cache-root",
        str(directory / "cache"),
    ]
    started = time.monotonic()
    probes: list[subprocess.Popen[bytes]] = []
    handles = []
    timed_out = False
    with (directory / "console.log").open("wb") as log:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                if time.monotonic() >= min(deadline - 135, started + 300 + point.seconds + 135):
                    timed_out = True
                    stop(process)
                    break
                if point.mode is Mode.DRAIN and not probes and list((directory / "cache").rglob("measurement.json")):
                    for index, device in enumerate(devices):
                        probe_environment = dict(environment, CUDA_VISIBLE_DEVICES=device)
                        handle = (directory / f"drain-{index}.log").open("wb")
                        handles.append(handle)
                        probes.append(
                            subprocess.Popen(
                                [
                                    sys.executable,
                                    "-m",
                                    "scripts.observability_capacity.drain",
                                    "--output",
                                    str(directory / f"drain-{index}"),
                                    "--seconds",
                                    "30",
                                    "--max-bytes",
                                    str(512 * 1024**2),
                                    *(["--sync"] if point.probe_sync else []),
                                ],
                                env=probe_environment,
                                stdout=handle,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                        )
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                stop(process)
            for probe in probes:
                try:
                    probe.wait(timeout=max(1, min(45, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    stop(probe)
            for handle in handles:
                handle.close()
    manifests = list((directory / "cache").rglob("repetition.json"))
    cleanup = bool(manifests) and all(
        json.loads(path.read_text()).get("cleanup_verified") is True for path in manifests
    )
    values = [estimate(Histogram.model_validate_json(path.read_bytes())) for path in counters.glob("counter.*.json")]
    peak = sum(value.peak_event_bytes_s["100"] + value.peak_routing_payload_bytes_s["100"] for value in values)
    reason = "deadline interrupted" if timed_out else None
    if any(probe.returncode != 0 for probe in probes):
        reason = "concurrent drain failed; inspect retained logs"
    if point.mode is Mode.DRAIN and not probes:
        reason = "measurement never began; concurrent drain not executed"
    if point.mode is not Mode.DISABLED and (
        len(values) != 3 or any(value.evidence == "incomplete" for value in values)
    ):
        reason = "counter evidence missing or incomplete"
    return Outcome(
        point=point,
        path=str(directory),
        command=command,
        returncode=process.returncode,
        elapsed_s=time.monotonic() - started,
        cleanup_verified=cleanup,
        peak_bytes_s=peak if values else None,
        reason=reason,
        event_peak_bytes_s=sum(v.peak_event_bytes_s["100"] for v in values) if values else None,
        attachment_peak_bytes_s=sum(v.peak_routing_payload_bytes_s["100"] for v in values) if values else None,
    )


def batch(output: Path, runtime: Path, devices: tuple[str, str], seconds: float) -> None:
    """Two-hour default batch, prioritized across models and stopped on unsafe cleanup."""
    output, runtime = permitted(output), permitted(runtime)
    if len(set(devices)) != 2 or any(not device.startswith("GPU-") for device in devices):
        raise ValueError("two distinct full GPU UUIDs are required, attention first")
    from xpool.config import XpoolConfig

    config = XpoolConfig.from_file(runtime, env={})
    from xpool.model import ModelId

    for model in MODELS:
        weights = permitted(config.model_path_of(ModelId(model)))
        if not weights.is_dir() or not (weights / "config.json").is_file():
            raise FileNotFoundError(f"missing local model metadata: {weights}")
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 9 * 1024**3:
        raise OSError("capacity batch requires 8 GiB probe allowance plus 1 GiB free-space reserve")
    shutil.copyfile(runtime, output / "runtime.toml")
    deadline = time.monotonic() + seconds
    (output / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO))
    (output / "revision.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO))
    for name, command in {
        "devices": ["nvidia-smi", "-q"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "filesystem": ["findmnt", "-T", str(output)],
        "space": ["df", "-B1", str(output)],
        "cpu": ["lscpu"],
    }.items():
        result = subprocess.run(command, capture_output=True, check=False)
        (output / f"{name}.txt").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"preflight failed: {name}")
    outcomes: list[Outcome] = []
    inventory = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard"], cwd=REPO, text=True)
    sources = {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in inventory.splitlines()
        if (REPO / name).is_file() and not (REPO / name).is_symlink()
    }
    (output / "source-files.json").write_text(json.dumps(sources, indent=2) + "\n")
    # Use the canonical Native supervisor before spending the serving reservation.
    native_environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices))
    with (output / "native-check.log").open("wb") as native_log:
        native = subprocess.Popen(
            [str(REPO / ".venv/bin/xtest"), "run", "--suite", "cext", "--cache-root", str(output / "native-check")],
            env=native_environment,
            stdout=native_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            native.wait(timeout=min(600, max(1, deadline - time.monotonic() - 135)))
        except subprocess.TimeoutExpired:
            stop(native)
            raise RuntimeError("Native preflight timed out; batch stopped before serving")
        if native.returncode:
            raise RuntimeError("Native preflight failed; inspect native-check.log and retained Native artifacts")
    # Four idle probes reserve at most one GiB in total. Shared-disk pairs run together.
    for sync in (False, True):
        idle = []
        logs = []
        try:
            for index, device in enumerate(devices):
                log = (output / f"idle-{sync}-{index}.log").open("wb")
                logs.append(log)
                command = [
                    sys.executable,
                    "-m",
                    "scripts.observability_capacity.drain",
                    "--output",
                    str(output / f"idle-{sync}-{index}"),
                    "--max-bytes",
                    str(256 * 1024**2),
                ]
                if sync:
                    command.append("--sync")
                idle.append(
                    subprocess.Popen(
                        command,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
            for probe in idle:
                try:
                    probe.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    stop(probe)
                    raise RuntimeError("idle probe timed out; batch stopped with cleanup unverified")
                if probe.returncode:
                    raise RuntimeError("idle probe failed; inspect retained logs")
        finally:
            for probe in idle:
                if probe.poll() is None:
                    stop(probe)
            for log in logs:
                log.close()

    skipped: list[str] = []

    def attempt(point: Point, *, screen: bool = False) -> Outcome | None:
        remaining = deadline - time.monotonic()
        if remaining < 360 or (screen and remaining < seconds / 2):
            skipped.append(point.model_dump_json() + "; deadline reserve")
            (output / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
            return None
        result = execute(point, output / f"attempt-{len(outcomes):03d}", runtime, devices, deadline)
        outcomes.append(result)
        checkpoint(output / "outcomes.json", outcomes)
        if not result.cleanup_verified:
            raise RuntimeError("cleanup unverified; batch stopped before admitting more GPU work")
        return result

    # Visit both models at each load before exploring higher pressure.
    winners: dict[str, Outcome] = {}
    attachment_winners: dict[str, Outcome] = {}
    for rate in (1.0, 4.0, 16.0, 64.0):
        for inputs, outputs in PROFILES:
            for model in MODELS:
                result = attempt(Point(model=model, input_tokens=inputs, output_tokens=outputs, rate=rate), screen=True)
                if result is None:
                    skipped.append(f"screen: {model} {inputs}/{outputs} {rate} RPS; deadline reserve")
                    break
                if result.reason is None and result.returncode in (0, 1) and result.peak_bytes_s is not None:
                    previous = winners.get(model)
                    if previous is None or (result.event_peak_bytes_s or 0) > (previous.event_peak_bytes_s or 0):
                        winners[model] = result
                    previous_attachment = attachment_winners.get(model)
                    if previous_attachment is None or (result.attachment_peak_bytes_s or 0) > (
                        previous_attachment.attachment_peak_bytes_s or 0
                    ):
                        attachment_winners[model] = result
        # Reserve half the wall-clock budget for controls and concurrent probes.
        if deadline - time.monotonic() < seconds / 2:
            break
    for model in MODELS:
        winner = winners.get(model)
        if winner is None:
            continue
        attempt(winner.point.model_copy(update={"inflight": 32}))
        attachment = attachment_winners.get(model)
        if attachment is not None and attachment.path != winner.path:
            attempt(attachment.point.model_copy(update={"seconds": 60.0}))
    for mode, sync in ((Mode.DISABLED, False), (Mode.COUNTED, False), (Mode.DRAIN, False), (Mode.DRAIN, True)):
        for model in MODELS:
            winner = winners.get(model)
            if winner is not None:
                attempt(winner.point.model_copy(update={"mode": mode, "seconds": 300.0, "probe_sync": sync}))
    (output / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    report(output)


def report(root: Path) -> None:
    """Combine independent raw domains and measured probe floors conservatively."""
    from xbench.harness.serving.measure import BenchCaseManifest, load_measurement

    estimates = {}
    serving = {}
    errors: dict[str, str] = {}
    drains: list[DrainResult] = []
    histograms: list[tuple[Path, Histogram]] = []
    for path in root.rglob("drain.json"):
        try:
            drains.append(DrainResult.model_validate_json(path.read_bytes()))
        except (OSError, ValueError) as error:
            errors[str(path.relative_to(root))] = str(error)
    for path in root.rglob("counter.*.json"):
        try:
            histograms.append((path, Histogram.model_validate_json(path.read_bytes())))
        except (OSError, ValueError) as error:
            errors[str(path.relative_to(root))] = str(error)
    lines = ["CrossPool observability capacity estimate", "", "Rates are scoped to retained experiment windows."]
    for path, histogram in histograms:
        local_drains = [value for value in drains if value.device_uuid == histogram.device]
        peers = sum(
            1 for other, value in histograms if other.parent == path.parent and value.device == histogram.device
        )
        service = None
        samples = [item for value in local_drains for item in value.transfers]
        if samples:
            copy = min((item.bytes - 128) / max(1e-9, item.completed_s - item.submitted_s) for item in samples)
            writer = min(item.bytes / max(1e-9, item.published_s - item.write_begin_s) for item in samples)
            service = Service(copy_bytes_s=copy / max(peers, 1), writer_bytes_s=writer / 3)
        result = estimate(histogram, service)
        estimates[str(path.relative_to(root))] = result.model_dump(mode="json")
        lines.extend(
            [
                "",
                str(path.relative_to(root)),
                f"  evidence={result.evidence} duration={result.duration_s:.2f}s",
                f"  event_payload={result.event_bytes_s / 1024**2:.3f} MiB/s; "
                f"routing_payload={result.routing_payload_bytes_s / 1024**2:.3f} MiB/s",
                f"  estimated_graph={result.estimated_graph_bytes_s / 1024**2:.3f} MiB/s",
                f"  estimated_quota_lifetime={result.estimated_disk_seconds}s",
                f"  five_minute_estimate={result.estimated_five_minute_file_bytes} bytes; "
                f"with_margin={result.estimated_five_minute_file_bytes_with_margin}",
            ]
        )
        for capacity in result.capacities:
            lines.append(
                f"  interval={capacity.interval_ms}ms slots={capacity.device_slots} "
                f"host={capacity.host_bytes} bytes overload={capacity.sustained_overload}"
            )
    for path in root.rglob("repetition.json"):
        try:
            case_dir = path.parent.parent
            manifest = BenchCaseManifest.model_validate_json((case_dir / "case.json").read_bytes())
            _, summary = load_measurement(path.parent, manifest.load_workload(case_dir))
            serving[str(path.relative_to(root))] = summary.model_dump(mode="json")
            lines.extend(["", str(path.relative_to(root)), f"  client_window={summary.window_kind}"])
            for model, target in summary.targets.items():
                lines.append(
                    f"  {model}: input_TPS={target.input_tokens_per_second}; "
                    f"output_TPS={target.output_tokens_per_second}; outcomes={target.outcomes}"
                )
        except (OSError, ValueError, RuntimeError) as error:
            errors[str(path.relative_to(root))] = str(error)
    payload = {
        "domains": estimates,
        "serving": serving,
        "report_errors": errors,
        "probe_windows": [value.model_dump(exclude={"transfers"}, mode="json") for value in drains],
        "limitations": [
            "Each counter domain includes startup and drain; client TPS windows remain separate.",
            "Probe minimum rates are finite-window conservative assumptions, not sustained bandwidth proofs.",
            "Writer service is divided across three serving producers; Device copy service across same-Device peers.",
            "Host forward/capture/replay counters and native Graph structure bytes remain separate raw evidence.",
            "Payload-only capacity candidates require additional Host/snapshot/metadata and attachment-layout budgets.",
            "Protocol and attachment peak candidates are selected separately; missing controls remain unqualified.",
        ],
    }
    (root / "estimates.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines.extend(["", *payload["limitations"], "", "Report errors: " + json.dumps(errors)])
    (root / "summary.txt").write_text("\n".join(lines) + "\n")
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    }
    (root / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")


def main() -> None:
    """Run the user-owned batch or regenerate its offline report."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--devices", nargs=2, required=True)
    run_parser.add_argument("--seconds", type=float, default=7200)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        report(permitted(args.root))
    else:
        if not 600 <= args.seconds <= 7200:
            raise ValueError("batch duration must be within [600, 7200] seconds")
        try:
            batch(args.output, args.config, (args.devices[0], args.devices[1]), args.seconds)
        finally:
            if args.output.is_dir():
                report(args.output)


if __name__ == "__main__":
    main()
