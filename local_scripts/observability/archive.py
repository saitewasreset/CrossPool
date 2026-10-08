"""Byte-preserving archival with explicit missing-file accounting."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from local_scripts.observability.models import Manifest, RunContext, SourceFile


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_manifest(path: Path) -> Manifest:
    try:
        return Manifest.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid archive manifest {path}: {error}") from error


def verify_archive(manifest: Manifest, root: Path) -> None:
    """Require the recorded bytes, including evidence files, to remain unchanged."""

    for source in manifest.files:
        path = root / source.path
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"archive input escapes root: {path}")
        if not path.is_file() or path.stat().st_size != source.size or digest(path) != source.sha256:
            raise ValueError(f"archive input missing or checksum mismatch: {path}")


def archive_attempt(source: Path, output: Path, context: RunContext) -> Path:
    """Copy a retired Harness Attempt, never overwriting an existing archive.

    The caller must wait for managed-resource retirement before invoking this.
    Failure leaves a partial directory for diagnosis, without a final Manifest.
    """

    if not source.is_dir():
        raise ValueError(f"attempt directory does not exist: {source}")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("archive output must be outside the source Attempt")
    output.mkdir(parents=True, exist_ok=False)
    started = timestamp()
    inventory: list[SourceFile] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing symlink in Attempt: {path}")
        if not path.is_file():
            continue
        # Checkpoints and numerical tensors are not prototype evidence.
        if path.suffix not in {".json", ".jsonl", ".toml", ".log", ".txt", ".tmp"}:
            continue
        name = path.name
        kind = "evidence"
        if name.startswith("xpool.fabric-observer.") and path.suffix == ".json":
            kind = "fabric"
        elif name.startswith("xpool.transport-observer.") and path.suffix == ".json":
            kind = "transport"
        elif name.startswith("xpool.graph-observer."):
            kind = "graph_events" if path.suffix == ".jsonl" else "graph" if path.suffix == ".json" else "evidence"
        relative = Path("raw") / path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = digest(path)
        with path.open("rb") as reader, destination.open("xb") as writer:
            while block := reader.read(1024 * 1024):
                writer.write(block)
        after = digest(destination)
        if before != after or before != digest(path):
            raise ValueError(f"input changed while archiving: {path}")
        inventory.append(SourceFile(path=relative.as_posix(), kind=kind, size=destination.stat().st_size, sha256=after))
    counts = {
        kind: sum(item.kind == kind for item in inventory) for kind in ("fabric", "transport", "graph", "graph_events")
    }
    expected = {"fabric": 2, "transport": 2, "graph": 1, "graph_events": 1}
    missing = [
        f"{kind}: expected {count}, found {counts[kind]}" for kind, count in expected.items() if counts[kind] < count
    ]
    manifest = Manifest(
        context=context,
        archive_started_at=started,
        archive_finished_at=timestamp(),
        files=inventory,
        missing=missing,
    )
    verify_archive(manifest, output)
    path = output / "run_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path
