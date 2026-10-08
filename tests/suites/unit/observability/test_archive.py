from pathlib import Path

import pytest
from local_scripts.observability.archive import archive_attempt, load_manifest, verify_archive
from local_scripts.observability.models import Participant, RunContext, SourceFile


def test_archive_preserves_bytes_and_reports_missing(tmp_path: Path) -> None:
    source = tmp_path / "attempt"
    source.mkdir()
    data = b'{"records": []}\n\n'
    (source / "xpool.fabric-observer.sample.0.json").write_bytes(data)
    output = tmp_path / "archive"
    manifest = load_manifest(archive_attempt(source, output, RunContext(run_id="r", attempt=1)))
    assert (output / manifest.files[0].path).read_bytes() == data
    assert manifest.missing
    assert manifest.snapshot_capture_time is None
    verify_archive(manifest, output)
    (output / manifest.files[0].path).write_bytes(data + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_archive(manifest, output)


def test_attempt_isolation_and_no_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = load_manifest(archive_attempt(source, tmp_path / "one", RunContext(run_id="r", attempt=1)))
    second = load_manifest(archive_attempt(source, tmp_path / "two", RunContext(run_id="r", attempt=2)))
    assert first.context.attempt != second.context.attempt
    with pytest.raises(FileExistsError):
        archive_attempt(source, tmp_path / "one", first.context)


def test_metadata_requires_provenance_and_safe_paths() -> None:
    with pytest.raises(ValueError, match="evidence"):
        Participant(pe=0, device_uuid="GPU-example")
    with pytest.raises(ValueError, match="contained"):
        SourceFile(path="../escape", kind="fabric", size=0, sha256="a" * 64)
    assert Participant(pe=0).device_uuid is None
    with pytest.raises(ValueError, match="duplicate PE"):
        RunContext(run_id="r", attempt=1, participants=[Participant(pe=0), Participant(pe=0)])


def test_archive_rejects_symlink_and_nested_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="outside"):
        archive_attempt(source, source / "nested", RunContext(run_id="r", attempt=1))
    (source / "link.json").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlink"):
        archive_attempt(source, tmp_path / "archive", RunContext(run_id="r", attempt=1))
