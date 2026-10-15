"""Real published files connect strict parsing, reconstruction and attachments."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from safetensors import safe_open
from tests.suites.unit.observability.fixtures import events_for, fixture, fixture_root, metadata_for, scenario

from xpool.observability import (
    ArtifactInputs,
    AttachmentIndex,
    ChunkHeader,
    ObservationError,
    RoutingAttachment,
    encode_chunk,
    parse_chunk,
    read_artifacts,
    reconstruct,
)


def write_inventory(root: Path, data: dict) -> ArtifactInputs:
    events = events_for(data)
    chunks: list[Path] = []
    metadata_paths: dict[int, Path] = {}
    metadata_bytes: dict[int, bytes] = {}
    for sequence, event in enumerate(events, 1):
        producer = event.source.identity.producer
        if producer not in metadata_paths:
            metadata_path = root / f"metadata.{producer}.1.json"
            metadata_paths[producer] = metadata_path
            metadata_bytes[producer] = event.metadata.model_dump_json().encode()
            metadata_path.write_bytes(metadata_bytes[producer])
        header = ChunkHeader(event.metadata.run_id, producer, sequence, 1, 0, 10000)
        chunk_path = root / f"{producer}.{sequence}.xpool"
        chunk_path.write_bytes(encode_chunk(header, (event.record,), metadata_bytes[producer]))
        chunks.append(chunk_path)
    if not events:
        path = root / "metadata.3.1.json"
        path.write_text(json.dumps(metadata_for(data)))
        metadata_paths[3] = path
    return ArtifactInputs(tuple(chunks), tuple(metadata_paths.values()))


@pytest.mark.parametrize("case", fixture("routing")["cases"], ids=lambda case: case["id"])
def test_routing_frozen_artifacts(tmp_path: Path, case: dict) -> None:
    data = copy.deepcopy(case["input"])
    inputs = write_inventory(tmp_path, data)
    target = tmp_path / "routing.safetensors"
    if case["id"] != "routing_missing":
        target.write_bytes((fixture_root / "routing.safetensors").read_bytes())
    entries = TypeAdapter(list[RoutingAttachment]).validate_python(data.get("attachments", []))
    index = AttachmentIndex(fixture("metadata")["run_id"], 5, tuple(entries), data.get("closing_index_complete", False))
    inputs = ArtifactInputs(inputs.chunks, inputs.metadata, attachment_indices=(index,), attachment_root=tmp_path)
    artifacts = read_artifacts(inputs)
    result = reconstruct(artifacts)
    expected = case["expected"]
    assert len(result.attachments) == 1
    evidence = result.attachments[0]
    assert evidence.available == expected.get("attachment_available", False)
    assert bool(evidence.sources) == expected.get("event_available", bool(data.get("events")))
    if "attachment_status" in expected:
        assert evidence.status == expected["attachment_status"]
    assert set(expected.get("quality", [])) <= result.quality.codes
    if evidence.available:
        with safe_open(target, framework="numpy") as handle:
            assert handle.get_tensor("records.0.topk_ids").tolist() == [[0, 1], [1, 0]]
            assert handle.get_tensor("records.0.topk_weights").tolist() == [[0.75, 0.25], [0.5, 0.5]]
    if not expected.get("event_available", True):
        assert not any(event.record.kind == 550 for event in result.events.values())


def test_published_files_out_of_order_duplicates_and_provenance(tmp_path: Path) -> None:
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    inputs = ArtifactInputs((*reversed(inputs.chunks), inputs.chunks[0]), inputs.metadata)
    result = reconstruct(read_artifacts(inputs))
    assert len(result.of_kind("operation")) == len(result.of_kind("invocation")) == 1
    assert "duplicate_event" in result.quality.codes
    assert any(len(sources) == 2 for sources in result.event_sources.values())
    assert all(
        source.path in inputs.chunks and source.offset == 128 for edge in result.edges for source in edge.sources
    )


@pytest.mark.parametrize("failure", ["checksum", "path_escape", "identity"])
def test_routing_integrity_failures_are_structured(tmp_path: Path, failure: str) -> None:
    data = copy.deepcopy(fixture("routing")["cases"][0]["input"])
    inputs = write_inventory(tmp_path, data)
    target = tmp_path / "routing.safetensors"
    target.write_bytes((fixture_root / "routing.safetensors").read_bytes())
    entry = data["attachments"][0]
    expected = "payload_checksum"
    if failure == "checksum":
        target.write_bytes(target.read_bytes() + b"corrupt")
    elif failure == "path_escape":
        entry["path"] = "../routing.safetensors"
        expected = "invalid_event"
    else:
        entry["lease_sequence"] += 1
        expected = "identity_conflict"
    index = AttachmentIndex(fixture("metadata")["run_id"], 5, (RoutingAttachment.model_validate(entry),), True)
    with pytest.raises(ObservationError) as caught:
        read_artifacts(
            ArtifactInputs(inputs.chunks, inputs.metadata, attachment_indices=(index,), attachment_root=tmp_path)
        )
    assert caught.value.code == expected


def test_quality_files_and_missing_required_metadata(tmp_path: Path) -> None:
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    quality_path = tmp_path / "quality.3.1.json"
    checkpoint = fixture("quality")["checkpoint"]
    checkpoint["counters"].update(attempted=6, committed=6, copied=6, exported=6, dropped_full=0, abandoned=0)
    checkpoint["last_attempted_sequence"] = 6
    quality_path.write_text(json.dumps(checkpoint))
    result = reconstruct(read_artifacts(ArtifactInputs(inputs.chunks, inputs.metadata, (quality_path,))))
    assert result.quality.checkpoints[(fixture("metadata")["run_id"], 3)].counters.attempted == 6
    with pytest.raises(ObservationError) as caught:
        read_artifacts(ArtifactInputs(inputs.chunks, ()))
    assert caught.value.code == "metadata_unavailable"


def test_original_artifacts_are_unchanged(tmp_path: Path) -> None:
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in (*inputs.chunks, *inputs.metadata)}
    reconstruct(read_artifacts(inputs))
    assert before == {path: hashlib.sha256(path.read_bytes()).digest() for path in before}


def test_io_errors_keep_the_original_cause_and_temporary_files_are_not_published(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ObservationError) as caught:
        read_artifacts(ArtifactInputs((), (missing,)))
    assert caught.value.path == missing
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    temporary = tmp_path / "1.xpool.tmp"
    temporary.write_bytes(inputs.chunks[0].read_bytes())
    with pytest.raises(ObservationError) as caught:
        read_artifacts(ArtifactInputs((temporary,), inputs.metadata))
    assert caught.value.code == "invalid_layout"


def test_conflicting_chunk_publication_is_not_deduplicated(tmp_path: Path) -> None:
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    duplicate = tmp_path / "conflicting.xpool"
    original = read_artifacts(inputs).chunks[0]
    producer = original.header.producer
    metadata_path = next(path for path in inputs.metadata if path.name == f"metadata.{producer}.1.json")
    metadata = metadata_path.read_bytes()
    changed = original.events[0].record.model_copy(update={"raw_time_ns": 11})
    duplicate.write_bytes(encode_chunk(original.header, (changed,), metadata))
    assert parse_chunk(duplicate.read_bytes(), metadata).events[0].record.raw_time_ns == 11
    with pytest.raises(ObservationError) as caught:
        read_artifacts(ArtifactInputs((*inputs.chunks, duplicate), inputs.metadata))
    assert caught.value.code == "identity_conflict"
