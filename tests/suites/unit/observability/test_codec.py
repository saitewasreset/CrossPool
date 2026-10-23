"""Independent frozen byte vectors exercise the public wire boundary."""

import hashlib
import json
import struct
import zlib
from dataclasses import replace
from pathlib import Path

import pytest
from tests.suites.unit.observability.fixtures import events_for, scenario

from xpool.observability.codec import encode_chunk, parse_chunk, validate_event
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import Metadata, QualityCheckpoint
from xpool.observability.wire import PresenceBit

fixture_root = Path(__file__).resolve().parents[4] / "docs/plans/unified-timeline-observability/fixtures"
vectors = json.loads((fixture_root / "encoding.json").read_bytes())["vectors"]
checksums = json.loads((fixture_root / "checksums.json").read_bytes())["files"]


@pytest.mark.parametrize("entry", checksums, ids=lambda entry: entry["path"])
def test_frozen_fixture_integrity(entry: dict) -> None:
    data = (fixture_root / entry["path"]).read_bytes()
    assert len(data) == entry["bytes"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("version", [1, 3])
@pytest.mark.parametrize("model", [Metadata, QualityCheckpoint])
def test_artifact_version_is_rejected_before_schema_validation(
    model: type[Metadata] | type[QualityCheckpoint], version: int
) -> None:
    path = Path("artifact.json")
    with pytest.raises(ObservationError) as caught:
        model.parse_artifact(json.dumps({"contract_version": version}).encode(), path, ErrorCode.INVALID_EVENT)
    assert caught.value.code == ErrorCode.UNSUPPORTED_VERSION
    assert caught.value.path == path


@pytest.mark.parametrize("version", [1, 3])
def test_chunk_version_is_rejected_before_checksums(version: int) -> None:
    data = bytearray.fromhex(vectors[0]["hex"])
    struct.pack_into("<H", data, 8, version)
    with pytest.raises(ObservationError) as caught:
        parse_chunk(bytes(data), None)
    assert caught.value.code == ErrorCode.UNSUPPORTED_VERSION
    assert caught.value.offset == 8


@pytest.mark.parametrize("vector", vectors, ids=[item["id"] for item in vectors])
def test_frozen_encoding(vector: dict) -> None:
    data = bytes.fromhex(vector["hex"])
    metadata = (fixture_root / "metadata.json").read_bytes()
    expected = vector["expected"]
    if "error" in expected:
        with pytest.raises(ObservationError) as caught:
            parse_chunk(data, metadata)
        assert caught.value.code == expected["error"]
        assert caught.value.path == Path("<memory>")
    else:
        chunk = parse_chunk(data, metadata)
        assert chunk.header.producer == expected["producer"]
        assert chunk.events[0].record.model_dump() == expected["event"]
        assert encode_chunk(chunk.header, tuple(event.record for event in chunk.events), metadata) == data


@pytest.mark.parametrize(
    "offset,value,code", [(184, 3, "invalid_event"), (204, 99, "invalid_event"), (16, 99, "invalid_event")]
)
def test_valid_crcs_do_not_hide_invalid_semantics(offset: int, value: int, code: str) -> None:
    data = bytearray.fromhex(vectors[0]["hex"])
    struct.pack_into("<I", data, 128 + offset, value)
    struct.pack_into("<I", data, 88, zlib.crc32(data[128:]))
    struct.pack_into("<I", data, 92, 0)
    struct.pack_into("<I", data, 92, zlib.crc32(data[:128]))
    with pytest.raises(ObservationError) as caught:
        parse_chunk(bytes(data), (fixture_root / "metadata.json").read_bytes())
    assert caught.value.code == code
    assert caught.value.producer == 3
    assert caught.value.event_sequence == 1


@pytest.mark.parametrize(
    "mutation,code",
    [
        (b'{"contract_version":1}', "unsupported_version"),
        (b'{"contract_version":2,"contract_version":2}', "metadata_unavailable"),
        (b'{"contract_version":2,"configuration":{"bad":NaN}}', "metadata_unavailable"),
        (b'{"contract_version":2,"configuration":{"bad":1e400}}', "metadata_unavailable"),
    ],
)
def test_json_boundary_rejects_versions_duplicates_and_nonfinite(mutation: bytes, code: str) -> None:
    with pytest.raises(ObservationError) as caught:
        Metadata.parse_artifact(mutation, Path("metadata.json"), ErrorCode.METADATA_UNAVAILABLE)
    assert caught.value.code == code


def test_integrity_stage_precedence() -> None:
    data = bytearray.fromhex(vectors[0]["hex"])
    data[128] ^= 1
    # Both the metadata and payload are corrupt; metadata fails first.
    with pytest.raises(ObservationError) as caught:
        parse_chunk(bytes(data), b"corrupted metadata")
    assert caught.value.code == "metadata_checksum"


@pytest.mark.parametrize("bit", list(PresenceBit))
def test_presence_queries_distinguish_each_wire_group(bit: PresenceBit) -> None:
    chunk = parse_chunk(bytes.fromhex(vectors[0]["hex"]), (fixture_root / "metadata.json").read_bytes())
    record = chunk.events[0].record.model_copy(update={"presence": 1 << bit})
    assert record.has(bit)
    assert not any(record.has(other) for other in PresenceBit if other != bit)


@pytest.mark.parametrize(
    "dependent,prerequisite,detail",
    [
        (PresenceBit.OPERATION, PresenceBit.ENDPOINT, "Operation requires Endpoint"),
        (PresenceBit.INVOCATION, PresenceBit.GENERATION, "Invocation/Lease requires Generation"),
        (PresenceBit.LEASE, PresenceBit.GENERATION, "Invocation/Lease requires Generation"),
        (PresenceBit.REQUEST_FACTS, PresenceBit.INVOCATION_FACTS, "request facts require invocation facts presence"),
    ],
)
def test_presence_dependencies_are_rejected(dependent: PresenceBit, prerequisite: PresenceBit, detail: str) -> None:
    event = next(event for event in events_for(scenario("normal_dense")) if event.record.has(dependent))
    record = event.record.model_copy(update={"presence": event.record.presence & ~(1 << prerequisite)})
    with pytest.raises(ObservationError, match=detail) as caught:
        validate_event(replace(event, record=record))
    assert caught.value.code == ErrorCode.INVALID_EVENT
    assert caught.value.event_sequence == record.event_sequence


@pytest.mark.parametrize("unknown_bit", [18, 31])
def test_unknown_presence_bits_are_rejected(unknown_bit: int) -> None:
    event = events_for(scenario("normal_dense"))[0]
    record = event.record.model_copy(update={"presence": event.record.presence | (1 << unknown_bit)})
    with pytest.raises(ObservationError, match="unknown presence bits") as caught:
        validate_event(replace(event, record=record))
    assert caught.value.code == ErrorCode.INVALID_EVENT


def test_graph_owner_remains_present_via_execution_group() -> None:
    event = next(
        event for event in events_for(scenario("graph_replay")) if event.record.has(PresenceBit.GRAPH_EXECUTION)
    )
    record = event.record.model_copy(
        update={"presence": event.record.presence & ~(1 << PresenceBit.GRAPH_TEMPLATE), "graph_template": 0}
    )
    assert record.graph_owner > 0
    validate_event(replace(event, record=record))


@pytest.mark.parametrize(
    "field,value,group",
    [
        ("forward_mode", 3, PresenceBit.REQUEST_FACTS),
        ("output_requirement", 0, PresenceBit.INVOCATION_FACTS),
        ("dp_row_layout", 3, PresenceBit.REQUEST_FACTS),
        ("delivery", 0, PresenceBit.DELIVERY),
        ("result_code", 4, PresenceBit.RESULT_CODE),
        ("role", 0, PresenceBit.LOCAL_TRACE_ID),
    ],
)
def test_unknown_scalar_enums_are_rejected(field: str, value: int, group: PresenceBit) -> None:
    event = next(event for event in events_for(scenario("normal_dense")) if event.record.has(group))
    with pytest.raises(ObservationError) as caught:
        validate_event(replace(event, record=event.record.model_copy(update={field: value})))
    assert caught.value.code == ErrorCode.INVALID_EVENT


@pytest.mark.parametrize("field,value", [("data0", 4), ("data1", 0), ("data4", 6)])
def test_unknown_graph_payload_enums_are_rejected(field: str, value: int) -> None:
    event = next(
        event for event in events_for(scenario("graph_replay")) if event.record.has(PresenceBit.GRAPH_EXECUTION)
    )
    with pytest.raises(ObservationError, match="invalid Graph payload") as caught:
        validate_event(replace(event, record=event.record.model_copy(update={field: value})))
    assert caught.value.code == ErrorCode.INVALID_EVENT
