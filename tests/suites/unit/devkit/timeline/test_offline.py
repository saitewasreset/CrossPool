import json
import struct
from pathlib import Path
from uuid import uuid4

import pytest

from xpool.config import TimelineDebugConfig
from xpool.devkit.timeline.format import RECORD, Record, write_chunk
from xpool.devkit.timeline.models import ProcessRole, Producer, ProducerRequest, Quality, SessionManifest
from xpool.devkit.timeline.offline import export, verify
from xpool.devkit.timeline.session import Session, publish_metadata


def archive(tmp_path: Path) -> Path:
    owner = Session(
        tmp_path, TimelineDebugConfig(enable=True, outdir=tmp_path).model_dump(mode="json"), ["daemon:host"]
    )
    producer = owner.register(
        ProducerRequest(startup_id=uuid4(), pid=1, create_time=1, role="daemon", slot="daemon", source="host")
    )
    payload = RECORD.pack(1, 100, 1, *([0] * 13)) + RECORD.pack(2, 200, 2, *([0] * 13))
    with memoryview(payload) as view:
        write_chunk(producer, 1, 1, 300, 400, view)
    quality = Quality(attempted=2, committed=2, exported=2, closed=True, unknown_tail=False)
    publish_metadata(producer.directory / "summary.json", quality.model_dump_json().encode(), 16384)
    owner.close()
    return owner.directory


def test_archive_is_movable_and_projection_keeps_clock_domain(tmp_path: Path) -> None:
    session = archive(tmp_path)
    moved = tmp_path / "moved"
    session.rename(moved)
    report = verify(moved)
    assert report.complete
    assert report.records == 2
    output = tmp_path / "derived"
    assert export(moved, output).complete
    payload = json.loads(next(output.glob("*.trace.json")).read_text())
    assert payload["metadata"]["clock_calibrated"] is False
    instants = [event for event in payload["traceEvents"] if event["ph"] == "I"]
    assert [event["args"]["raw_ns"] for event in instants] == ["100", "200"]
    with pytest.raises(ValueError, match="outside"):
        export(moved, moved / "derived")


def test_corrupt_chunk_is_rejected_before_export(tmp_path: Path) -> None:
    session = archive(tmp_path)
    path = next(session.glob("producer-*/*.bin"))
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        export(session, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_unknown_tail_and_partial_do_not_create_complete_evidence(tmp_path: Path) -> None:
    session = archive(tmp_path)
    summary = next(session.glob("producer-*/summary.json"))
    summary.unlink()
    summary.with_suffix(".partial").write_bytes(b"unfinished")
    report = verify(session)
    assert not report.complete
    assert {issue.reason for issue in report.issues} >= {"unknown_tail", "unpublished_partial"}
    assert report.records == 2


def test_duplicate_record_identity_is_not_silently_filtered(tmp_path: Path) -> None:
    session = archive(tmp_path)
    path = next(session.glob("producer-*/*.bin"))
    path.with_name("chunk-00000000000000000002.bin").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="duplicate Chunk"):
        verify(session)


@pytest.mark.parametrize("kind", [103, 104])
def test_either_missing_boundary_remains_unpaired(tmp_path: Path, kind: int) -> None:
    session = archive(tmp_path)
    manifest = SessionManifest.model_validate_json((session / "manifest.json").read_bytes())
    producer = Producer.model_validate(manifest.producers[0].model_dump())
    producer.directory = session / producer.directory
    next(producer.directory.glob("*.bin")).unlink()
    payload = RECORD.pack(1, 100, kind, 1, 11, 12, 1, 2, 3, *([0] * 7))
    with memoryview(payload) as view:
        write_chunk(producer, 1, 1, 300, 400, view)
    quality = Quality(attempted=2, committed=1, exported=1, dropped=1, closed=True, unknown_tail=False)
    publish_metadata(producer.directory / "summary.json", quality.model_dump_json().encode(), 16384)
    report = verify(session)
    assert {issue.reason for issue in report.issues} >= {"unpaired_boundary", "record_loss"}


def test_projection_preserves_large_integer_identity_and_local_duration(tmp_path: Path) -> None:
    session = archive(tmp_path)
    manifest = SessionManifest.model_validate_json((session / "manifest.json").read_bytes())
    producer = manifest.producers[0]
    producer.directory = session / producer.directory
    next(producer.directory.glob("*.bin")).unlink()
    base = 2**53 + 123
    payload = RECORD.pack(1, base, 103, 1, 2**63 + 1, 12, 1, 2, 3, *([0] * 7))
    payload += RECORD.pack(2, base + 2500, 104, 1, 2**63 + 1, 12, 1, 2, 3, *([0] * 7))
    with memoryview(payload) as view:
        write_chunk(producer, 1, 1, 300, 400, view)
    output = tmp_path / "derived"
    assert export(session, output).complete
    events = json.loads(next(output.glob("*.trace.json")).read_text())["traceEvents"]
    activity = next(event for event in events if event["ph"] == "X")
    assert activity["dur"] == 2.5
    assert activity["args"]["raw_ns"] == str(base)
    assert activity["args"]["generation_high"] == str(2**63 + 1)
    assert activity["args"]["end_raw_ns"] == str(base + 2500)


def test_host_phase_uses_the_real_serving_confirmation_boundary(tmp_path: Path) -> None:
    session = archive(tmp_path)
    manifest = SessionManifest.model_validate_json((session / "manifest.json").read_bytes())
    producer = manifest.producers[0]
    producer.directory = session / producer.directory
    next(producer.directory.glob("*.bin")).unlink()
    payload = b"".join(
        RECORD.pack(sequence, timestamp, kind, *([0] * 13))
        for sequence, timestamp, kind in ((1, 100, 1), (2, 150, 7), (3, 200, 2))
    )
    with memoryview(payload) as view:
        write_chunk(producer, 1, 1, 300, 400, view)
    quality = Quality(attempted=3, committed=3, exported=3, closed=True, unknown_tail=False)
    publish_metadata(producer.directory / "summary.json", quality.model_dump_json().encode(), 16384)
    output = tmp_path / "derived"
    assert export(session, output).complete
    trace = json.loads(next(output.glob("*.trace.json")).read_text())
    instants = [event for event in trace["traceEvents"] if event["ph"] == "I"]
    assert [event["args"]["phase"] for event in instants] == [
        "before_serving_confirmation",
        "serving_confirmed",
        "serving_confirmed",
    ]
    assert trace["metadata"]["serving_confirmation_host_ns"] == "150"


def test_payload_decoder_preserves_fields_and_explicit_abort() -> None:
    fields = (17, 2**63 + 19, 104, 23, 2**63 + 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 0)
    payload = RECORD.pack(*fields) + RECORD.pack(18, 79, 0, *([0] * 13))
    with memoryview(payload) as view:
        records = list(Record.iter_payload(view))
    record, aborted = records
    assert record == Record(
        sequence=17,
        timestamp=2**63 + 19,
        kind=104,
        site=23,
        generation_high=2**63 + 29,
        generation_low=31,
        endpoint_creator=37,
        endpoint_index=41,
        operation=43,
        instance=47,
        layer=53,
        lane=59,
        lease=61,
        result=67,
        rows=71,
        reserved=0,
    )
    assert aborted.sequence == 18
    assert aborted.kind == 0
    with pytest.raises(struct.error):
        list(Record.iter_payload(payload[:-1]))


def test_cross_device_reference_keeps_named_peer_identity_and_clock(tmp_path: Path) -> None:
    owner = Session(
        tmp_path,
        TimelineDebugConfig(enable=True, outdir=tmp_path).model_dump(mode="json"),
        ["atn:device", "ffn:device"],
    )
    sources: tuple[tuple[ProcessRole, str, str], ...] = (
        ("atnagent", "atn", "GPU-publisher"),
        ("ffnagent", "ffn", "GPU-observer"),
    )
    producers = [
        owner.register(
            ProducerRequest(
                startup_id=uuid4(),
                pid=index + 1,
                create_time=1,
                role=role,
                slot=slot,
                source="device",
                device_uuid=device,
            )
        )
        for index, (role, slot, device) in enumerate(sources)
    ]
    for producer, kind, timestamp in zip(producers, (101, 102), (1000, 7), strict=True):
        payload = RECORD.pack(1, timestamp, kind, 5, 11, 13, 17, 19, 23, *([0] * 7))
        with memoryview(payload) as view:
            write_chunk(producer, 1, 1, 300, 400, view)
        quality = Quality(attempted=1, committed=1, exported=1, closed=True, unknown_tail=False)
        publish_metadata(producer.directory / "summary.json", quality.model_dump_json().encode(), 16384)
    owner.close()
    output = tmp_path / "derived"
    assert export(owner.directory, output).complete
    traces = [json.loads(path.read_text()) for path in output.glob("*.trace.json")]
    assert len(traces) == 2
    events = [event for trace in traces for event in trace["traceEvents"] if event["ph"] == "I"]
    publication = next(event for event in events if event["args"]["kind"] == 101)
    assert publication["args"]["observation_event"] == f"{producers[1].producer_id}/1"
    assert publication["args"]["observation_clock_domain"] == producers[1].clock_domain
    assert all(event["ph"] != "X" for trace in traces for event in trace["traceEvents"])
