"""Read-only validation and per-clock-domain Perfetto projection of raw evidence."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xpool.devkit.timeline.format import read_chunk
from xpool.devkit.timeline.models import Quality, Reason, SessionManifest
from xpool.devkit.timeline.session import MANIFEST_LIMIT, PRODUCER_METADATA_LIMIT

__all__ = ["Report", "export", "verify"]

NAMES = {
    1: "producer.initialized",
    2: "producer.stopped",
    3: "transport.endpoint_opened",
    4: "transport.endpoint_closing",
    5: "fabric.joined",
    6: "fabric.closing",
    7: "serving.health_confirmed",
    100: "transport.staging_started",
    101: "transport.request_published",
    102: "transport.request_observed",
    103: "transport.execution_started",
    104: "transport.execution_completed",
    105: "transport.result_published",
    106: "transport.result_observed",
    107: "transport.output_copied",
    108: "transport.closed",
    200: "fabric.lane_observed",
    201: "fabric.compute_dispatch_started",
    202: "fabric.compute_completed",
    203: "fabric.lane_scheduled",
    204: "fabric.lane_released",
}
PAIRS = {
    100: (107, "transport.device_round_trip"),
    103: (104, "transport.fabric_execution"),
    201: (202, "fabric.compute_protocol_bracket"),
    203: (204, "fabric.lane_lease"),
}


class Issue(BaseModel):
    """Localized quality reason; missing endpoints never acquire invented times."""

    model_config = ConfigDict(extra="forbid")
    reason: str
    producer_id: int | None = None
    sequence: int | None = None
    detail: str = ""


class Report(BaseModel):
    """Structural validity is separate from loss-free declared coverage."""

    model_config = ConfigDict(extra="forbid")
    session_id: str
    clock_alignment: Literal["unavailable"] = "unavailable"
    records: int = 0
    chunks: int = 0
    bytes: int = 0
    issues: list[Issue] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Completeness applies only to the capabilities and window in the manifest."""
        return not self.issues


def populate(session: Path, database: sqlite3.Connection) -> tuple[SessionManifest, Report]:
    """Stream bounded Chunks into an offline disk index, preserving exact integers."""
    if (session / "manifest.json").stat().st_size > MANIFEST_LIMIT:
        raise ValueError("timeline manifest exceeds metadata limit")
    database.row_factory = sqlite3.Row
    manifest = SessionManifest.model_validate_json((session / "manifest.json").read_bytes())
    report = Report(session_id=str(manifest.session_id))
    database.execute(
        "CREATE TABLE event (producer INTEGER, sequence INTEGER, timestamp INTEGER, kind INTEGER, "
        "domain TEXT, track TEXT, operation TEXT, raw TEXT, source TEXT, PRIMARY KEY(producer,sequence))"
    )
    database.execute("CREATE INDEX boundary ON event(producer,operation,kind)")
    database.execute("CREATE INDEX relation ON event(operation,kind)")
    known_slots = {f"{p.slot}:{p.source}" for p in manifest.producers}
    for slot in sorted(set(manifest.expected_slots) - known_slots):
        report.issues.append(Issue(reason=Reason.MISSING_PRODUCER, detail=slot))
    if not manifest.closed:
        report.issues.append(Issue(reason=Reason.ABNORMAL_EXIT, detail="Session has no closing manifest"))
    for producer in manifest.producers:
        if producer.session_id != manifest.session_id or producer.machine_id != manifest.machine_id:
            raise ValueError(f"timeline manifest Producer identity conflict: {producer.producer_id}")
        if producer.directory.is_absolute() or len(producer.directory.parts) != 1:
            raise ValueError("timeline manifest Producer directory must be one relative component")
        directory = session / producer.directory
        if directory.resolve().parent != session.resolve() or directory.is_symlink():
            raise ValueError("timeline Producer directory escapes its Session")
        summary_path = directory / "summary.json"
        quality: Quality | None = None
        if summary_path.exists():
            if summary_path.is_symlink() or summary_path.stat().st_size > PRODUCER_METADATA_LIMIT:
                raise ValueError(f"timeline summary exceeds ownership or metadata limit: {summary_path}")
            quality = Quality.model_validate_json(summary_path.read_bytes())
            if quality.dropped or quality.lost_after_commit:
                report.issues.append(Issue(reason=Reason.RECORD_LOSS, producer_id=producer.producer_id))
            for reason in sorted(quality.reasons):
                report.issues.append(Issue(reason=reason, producer_id=producer.producer_id))
        if quality is None or not quality.closed or quality.unknown_tail:
            report.issues.append(Issue(reason=Reason.UNKNOWN_TAIL, producer_id=producer.producer_id))
        if list(directory.glob("*.partial")):
            report.issues.append(Issue(reason="unpublished_partial", producer_id=producer.producer_id))
        expected_chunk = 1
        semantic_count = 0
        actual_sequences: set[int] = set()
        # This set only tracks Chunk identities, bounded by Session quota / minimum
        # file header size. Event identity uniqueness is enforced by the disk index.
        for path in sorted(directory.glob("chunk-*.bin")):
            if path.is_symlink():
                raise ValueError(f"timeline Chunk must be a regular owned file: {path}")
            header, records = read_chunk(path, producer, int(manifest.config["chunk_bytes"] or 0))
            if header.sequence in actual_sequences:
                raise ValueError(f"timeline duplicate Chunk identity: {path}")
            actual_sequences.add(header.sequence)
            if header.sequence != expected_chunk:
                report.issues.append(
                    Issue(reason="missing_chunk", producer_id=producer.producer_id, sequence=expected_chunk)
                )
            expected_chunk = header.sequence + 1
            report.chunks += 1
            report.bytes += path.stat().st_size
            for record in records:
                if record.kind and record.kind not in NAMES:
                    raise ValueError(f"timeline unknown event kind={record.kind} source={path}")
                if record.kind >= 100 and not (record.generation_high or record.generation_low):
                    raise ValueError(f"timeline data event lacks generation: {path}/{record.sequence}")
                if 100 <= record.kind < 200 and not (
                    record.endpoint_creator and record.endpoint_index and (record.operation or record.kind == 108)
                ):
                    raise ValueError(f"timeline Transport event lacks shared identity: {path}/{record.sequence}")
                track = (
                    f"{producer.slot}:{record.site}"
                    if record.kind < 200
                    else f"{producer.slot}:PE{record.site}:lane{record.lane}"
                )
                if 100 <= record.kind < 200:
                    track = (
                        f"{producer.slot}:endpoint{record.endpoint_creator}/{record.endpoint_index}:site{record.site}"
                    )
                operation = (
                    f"{record.generation_high}:{record.generation_low}:{record.endpoint_creator}:"
                    f"{record.endpoint_index}:{record.operation}:{record.instance}"
                )
                if record.kind >= 200:
                    operation += f":{record.lane}:{record.lease}"
                raw = json.dumps(asdict(record))
                try:
                    database.execute(
                        "INSERT INTO event VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            producer.producer_id,
                            record.sequence,
                            record.timestamp,
                            record.kind,
                            producer.clock_domain,
                            track,
                            operation,
                            raw,
                            f"{producer.directory}/{path.name}",
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"timeline duplicate record identity: {path}/{record.sequence}") from error
                if record.kind:
                    semantic_count += 1
                    report.records += 1
        if quality is not None and quality.closed:
            if semantic_count != quality.exported:
                raise ValueError(f"timeline published count mismatch producer={producer.producer_id}")
            if quality.committed != quality.exported + quality.lost_after_commit:
                raise ValueError(f"timeline committed count mismatch producer={producer.producer_id}")
            if quality.attempted != quality.committed + quality.dropped:
                raise ValueError(f"timeline attempted count mismatch producer={producer.producer_id}")
        previous = 0
        for (sequence,) in database.execute(
            "SELECT sequence FROM event WHERE producer=? ORDER BY sequence", (producer.producer_id,)
        ):
            if sequence != previous + 1:
                report.issues.append(
                    Issue(reason=Reason.RECORD_LOSS, producer_id=producer.producer_id, sequence=previous + 1)
                )
            previous = sequence
        if quality is not None and previous < quality.attempted:
            report.issues.append(
                Issue(reason=Reason.RECORD_LOSS, producer_id=producer.producer_id, sequence=previous + 1)
            )
    if report.bytes + int(manifest.config["metadata_reserve_bytes"] or 0) > int(
        manifest.config["session_max_bytes"] or 0
    ):
        raise ValueError("timeline published files exceed Session quota")
    database.commit()
    # Pair only same-Producer true boundaries. Cross-Producer relations are
    # exported as explicit identity references and optional same-domain flows.
    for begin, (end, _) in PAIRS.items():
        for producer, sequence, timestamp, operation in database.execute(
            "SELECT producer,sequence,timestamp,operation FROM event WHERE kind=?", (begin,)
        ):
            endpoints = database.execute(
                "SELECT timestamp FROM event WHERE producer=? AND operation=? AND kind=?", (producer, operation, end)
            ).fetchall()
            if len(endpoints) != 1:
                report.issues.append(Issue(reason=Reason.UNPAIRED_BOUNDARY, producer_id=producer, sequence=sequence))
            elif endpoints[0]["timestamp"] < timestamp:
                raise ValueError(f"timeline reversed same-domain boundary producer={producer} sequence={sequence}")
        for producer, sequence, operation in database.execute(
            "SELECT producer,sequence,operation FROM event WHERE kind=?", (end,)
        ):
            starts = database.execute(
                "SELECT COUNT(*) AS starts FROM event WHERE producer=? AND operation=? AND kind=?",
                (producer, operation, begin),
            ).fetchone()["starts"]
            if starts != 1:
                report.issues.append(Issue(reason=Reason.UNPAIRED_BOUNDARY, producer_id=producer, sequence=sequence))
    return manifest, report


def verify(session: Path) -> Report:
    """Verify published evidence without writing to the Session or invoking CUDA."""
    with tempfile.TemporaryDirectory(prefix="xpool-timeline-") as temporary:
        with sqlite3.connect(Path(temporary) / "index.sqlite") as database:
            _, report = populate(session, database)
            return report


def export(session: Path, output: Path) -> Report:
    """Project each declared clock domain separately, retaining raw endpoints."""
    if output.resolve().is_relative_to(session.resolve()):
        raise ValueError("timeline export output must be outside the Session")
    with tempfile.TemporaryDirectory(prefix="xpool-timeline-") as temporary:
        with sqlite3.connect(Path(temporary) / "index.sqlite") as database:
            manifest, report = populate(session, database)
            output.mkdir(parents=True, exist_ok=False)
            domains = database.execute("SELECT DISTINCT domain FROM event WHERE kind!=0 ORDER BY domain").fetchall()
            for index, (domain,) in enumerate(domains):
                serving_edges = database.execute(
                    "SELECT timestamp FROM event WHERE domain=? AND kind=7", (domain,)
                ).fetchall()
                if len(serving_edges) > 1:
                    raise ValueError("timeline has duplicate serving confirmation boundaries")
                serving_begin = serving_edges[0]["timestamp"] if serving_edges else None
                host_domain = domain.endswith(":CLOCK_MONOTONIC")
                origin = database.execute(
                    "SELECT MIN(timestamp) AS origin FROM event WHERE domain=? AND kind!=0", (domain,)
                ).fetchone()["origin"]
                tracks = {
                    track: ordinal + 1
                    for ordinal, (track,) in enumerate(
                        database.execute(
                            "SELECT DISTINCT track FROM event WHERE domain=? AND kind!=0 ORDER BY track", (domain,)
                        )
                    )
                }
                path = output / f"domain-{index}.trace.json"
                with path.open("x", encoding="utf-8") as stream:
                    stream.write('{"traceEvents":[')
                    separator = ""

                    def emit(event: dict[str, object]) -> None:
                        nonlocal separator
                        stream.write(separator + json.dumps(event))
                        separator = ","

                    for track, tid in tracks.items():
                        emit({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": {"name": track}})
                    rows = database.execute(
                        "SELECT producer,sequence,timestamp,kind,track,operation,raw,source FROM event "
                        "WHERE domain=? AND kind!=0 ORDER BY timestamp,producer,sequence",
                        (domain,),
                    )
                    for producer, sequence, timestamp, kind, track, operation, raw, source in rows:
                        args = {
                            "producer_id": str(producer),
                            "record_sequence": str(sequence),
                            "raw_ns": str(timestamp),
                            "source": source,
                            **{
                                key: str(value) if abs(value) > 2**53 else value
                                for key, value in json.loads(raw).items()
                            },
                        }
                        if host_domain:
                            args["phase"] = (
                                "serving_confirmed"
                                if serving_begin is not None and timestamp >= serving_begin
                                else "before_serving_confirmation"
                            )
                        if kind in (101, 105):
                            target = 102 if kind == 101 else 106
                            peers = database.execute(
                                "SELECT producer,sequence,domain FROM event WHERE operation=? AND kind=?",
                                (operation, target),
                            ).fetchall()
                            if len(peers) == 1:
                                args["observation_event"] = f"{peers[0]['producer']}/{peers[0]['sequence']}"
                                args["observation_clock_domain"] = peers[0]["domain"]
                        emit(
                            {
                                "name": NAMES[kind],
                                "ph": "I",
                                "s": "t",
                                "cat": "timeline",
                                "pid": 1,
                                "tid": tracks[track],
                                "ts": (timestamp - origin) / 1000,
                                "args": args,
                            }
                        )
                        if kind in PAIRS:
                            end, name = PAIRS[kind]
                            endpoints = database.execute(
                                "SELECT sequence,timestamp FROM event WHERE producer=? AND operation=? AND kind=?",
                                (producer, operation, end),
                            ).fetchall()
                            if len(endpoints) == 1:
                                # Activity rows remain independent of the Instant
                                # row, so a protocol bracket never hides its facts.
                                activity_tid = tracks[track] + len(tracks) * (kind + 1)
                                emit(
                                    {
                                        "name": "thread_name",
                                        "ph": "M",
                                        "pid": 1,
                                        "tid": activity_tid,
                                        "args": {"name": f"{track} {name}"},
                                    }
                                )
                                emit(
                                    {
                                        "name": name,
                                        "ph": "X",
                                        "cat": "timeline.local",
                                        "pid": 1,
                                        "tid": activity_tid,
                                        "ts": (timestamp - origin) / 1000,
                                        "dur": (endpoints[0]["timestamp"] - timestamp) / 1000,
                                        "args": {
                                            **args,
                                            "end_record_sequence": str(endpoints[0]["sequence"]),
                                            "end_raw_ns": str(endpoints[0]["timestamp"]),
                                        },
                                    }
                                )
                    stream.write(
                        '],"metadata":'
                        + json.dumps(
                            {
                                "session_id": str(manifest.session_id),
                                "clock_domain": domain,
                                "origin_ns": str(origin),
                                "timestamp_unit": "us",
                                "clock_calibrated": False,
                                "quality": "complete" if report.complete else "degraded",
                                "serving_confirmation_host_ns": str(serving_begin)
                                if serving_begin is not None
                                else None,
                                "device_phase_alignment": "unavailable" if not host_domain else "not_applicable",
                            }
                        )
                        + "}"
                    )
            (output / "quality.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return report
