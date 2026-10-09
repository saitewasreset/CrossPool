"""Offline Fabric correlation; original snapshots remain authoritative."""

import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from heapq import heappop, heappush
from pathlib import Path
from typing import Literal, TypedDict

from local_scripts.observability.archive import load_manifest, verify_archive
from local_scripts.observability.models import Manifest, QualityIssue, QualityReport
from pydantic import BaseModel, JsonValue, TypeAdapter

import xpool.native
from xpool.devkit.fabric_observer import FabricRecordJson, FabricSnapshotJson
from xpool.devkit.graph_observer import FfnGraphObserverSnapshot
from xpool.devkit.transport_observer import TransportSnapshotJson
from xpool.fabric import FabricGenerationId
from xpool.integrations.sglang.devkit import SglangGraphEvent

ROLE_EVENTS = {
    "atnagent": xpool.native.devkit.fabric_observer.AtnAgentEvent,
    "coordinator": xpool.native.devkit.fabric_observer.CoordinatorEvent,
    "ffnagent": xpool.native.devkit.fabric_observer.FfnAgentEvent,
}
REQUIRED_EVENTS = {
    "atnagent": (
        "submission_prepared",
        "submission_published",
        "admission_observed",
        "input_ready_published",
        "output_commit_observed",
        "output_acknowledgement_published",
    ),
    "coordinator": (
        "enqueued",
        "scheduled",
        "admission_published",
        "lane_execution_published",
        "ffnagent_completions_observed",
        "output_commit_published",
        "output_acknowledgements_observed",
        "lane_released",
    ),
    "ffnagent": (
        "lane_execution_observed",
        "input_ready_observed",
        "compute_started",
        "compute_completed",
        "completion_published",
    ),
}
INTERVALS = {
    "atnagent": (
        ("admission_wait", "submission_published", "admission_observed"),
        ("output_wait", "admission_observed", "output_commit_observed"),
        ("total", "submission_prepared", "output_acknowledgement_published"),
    ),
    "coordinator": (
        ("scheduling", "enqueued", "scheduled"),
        ("execution", "lane_execution_published", "ffnagent_completions_observed"),
        ("active_total", "scheduled", "lane_released"),
        ("total", "enqueued", "lane_released"),
    ),
    "ffnagent": (
        ("input_wait", "lane_execution_observed", "input_ready_observed"),
        ("compute", "compute_started", "compute_completed"),
        ("total", "lane_execution_observed", "completion_published"),
    ),
}


class Observation(BaseModel):
    """One source-local record with an invocation key that excludes constraints."""

    identity: str
    source: str
    index: int
    generation: str
    pe: int
    device_uuid: str | None
    record: FabricRecordJson
    invocation: str

    def event_id(self, name: str) -> str:
        return f"{self.identity}/{name}"


class Edge(BaseModel):
    """Identity-backed protocol dependency, without cross-domain duration."""

    invocation: str
    source_event: str
    target_event: str
    kind: Literal["observation", "aggregate_condition"]
    participant_set: list[int]


class Interval(BaseModel):
    """A valid local interval retaining both original event references."""

    invocation: str
    record_id: str
    name: str
    start_event: str
    end_event: str
    start_ns: int
    end_ns: int
    device_uuid: str


class Timeline(BaseModel):
    manifest: Manifest
    snapshots: dict[str, FabricSnapshotJson]
    observations: list[Observation]
    intervals: list[Interval]
    edges: list[Edge]
    quality: QualityReport


def invocation_key(manifest: Manifest, generation: str, record: FabricRecordJson) -> str:
    return json.dumps(
        [
            manifest.context.run_id,
            manifest.context.attempt,
            generation,
            record["instance_index"],
            record["invocation_sequence"],
        ],
        separators=(",", ":"),
    )


def read_timeline(manifest_path: Path) -> Timeline:
    """Validate every archived observer before constructing any derived output."""

    manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    verify_archive(manifest, root)
    issues = [QualityIssue(kind="missing_artifact", detail=missing) for missing in manifest.missing]
    issues.extend(
        QualityIssue(kind="metadata_unavailable", detail=detail) for detail in manifest.context.metadata_issues
    )
    if manifest.context.outcome != "passed":
        issues.append(QualityIssue(kind="abnormal_or_unknown_exit", detail=manifest.context.outcome))
    snapshots: dict[str, FabricSnapshotJson] = {}
    observations: list[Observation] = []
    mapping = {participant.pe: participant.device_uuid for participant in manifest.context.participants}
    graph_mapping: dict[tuple[str, int], str] = {}
    for source in manifest.files:
        path = root / source.path
        try:
            match source.kind:
                case "fabric":
                    snapshot = TypeAdapter(FabricSnapshotJson).validate_json(
                        path.read_bytes(), strict=True, extra="allow"
                    )
                    FabricGenerationId.parse(snapshot["generation"])
                    # This disposable prototype qualifies precisely the accepted two-device case.
                    if (snapshot["atnagent_count"], snapshot["ffnagent_count"], snapshot["model_topologies"]) != (
                        1,
                        1,
                        [{"atn_tp_size": 1, "atn_dp_size": 1}],
                    ):
                        raise ValueError("prototype requires the serving-001 topology")
                    if snapshot["pe"] not in {0, 1} or snapshot["sequence"] < 0 or snapshot["dropped"] < 0:
                        raise ValueError("invalid snapshot PE, sequence or dropped count")
                    if any(
                        (previous["generation"], previous["pe"]) == (snapshot["generation"], snapshot["pe"])
                        for previous in snapshots.values()
                    ):
                        raise ValueError("duplicate Generation/PE snapshot")
                    snapshots[source.path] = snapshot
                case "transport":
                    transport = TypeAdapter(TransportSnapshotJson).validate_json(
                        path.read_bytes(), strict=True, extra="allow"
                    )
                    if transport["dropped"] < 0:
                        raise ValueError("negative Transport dropped count")
                    if transport["dropped"]:
                        issues.append(
                            QualityIssue(
                                kind="snapshot_loss",
                                source=source.path,
                                detail=f"Transport dropped={transport['dropped']}",
                            )
                        )
                    for record in transport["records"]:
                        if any(value < 0 for value in record["durations_ns"].values()):
                            raise ValueError("negative Transport duration")
                        terminal = "result_published" if transport["site"] == "atnagent" else "result_acknowledged"
                        if record[terminal] == 0:
                            issues.append(
                                QualityIssue(
                                    kind="transport_incomplete",
                                    source=source.path,
                                    detail=f"trace_id={record['trace_id']}",
                                )
                            )
                case "graph":
                    graph = FfnGraphObserverSnapshot.model_validate_json(path.read_bytes(), strict=True)
                    key = (graph.generation.format(), graph.pe)
                    if key in graph_mapping:
                        raise ValueError("duplicate Graph Generation/PE")
                    uuid = graph.device_uuid if graph.device_uuid.startswith("GPU-") else f"GPU-{graph.device_uuid}"
                    if mapping.get(graph.pe) is not None and mapping[graph.pe] != uuid:
                        raise ValueError("Graph UUID conflicts with recorded launch placement")
                    graph_mapping[key] = uuid
                case "graph_events":
                    for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
                        if line.strip():
                            try:
                                TypeAdapter(SglangGraphEvent).validate_json(line, strict=True, extra="allow")
                            except ValueError as error:
                                raise ValueError(f"line {line_number}: {error}") from error
                case "evidence":
                    pass
        except (ValueError, OSError) as error:
            raise ValueError(f"invalid Observer input {source.path}: {error}") from error
    for source, snapshot in snapshots.items():
        generation, pe = snapshot["generation"], snapshot["pe"]
        uuid = mapping.get(pe) or graph_mapping.get((generation, pe))
        if uuid is None:
            issues.append(QualityIssue(kind="unknown_device_mapping", source=source, detail=f"PE {pe}"))
        if snapshot["dropped"]:
            issues.append(
                QualityIssue(
                    kind="snapshot_loss",
                    source=source,
                    detail=f"Fabric dropped={snapshot['dropped']}; invocation coverage unknown",
                )
            )
        local_ids: set[int] = set()
        for index, record in enumerate(snapshot["records"]):
            identity = f"{source}#{record['local_trace_id']}"
            try:
                if record["local_trace_id"] <= 0 or record["local_trace_id"] in local_ids:
                    raise ValueError("duplicate or nonpositive Local Trace ID")
                local_ids.add(record["local_trace_id"])
                if record["instance_index"] != 0 or not 0 < record["invocation_sequence"] <= 2**63 - 1:
                    raise ValueError("invalid instance or Invocation Sequence for serving-001")
                if (record["kind"] == "atnagent") != (pe == 0):
                    raise ValueError("record role conflicts with PE")
                names = {name.lower() for name in ROLE_EVENTS[record["kind"]].__members__}
                if set(record["events_ns"]) != names:
                    raise ValueError("event family differs from the pinned native declaration")
                if any(
                    isinstance(value, bool) or not 0 <= value <= 2**63 - 1 for value in record["events_ns"].values()
                ):
                    raise ValueError("event timestamp outside SQLite integer range")
                if any(record[field] < 0 for field in ("layer_ordinal", "payload_rows")):
                    raise ValueError("negative record geometry")
                lane, lease = record["facts"]["executor_lane_index"], record["facts"]["executor_lease_sequence"]
                if (lane is None) != (lease is None) or (
                    lane is not None and (lane != 0 or lease is None or lease <= 0)
                ):
                    raise ValueError("invalid Lane/Lease for serving-001")
                observations.append(
                    Observation(
                        identity=identity,
                        source=source,
                        index=index,
                        generation=generation,
                        pe=pe,
                        device_uuid=uuid,
                        record=record,
                        invocation=invocation_key(manifest, generation, record),
                    )
                )
            except ValueError as error:
                raise ValueError(f"invalid Fabric record {identity}: {error}") from error
    intervals: list[Interval] = []
    edges: list[Edge] = []
    grouped: dict[str, list[Observation]] = defaultdict(list)
    lease_owner: dict[tuple[str, int, int], str] = {}
    for observation in observations:
        grouped[observation.invocation].append(observation)
        lane = observation.record["facts"]["executor_lane_index"]
        lease = observation.record["facts"]["executor_lease_sequence"]
        if lane is not None and lease is not None:
            lease_key = (observation.generation, lane, lease)
            previous = lease_owner.setdefault(lease_key, observation.invocation)
            if previous != observation.invocation:
                raise ValueError(f"Lane/Lease reused by conflicting Invocations: {previous}, {observation.invocation}")
    for invocation, records in grouped.items():
        roles: dict[str, Observation] = {}
        constraints = {
            (item.record["layer_ordinal"], item.record["payload_rows"], item.record["output_requirement"])
            for item in records
        }
        leases = {
            (item.record["facts"]["executor_lane_index"], item.record["facts"]["executor_lease_sequence"])
            for item in records
            if item.record["facts"]["executor_lane_index"] is not None
        }
        if len(constraints) != 1 or len(leases) > 1:
            raise ValueError(f"conflicting Layer/payload/output/Lane/Lease for Invocation {invocation}")
        for item in records:
            role = item.record["kind"]
            if role in roles:
                raise ValueError(f"duplicate {role} record for Invocation {invocation}")
            roles[role] = item
            events = item.record["events_ns"]
            for event in REQUIRED_EVENTS[role]:
                if events[event] == 0:
                    issues.append(
                        QualityIssue(
                            kind="missing_boundary",
                            source=item.source,
                            invocation=invocation,
                            detail=f"{item.identity}/{event}",
                        )
                    )
            if item.record["facts"]["executor_lane_index"] is None:
                issues.append(
                    QualityIssue(kind="missing_lease", source=item.source, invocation=invocation, detail=item.identity)
                )
            for name, start_name, end_name in INTERVALS[role]:
                start, end = events[start_name], events[end_name]
                if start == 0 or end == 0:
                    continue
                if end < start:
                    issues.append(
                        QualityIssue(
                            kind="reversed_interval",
                            source=item.source,
                            invocation=invocation,
                            detail=f"{item.identity}/{name}",
                        )
                    )
                elif item.device_uuid is not None:
                    intervals.append(
                        Interval(
                            invocation=invocation,
                            record_id=item.identity,
                            name=name,
                            start_event=item.event_id(start_name),
                            end_event=item.event_id(end_name),
                            start_ns=start,
                            end_ns=end,
                            device_uuid=item.device_uuid,
                        )
                    )
        if set(roles) != {"atnagent", "coordinator", "ffnagent"}:
            issues.append(
                QualityIssue(kind="missing_participant", invocation=invocation, detail=f"present roles={sorted(roles)}")
            )
        for left_role, left_event, right_role, right_event, kind in (
            ("atnagent", "submission_published", "coordinator", "enqueued", "aggregate_condition"),
            ("coordinator", "admission_published", "atnagent", "admission_observed", "observation"),
            ("coordinator", "lane_execution_published", "ffnagent", "lane_execution_observed", "observation"),
            ("ffnagent", "completion_published", "coordinator", "ffnagent_completions_observed", "aggregate_condition"),
            ("coordinator", "output_commit_published", "atnagent", "output_commit_observed", "observation"),
            (
                "atnagent",
                "output_acknowledgement_published",
                "coordinator",
                "output_acknowledgements_observed",
                "aggregate_condition",
            ),
        ):
            left, right = roles.get(left_role), roles.get(right_role)
            if (
                left is None
                or right is None
                or not left.record["events_ns"][left_event]
                or not right.record["events_ns"][right_event]
            ):
                issues.append(
                    QualityIssue(
                        kind="missing_causal_endpoint",
                        invocation=invocation,
                        detail=f"{left_role}/{left_event} -> {right_role}/{right_event}",
                    )
                )
            elif (
                left.record["facts"]["executor_lease_sequence"] is None
                or right.record["facts"]["executor_lease_sequence"] is None
            ):
                issues.append(
                    QualityIssue(
                        kind="missing_causal_identity",
                        invocation=invocation,
                        detail=f"{left_role} -> {right_role}: unconfirmed Lease",
                    )
                )
            else:
                edges.append(
                    Edge(
                        invocation=invocation,
                        source_event=left.event_id(left_event),
                        target_event=right.event_id(right_event),
                        kind=kind,
                        participant_set=[left.pe],
                    )
                )
    prefill: str | None = None
    decode: str | None = None
    lease_sets: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for invocation, records in grouped.items():
        if any(issue.invocation == invocation for issue in issues):
            continue
        atn = next((item for item in records if item.record["kind"] == "atnagent"), None)
        if atn is None or atn.record["kind"] != "atnagent":
            continue
        if atn.record["facts"]["forward_mode"] == "prefill":
            prefill = prefill or invocation
        elif atn.record["facts"]["forward_mode"] == "decode":
            decode = decode or invocation
        lane, lease = atn.record["facts"]["executor_lane_index"], atn.record["facts"]["executor_lease_sequence"]
        if lane is not None and lease is not None:
            lease_sets[(atn.generation, atn.record["instance_index"], lane)].add(lease)
    reused = any(len(leases) > 1 for leases in lease_sets.values())
    representative = not issues and prefill is not None and decode is not None and reused
    quality = QualityReport(
        run_id=manifest.context.run_id,
        attempt=manifest.context.attempt,
        incomplete=bool(issues),
        issues=issues,
        prefill_invocation=prefill,
        decode_invocation=decode,
        lane_reused=reused,
        representative=representative,
    )
    return Timeline(
        manifest=manifest,
        snapshots=snapshots,
        observations=observations,
        intervals=intervals,
        edges=edges,
        quality=quality,
    )


SCHEMA = """
CREATE TABLE run (manifest_json TEXT NOT NULL);
CREATE TABLE source (path TEXT PRIMARY KEY, kind TEXT, sha256 TEXT, size INTEGER);
CREATE TABLE participant (pe INTEGER PRIMARY KEY, metadata_json TEXT);
CREATE TABLE invocation (id TEXT PRIMARY KEY, generation TEXT, instance_index INTEGER, sequence INTEGER);
CREATE TABLE record (id TEXT PRIMARY KEY, invocation TEXT REFERENCES invocation(id),
    source TEXT REFERENCES source(path),
    source_index INTEGER, pe INTEGER, role TEXT, device_uuid TEXT, raw_json TEXT);
CREATE TABLE event (id TEXT PRIMARY KEY, record_id TEXT REFERENCES record(id), name TEXT, timestamp_ns INTEGER,
    present INTEGER, clock_source TEXT);
CREATE TABLE interval (invocation TEXT REFERENCES invocation(id), record_id TEXT REFERENCES record(id), name TEXT,
    start_event TEXT REFERENCES event(id), end_event TEXT REFERENCES event(id), start_ns INTEGER, end_ns INTEGER,
    duration_ns INTEGER, device_uuid TEXT);
CREATE TABLE edge (invocation TEXT REFERENCES invocation(id), source_event TEXT REFERENCES event(id),
    target_event TEXT REFERENCES event(id), kind TEXT, participant_set_json TEXT);
CREATE TABLE quality (kind TEXT, detail TEXT, source TEXT, invocation TEXT);
"""


def write_database(timeline: Timeline, path: Path) -> None:
    path.touch(exist_ok=False)
    with closing(sqlite3.connect(path)) as database, database:
        database.execute("PRAGMA foreign_keys=ON")
        database.executescript(SCHEMA)
        database.execute("INSERT INTO run VALUES (?)", (timeline.manifest.model_dump_json(),))
        database.executemany(
            "INSERT INTO source VALUES (?,?,?,?)",
            [(source.path, source.kind, source.sha256, source.size) for source in timeline.manifest.files],
        )
        database.executemany(
            "INSERT INTO participant VALUES (?,?)",
            [(participant.pe, participant.model_dump_json()) for participant in timeline.manifest.context.participants],
        )
        for item in timeline.observations:
            database.execute(
                "INSERT OR IGNORE INTO invocation VALUES (?,?,?,?)",
                (item.invocation, item.generation, item.record["instance_index"], item.record["invocation_sequence"]),
            )
            database.execute(
                "INSERT INTO record VALUES (?,?,?,?,?,?,?,?)",
                (
                    item.identity,
                    item.invocation,
                    item.source,
                    item.index,
                    item.pe,
                    item.record["kind"],
                    item.device_uuid,
                    json.dumps(item.record),
                ),
            )
            database.executemany(
                "INSERT INTO event VALUES (?,?,?,?,?,?)",
                [
                    (item.event_id(name), item.identity, name, value, int(value != 0), "cuda_globaltimer_ns")
                    for name, value in item.record["events_ns"].items()
                ],
            )
        database.executemany(
            "INSERT INTO interval VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    item.invocation,
                    item.record_id,
                    item.name,
                    item.start_event,
                    item.end_event,
                    item.start_ns,
                    item.end_ns,
                    item.end_ns - item.start_ns,
                    item.device_uuid,
                )
                for item in timeline.intervals
            ],
        )
        database.executemany(
            "INSERT INTO edge VALUES (?,?,?,?,?)",
            [
                (edge.invocation, edge.source_event, edge.target_event, edge.kind, json.dumps(edge.participant_set))
                for edge in timeline.edges
            ],
        )
        database.executemany(
            "INSERT INTO quality VALUES (?,?,?,?)",
            [(issue.kind, issue.detail, issue.source, issue.invocation) for issue in timeline.quality.issues],
        )


class ChromeEvent(TypedDict, total=False):
    name: str
    ph: str
    cat: str
    pid: int
    tid: int
    ts: float
    dur: float
    s: str
    args: dict[str, JsonValue]


def pack_intervals(intervals: list[Interval]) -> list[list[Interval]]:
    """Reuse the lowest available row; overlapping intervals occupy separate rows."""

    rows: list[list[Interval]] = []
    busy: list[tuple[int, int]] = []
    available: list[int] = []
    for interval in sorted(intervals, key=lambda item: (item.start_ns, item.end_ns, item.record_id)):
        while busy and busy[0][0] <= interval.start_ns:
            _, row = heappop(busy)
            heappush(available, row)
        if available:
            row = heappop(available)
        else:
            row = len(rows)
            rows.append([])
        rows[row].append(interval)
        heappush(busy, (interval.end_ns, row))
    return rows


def observation_args(item: Observation) -> dict[str, JsonValue]:
    return {
        "invocation": item.invocation,
        "generation": item.generation,
        "pe": item.pe,
        "role": item.record["kind"],
        "instance_index": item.record["instance_index"],
        "layer_ordinal": item.record["layer_ordinal"],
        "executor_lane_index": item.record["facts"]["executor_lane_index"],
        "executor_lease_sequence": item.record["facts"]["executor_lease_sequence"],
        "source": item.source,
        "record_id": item.identity,
        "record_index": item.index,
    }


def write_traces(timeline: Timeline, output: Path) -> None:
    intervals_by_record: dict[str, list[Interval]] = defaultdict(list)
    for interval in timeline.intervals:
        intervals_by_record[interval.record_id].append(interval)
    for device in sorted({item.device_uuid for item in timeline.observations if item.device_uuid is not None}):
        records = sorted(
            (item for item in timeline.observations if item.device_uuid == device), key=lambda item: item.identity
        )
        times = [value for item in records for value in item.record["events_ns"].values() if value]
        if not times:
            continue
        origin = min(times)
        events: list[ChromeEvent] = []
        groups: dict[tuple[str, int, str, int], list[Observation]] = defaultdict(list)
        for item in records:
            lane = item.record["facts"]["executor_lane_index"]
            groups[(item.generation, item.pe, item.record["kind"], -1 if lane is None else lane)].append(item)
        multiple_generations = len({item.generation for item in records}) > 1
        track = 0
        for (generation, pe, role, lane), members in sorted(groups.items()):
            label = f"PE {pe} {role} lane {lane if lane >= 0 else 'unknown'}"
            if multiple_generations:
                label += f" generation {generation}"
            track += 1
            events.append(
                {"name": "thread_name", "ph": "M", "pid": 1, "tid": track, "args": {"name": f"{label} events"}}
            )
            for item in members:
                for name, value in sorted(item.record["events_ns"].items()):
                    if value:
                        events.append(
                            {
                                "name": name,
                                "ph": "I",
                                "s": "t",
                                "cat": "fabric",
                                "pid": 1,
                                "tid": track,
                                "ts": (value - origin) / 1000,
                                "args": {
                                    **observation_args(item),
                                    "event_id": item.event_id(name),
                                    "raw_ns": str(value),
                                },
                            }
                        )
            owners = {item.identity: item for item in members}
            activities: dict[str, list[Interval]] = defaultdict(list)
            for item in members:
                for interval in intervals_by_record[item.identity]:
                    activities[interval.name].append(interval)
            for name, intervals in sorted(activities.items()):
                rows = pack_intervals(intervals)
                for row, packed in enumerate(rows):
                    track += 1
                    suffix = f" parallel {row + 1}" if len(rows) > 1 else ""
                    events.append(
                        {
                            "name": "thread_name",
                            "ph": "M",
                            "pid": 1,
                            "tid": track,
                            "args": {"name": f"{label} {name}{suffix}"},
                        }
                    )
                    for interval in packed:
                        events.append(
                            {
                                "name": interval.name,
                                "ph": "X",
                                "cat": "fabric.local",
                                "pid": 1,
                                "tid": track,
                                "ts": (interval.start_ns - origin) / 1000,
                                "dur": (interval.end_ns - interval.start_ns) / 1000,
                                "args": {
                                    **observation_args(owners[interval.record_id]),
                                    "start_event": interval.start_event,
                                    "end_event": interval.end_event,
                                    "start_ns": str(interval.start_ns),
                                    "end_ns": str(interval.end_ns),
                                },
                            }
                        )
        payload = {
            "traceEvents": events,
            "displayTimeUnit": "ns",
            "metadata": {
                "device_uuid": device,
                "clock_source": "cuda_globaltimer_ns",
                "display_origin_ns": str(origin),
                "timestamp_unit": "us",
                "clock_calibrated": False,
                "run_id": timeline.manifest.context.run_id,
                "attempt": timeline.manifest.context.attempt,
            },
        }
        (output / f"{device}.trace.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def convert(manifest_path: Path, output: Path) -> QualityReport:
    """Publish derived files separately; invalid input creates no output directory."""

    if output.resolve().is_relative_to(manifest_path.parent.resolve() / "raw"):
        raise ValueError("derived output must be outside the raw input directory")
    timeline = read_timeline(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    write_database(timeline, output / "timeline.sqlite")
    write_traces(timeline, output)
    (output / "quality.json").write_text(timeline.quality.model_dump_json(indent=2), encoding="utf-8")
    verify_archive(timeline.manifest, manifest_path.parent)
    return timeline.quality


class InvocationQuery(TypedDict):
    invocation: str
    cross_device_duration_available: Literal[False]
    records: list[dict[str, JsonValue]]
    events: list[dict[str, JsonValue]]
    intervals: list[dict[str, JsonValue]]
    edges: list[dict[str, JsonValue]]
    quality: list[dict[str, JsonValue]]
    participants: list[dict[str, JsonValue]]


def query(path: Path, generation: str, instance: int, sequence: int) -> InvocationQuery:
    """Read one Invocation and include all file/run issues affecting its coverage."""

    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as database:
        database.row_factory = sqlite3.Row
        row = database.execute(
            "SELECT id FROM invocation WHERE generation=? AND instance_index=? AND sequence=?",
            (generation, instance, sequence),
        ).fetchone()
        if row is None:
            raise ValueError(f"Invocation not found: {generation}/{instance}/{sequence}")
        invocation = str(row["id"])
        sources = [
            str(record[0]) for record in database.execute("SELECT source FROM record WHERE invocation=?", (invocation,))
        ]
        result: dict[str, list[dict[str, JsonValue]]] = {}
        for name, statement in (
            ("records", "SELECT * FROM record WHERE invocation=? ORDER BY pe,role"),
            (
                "events",
                "SELECT event.* FROM event JOIN record ON record.id=event.record_id "
                "WHERE record.invocation=? ORDER BY record.pe,event.name",
            ),
            ("intervals", "SELECT * FROM interval WHERE invocation=?"),
            ("edges", "SELECT * FROM edge WHERE invocation=?"),
        ):
            result[name] = [dict(item) for item in database.execute(statement, (invocation,))]
        result["quality"] = [
            dict(issue)
            for issue in database.execute("SELECT * FROM quality")
            if issue["invocation"] == invocation
            or (issue["invocation"] is None and (issue["source"] is None or issue["source"] in sources))
        ]
        result["participants"] = [
            json.loads(item[0]) for item in database.execute("SELECT metadata_json FROM participant")
        ]
        return {
            "invocation": invocation,
            "cross_device_duration_available": False,
            "records": result["records"],
            "events": result["events"],
            "intervals": result["intervals"],
            "edges": result["edges"],
            "quality": result["quality"],
            "participants": result["participants"],
        }
