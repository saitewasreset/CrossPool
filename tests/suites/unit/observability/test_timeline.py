import json
import sqlite3
from pathlib import Path

import pytest
from local_scripts.observability.archive import archive_attempt, load_manifest
from local_scripts.observability.models import Participant, RunContext
from local_scripts.observability.timeline import convert, query, read_timeline
from pydantic import TypeAdapter

import xpool.native
from xpool.devkit.fabric_observer import FabricRecordJson, FabricSnapshotJson
from xpool.devkit.transport_observer import TransportRecordJson, TransportSnapshotJson

GENERATION = "00000000000000010000000000000002"


def fabric_records(pe: int) -> FabricSnapshotJson:
    records: list[FabricRecordJson] = []
    for sequence, mode in ((1, "prefill"), (2, "decode")):
        for role in ["atnagent"] if pe == 0 else ["coordinator", "ffnagent"]:
            match role:
                case "atnagent":
                    family = xpool.native.devkit.fabric_observer.AtnAgentEvent
                    facts = {
                        "dp_rank_payload_rows": 4,
                        "forward_mode": mode,
                        "executor_lane_index": 0,
                        "executor_lease_sequence": sequence,
                    }
                case "coordinator":
                    family = xpool.native.devkit.fabric_observer.CoordinatorEvent
                    facts = {
                        "executor_lane_index": 0,
                        "executor_lease_sequence": sequence,
                        "scheduler": {"policy": "fifo", "ready_ticket": sequence},
                    }
                case _:
                    family = xpool.native.devkit.fabric_observer.FfnAgentEvent
                    facts = {
                        "executor_lane_index": 0,
                        "executor_lease_sequence": sequence,
                        "payload_row_capacity": 8,
                        "delivery": "single_complete",
                    }
            events = {
                name.lower(): sequence * 10000 + (index + 1) * 100 for index, name in enumerate(family.__members__)
            }
            if role == "ffnagent":
                for event in (
                    "routing_metadata_published",
                    "routing_metadata_observed",
                    "partial_ready_published",
                    "peer_partials_ready_observed",
                ):
                    events[event] = 0
            records.append(
                TypeAdapter(FabricRecordJson).validate_python(
                    {
                        "local_trace_id": len(records) + 1,
                        "kind": role,
                        "instance_index": 0,
                        "invocation_sequence": sequence,
                        "layer_ordinal": 0,
                        "payload_rows": 4,
                        "output_requirement": "per_rank_complete",
                        "dp_row_layout": "uniform_by_rank" if pe == 0 else None,
                        "facts": facts,
                        "events_ns": events,
                        "durations_ns": {},
                    },
                    strict=True,
                )
            )
    return {
        "generation": GENERATION,
        "pe": pe,
        "atnagent_count": 1,
        "ffnagent_count": 1,
        "model_topologies": [{"atn_tp_size": 1, "atn_dp_size": 1}],
        "sequence": len(records),
        "dropped": 0,
        "records": records,
    }


def prepare_attempt(tmp_path: Path, snapshots: list[FabricSnapshotJson] | None = None, *, mapped: bool = True) -> Path:
    source = tmp_path / "attempt"
    source.mkdir()
    for index, snapshot in enumerate(snapshots if snapshots is not None else [fabric_records(0), fabric_records(1)]):
        (source / f"xpool.fabric-observer.{index}.json").write_text(json.dumps(snapshot))
    graph = {
        "generation": {"high": 1, "low": 2},
        "pe": 1,
        "device": 1,
        "device_uuid": "GPU-b",
        "primary_graphs": [{"node_counts": {}, "binding_site_count": 1}],
        "lane_graphs": [{"node_counts": {}, "compute_branch_count": 1, "delivery_branch_count": 1}],
    }
    (source / "xpool.graph-observer.sample.json").write_text(json.dumps(graph))
    (source / "xpool.graph-observer.123.jsonl").write_text(
        json.dumps(
            {"pid": 123, "time_ns": 10, "forward_phase": "prefill", "backend_class": "Breakable", "event": "capture"}
        )
        + "\n"
    )
    for site in ("instance", "atnagent"):
        record: TransportRecordJson = {
            "trace_id": 1,
            "payload_rows": 4,
            "layer_ordinal": 0,
            "forward_mode": "prefill",
            "output_requirement": "per_rank_complete",
            "dp_row_layout": "uniform_by_rank",
            "result_code": "ok",
            "request_staging_started": 1,
            "request_staging_completed": 2,
            "request_published": 3,
            "request_observed": 4,
            "execution_started": 5,
            "execution_completed": 6,
            "result_published": 7,
            "result_observed": 8,
            "output_copied": 9,
            "result_acknowledged": 10,
            "closed": 0,
            "durations_ns": {},
        }
        transport: TransportSnapshotJson = {
            "pid": 123,
            "site": site,
            "model_id": "Qwen/Qwen3-0.6B",
            "rank": 0,
            "arena_handle_suffix": "suffix",
            "sequence": 1,
            "dropped": 0,
            "phase_summary": {},
            "record_counts": {"retained": 1, "completed": 1, "closed": 0, "incomplete": 0},
            "records": [record],
        }
        (source / f"xpool.transport-observer.{site}.json").write_text(json.dumps(transport))
    participants = (
        [
            Participant(pe=0, device_uuid="GPU-a", evidence="synthetic launch"),
            Participant(pe=1, device_uuid="GPU-b", evidence="synthetic launch"),
        ]
        if mapped
        else []
    )
    return archive_attempt(
        source, tmp_path / "archive", RunContext(run_id="test", attempt=1, outcome="passed", participants=participants)
    )


def test_query_links_real_endpoints_and_keeps_clock_domains_separate(tmp_path: Path) -> None:
    manifest = prepare_attempt(tmp_path)
    output = tmp_path / "derived"
    report = convert(manifest, output)
    assert report.representative
    assert report.lane_reused
    result = query(output / "timeline.sqlite", GENERATION, 0, 1)
    assert len(result["records"]) == 3
    assert len(result["edges"]) == 6
    assert result["quality"] == []
    assert not result["cross_device_duration_available"]
    with sqlite3.connect(output / "timeline.sqlite") as database:
        assert database.execute("PRAGMA foreign_key_check").fetchall() == []
        assert database.execute("SELECT COUNT(*) FROM event WHERE present=0").fetchone()[0] > 0
    traces = sorted(output.glob("*.trace.json"))
    assert len(traces) == 2
    for trace in traces:
        payload = json.loads(trace.read_text())
        assert not payload["metadata"]["clock_calibrated"]
        assert all(event["ph"] in {"I", "X", "M"} for event in payload["traceEvents"])
        assert len({event["tid"] for event in payload["traceEvents"] if event["ph"] == "X"}) == sum(
            event["ph"] == "X" for event in payload["traceEvents"]
        )
    with pytest.raises(ValueError, match="not found"):
        query(output / "timeline.sqlite", GENERATION, 0, 99)


@pytest.mark.parametrize("conflict", ["lease", "layer", "duplicate"])
def test_conflicts_fail_without_publishing_derived_files(tmp_path: Path, conflict: str) -> None:
    snapshots = [fabric_records(0), fabric_records(1)]
    if conflict == "lease":
        snapshots[1]["records"][0]["facts"]["executor_lease_sequence"] = 9
    elif conflict == "layer":
        snapshots[1]["records"][0]["layer_ordinal"] = 1
    else:
        snapshots[0]["records"].append(snapshots[0]["records"][0])
    manifest = prepare_attempt(tmp_path, snapshots)
    with pytest.raises(ValueError, match=r"conflict|duplicate"):
        convert(manifest, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_loss_scope_missing_boundary_and_reversed_interval(tmp_path: Path) -> None:
    snapshots = [fabric_records(0), fabric_records(1)]
    snapshots[0]["dropped"] = 2
    snapshots[0]["records"][0]["events_ns"]["admission_observed"] = 0
    snapshots[1]["records"][1]["events_ns"]["compute_completed"] = 1
    manifest = prepare_attempt(tmp_path, snapshots)
    quality = convert(manifest, tmp_path / "derived")
    assert not quality.representative
    kinds = {issue.kind for issue in quality.issues}
    assert {"snapshot_loss", "missing_boundary", "missing_causal_endpoint", "reversed_interval"} <= kinds
    assert all(issue.invocation is None for issue in quality.issues if issue.kind == "snapshot_loss")
    result = query(tmp_path / "derived/timeline.sqlite", GENERATION, 0, 2)
    assert any(item["kind"] == "snapshot_loss" for item in result["quality"])


def test_generation_isolation_and_unknown_gpu(tmp_path: Path) -> None:
    snapshots = [fabric_records(0), fabric_records(1)]
    snapshots[1]["generation"] = "2" * 32
    timeline = read_timeline(prepare_attempt(tmp_path, snapshots, mapped=False))
    assert len({item.invocation for item in timeline.observations}) == 4
    assert all(
        "fabric-observer.1" in edge.source_event and "fabric-observer.1" in edge.target_event for edge in timeline.edges
    )
    assert any(issue.kind == "missing_participant" for issue in timeline.quality.issues)
    assert not timeline.intervals


def test_schema_damage_and_source_mutation_are_fatal(tmp_path: Path) -> None:
    snapshots = [fabric_records(0), fabric_records(1)]
    snapshots[0]["records"][0]["events_ns"]["submission_prepared"] = -1
    manifest = prepare_attempt(tmp_path, snapshots)
    with pytest.raises(ValueError, match="timestamp"):
        read_timeline(manifest)
    source = load_manifest(manifest).files[0]
    (manifest.parent / source.path).write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        read_timeline(manifest)


def test_optional_dense_events_remain_absent_not_fabricated(tmp_path: Path) -> None:
    timeline = read_timeline(prepare_attempt(tmp_path))
    assert timeline.quality.representative
    assert all("routing" not in edge.source_event for edge in timeline.edges)
    assert len(timeline.edges) == 12


def test_transport_is_not_joined_to_fabric(tmp_path: Path) -> None:
    timeline = read_timeline(prepare_attempt(tmp_path))
    assert all(
        "fabric-observer" in edge.source_event and "fabric-observer" in edge.target_event for edge in timeline.edges
    )
    assert all(item.record["local_trace_id"] > 0 for item in timeline.observations)


def test_reusing_one_lease_for_different_invocations_is_a_conflict(tmp_path: Path) -> None:
    snapshots = [fabric_records(0), fabric_records(1)]
    for snapshot in snapshots:
        for record in snapshot["records"]:
            record["facts"]["executor_lease_sequence"] = 1
    with pytest.raises(ValueError, match="conflicting Invocations"):
        read_timeline(prepare_attempt(tmp_path, snapshots))


def test_unknown_lease_never_creates_causal_links(tmp_path: Path) -> None:
    snapshots = [fabric_records(0), fabric_records(1)]
    record = snapshots[0]["records"][0]
    assert record["kind"] == "atnagent"
    record["facts"]["executor_lane_index"] = None
    record["facts"]["executor_lease_sequence"] = None
    timeline = read_timeline(prepare_attempt(tmp_path, snapshots))
    assert any(issue.kind == "missing_causal_identity" for issue in timeline.quality.issues)
    assert all("fabric-observer.0.json#1/" not in edge.source_event + edge.target_event for edge in timeline.edges)
