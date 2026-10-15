"""Independent legacy golden projections and incomplete-record behavior."""

from dataclasses import replace

import pytest
from pydantic import TypeAdapter

from xpool.devkit.fabric_observer import FabricSnapshotJson
from xpool.devkit.transport_observer import TransportSnapshotJson
from xpool.observability import SnapshotRequest, project_legacy_snapshots, reconstruct

from .fixtures import events_for, fixture, scenario


def requests(run_id: str) -> tuple[SnapshotRequest, ...]:
    return (
        SnapshotRequest(run_id, 3, 1, 1, endpoint=(2, 1)),
        SnapshotRequest(run_id, 2, 1, 1, endpoint=(2, 1)),
        SnapshotRequest(run_id, 2, 1, 2),
        SnapshotRequest(run_id, 5, 1, 1),
    )


def test_complete_golden_snapshots() -> None:
    result = reconstruct(events_for(scenario("normal_dense")))
    actual = project_legacy_snapshots(result, requests(result.metadata[0].run_id))
    golden = fixture("snapshots")
    assert [projection.snapshot for projection in actual[:2]] == golden["expected_transport"]
    assert [projection.snapshot for projection in actual[2:]] == golden["expected_fabric"]
    assert all(projection.sources for projection in actual)
    assert all(projection.unobserved_operations == projection.omitted_records == 0 for projection in actual)


def test_lost_opening_does_not_renumber_and_site_result_is_not_borrowed() -> None:
    result = reconstruct(events_for(scenario("missing_result")))
    projection = project_legacy_snapshots(result, requests(result.metadata[0].run_id)[:1])[0]
    assert projection.snapshot is not None
    snapshot = TypeAdapter(TransportSnapshotJson).validate_python(projection.snapshot)
    assert snapshot["records"][0]["result_code"] == "protocol_mismatch"
    assert snapshot["record_counts"]["incomplete"] == 1
    assert "unknown_result" in {finding.code for finding in projection.findings}


@pytest.mark.parametrize(
    "capacity,attempts,expected_ids,unobserved,dropped", [(1, 3, [1], 0, 2), (2, 3, [1, 2], 0, 1), (4, 4, [1, 2], 2, 0)]
)
def test_prefix_retention_and_unobserved_counts(
    capacity: int, attempts: int, expected_ids: list[int], unobserved: int, dropped: int
) -> None:
    result = reconstruct(events_for(scenario("graph_replay")))
    request = SnapshotRequest(result.metadata[0].run_id, 3, 1, attempts, capacity, (2, 1))
    projection = project_legacy_snapshots(result, (request,))[0]
    assert projection.snapshot is not None
    assert projection.snapshot["sequence"] == attempts
    assert projection.snapshot["dropped"] == dropped
    snapshot = TypeAdapter(TransportSnapshotJson).validate_python(projection.snapshot)
    assert [record["trace_id"] for record in snapshot["records"]] == expected_ids
    assert projection.unobserved_operations == unobserved
    # The raw offline inventory is not truncated by the legacy capacity.
    assert len(result.of_kind("operation")) == 2


def test_unknown_attempt_total_is_explicit() -> None:
    result = reconstruct(events_for(scenario("normal_dense")))
    request = SnapshotRequest(result.metadata[0].run_id, 3, 1, None, endpoint=(2, 1))
    projection = project_legacy_snapshots(result, (request,))[0]
    assert projection.snapshot is None
    assert projection.unobserved_operations is None
    assert projection.findings[0].code == "snapshot_total_unknown"


def test_zero_timestamp_preserves_raw_and_excludes_legacy_duration() -> None:
    events = list(events_for(scenario("normal_dense")))
    index = next(i for i, event in enumerate(events) if event.record.kind == 100)
    events[index] = replace(events[index], record=events[index].record.model_copy(update={"raw_time_ns": 0}))
    result = reconstruct(events)
    projection = project_legacy_snapshots(result, requests(result.metadata[0].run_id)[:1])[0]
    assert projection.snapshot is not None
    record = TypeAdapter(TransportSnapshotJson).validate_python(projection.snapshot)["records"][0]
    assert record["request_staging_started"] == 0
    assert "request_staging" not in record["durations_ns"]
    assert "legacy_zero_timestamp" in {finding.code for finding in projection.findings}
    assert any(interval.name == "request_staging" and interval.duration_ns == 110 for interval in result.intervals)


def test_fifo_missing_ticket_omits_record_instead_of_changing_scheduler() -> None:
    events = list(events_for(scenario("normal_dense")))
    for i, event in enumerate(events):
        if event.record.role == 3:
            events[i] = replace(
                event,
                record=event.record.model_copy(
                    update={
                        "presence": event.record.presence & ~(1 << 13),
                        "ready_ticket": 0,
                    }
                ),
            )
    result = reconstruct(events)
    projection = project_legacy_snapshots(result, requests(result.metadata[0].run_id)[2:3])[0]
    assert projection.snapshot is not None
    assert projection.omitted_records == 1
    snapshot = TypeAdapter(FabricSnapshotJson).validate_python(projection.snapshot)
    assert [record["kind"] for record in snapshot["records"]] == ["atnagent"]
    assert "legacy_fact_missing" in {finding.code for finding in projection.findings}


def test_endpoint_recreation_has_separate_legacy_domains() -> None:
    result = reconstruct(events_for(scenario("endpoint_recreated")))
    run_id = result.metadata[0].run_id
    projections = project_legacy_snapshots(
        result,
        (
            SnapshotRequest(run_id, 3, 1, 1, endpoint=(2, 1)),
            SnapshotRequest(run_id, 3, 1, 1, endpoint=(2, 2)),
        ),
    )
    assert all(projection.snapshot is not None for projection in projections)
    assert projections[0].sources[1][0].endpoint == (2, 1)
    assert projections[1].sources[1][0].endpoint == (2, 2)
