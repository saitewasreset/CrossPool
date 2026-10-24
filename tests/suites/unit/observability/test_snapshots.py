"""Independent legacy golden projections and incomplete-record behavior."""

from dataclasses import replace

import pytest
from pydantic import TypeAdapter

from xpool.devkit.fabric_observer import FabricSnapshotJson
from xpool.devkit.transport_observer import TransportSnapshotJson
from xpool.observability import (
    Metadata,
    ObservationError,
    ProjectionStatus,
    SnapshotEvidenceReport,
    SnapshotRequest,
    project_legacy_snapshots,
    reconstruct,
)

from .fixtures import events_for, expand, fixture, metadata_for, scenario


def requests(run_id: str) -> tuple[SnapshotRequest, ...]:
    return (
        SnapshotRequest(run_id, 3, 1, 1, endpoint=(2, 1)),
        SnapshotRequest(run_id, 2, 1, 1, endpoint=(2, 1)),
        SnapshotRequest(run_id, 2, 1, 2),
        SnapshotRequest(run_id, 5, 1, 1),
    )


def test_complete_golden_snapshots() -> None:
    result = reconstruct(events_for(scenario("normal_dense")), metadata=(Metadata.model_validate(fixture("metadata")),))
    actual = project_legacy_snapshots(result, requests(result.metadata[0].run_id))
    golden = fixture("snapshots")
    assert [projection.snapshot for projection in actual[:2]] == golden["expected_transport"]
    assert [projection.snapshot for projection in actual[2:]] == golden["expected_fabric"]
    assert [p.evidence_report.model_dump(mode="json") for p in actual if p.evidence_report] == golden[
        "expected_evidence_reports"
    ]
    assert all(p.evidence_report is not None and p.evidence_report.unobserved_operations == 0 for p in actual)


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
    assert projection.evidence_report is not None
    assert projection.evidence_report.unobserved_operations == unobserved
    # The raw offline inventory is not truncated by the legacy capacity.
    assert len(result.of_kind("operation")) == 2


def test_unknown_attempt_total_is_explicit() -> None:
    result = reconstruct(events_for(scenario("normal_dense")))
    request = SnapshotRequest(result.metadata[0].run_id, 3, 1, None, endpoint=(2, 1))
    projection = project_legacy_snapshots(result, (request,))[0]
    assert projection.snapshot is None
    assert projection.evidence_report is not None
    assert projection.evidence_report.unobserved_operations is None
    assert len(projection.records) == 1
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
    assert projection.evidence_report is not None
    assert sum(e.projection_status == ProjectionStatus.OMITTED for e in projection.evidence_report.entries) == 1
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
    for sequence, projection in enumerate(projections, 1):
        assert projection.evidence_report is not None
        operation = projection.evidence_report.entries[0].operation
        assert operation is not None
        assert (operation.endpoint_owner, operation.endpoint_sequence) == (2, sequence)


snapshot_cases = [c for c in fixture("snapshots")["cases"] if "base_case" in c["input"]]


@pytest.mark.parametrize("case", snapshot_cases, ids=[c["id"] for c in snapshot_cases])
def test_v3_snapshot_cases(case: dict) -> None:
    data = expand(case["input"])
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ObservationError) as caught:
            reconstruct(events_for(data))
        assert caught.value.code == expected["error"]
        assert caught.value.producer is not None
        assert caught.value.event_sequence is not None
        if "conflicting_sources" in expected:
            assert [
                {"producer": s.identity.producer, "event_sequence": s.identity.event_sequence}
                for s in caught.value.sources
            ] == expected["conflicting_sources"]
        return
    result = reconstruct(events_for(data), metadata=(Metadata.model_validate(metadata_for(data)),))
    selection = data["select"]
    counts = data["operation_counts"]
    request = SnapshotRequest(
        result.metadata[0].run_id,
        selection["producer"],
        1,
        counts["sequence"],
        counts["capacity"],
        (2, 1) if selection["observer"] == "transport" else None,
    )
    projection = project_legacy_snapshots(result, (request,))[0]
    report = projection.evidence_report
    assert report is not None
    # The independently frozen oracle includes exact source references and arrays.
    if "evidence_report" in expected:
        assert report.model_dump(mode="json") == expected["evidence_report"]
    if "evidence_entry" in expected:
        entry = next(e for e in report.entries if e.local_trace_id == selection["local_trace_id"])
        assert entry.model_dump(mode="json") == expected["evidence_entry"]
    if "legacy_records" in expected:
        assert list(projection.records) == expected["legacy_records"]
    if "legacy_record" in expected:
        selected_records = [
            r for r in projection.records if r.get("trace_id", r.get("local_trace_id")) == selection["local_trace_id"]
        ]
        assert selected_records == ([] if expected["legacy_record"] is None else [expected["legacy_record"]])
    if "legacy_snapshot_available" in expected:
        assert (projection.snapshot is not None) == expected["legacy_snapshot_available"]
    if "legacy_classification" in expected:
        assert projection.snapshot is not None
        snapshot = TypeAdapter(TransportSnapshotJson).validate_python(projection.snapshot)
        assert snapshot["record_counts"][expected["legacy_classification"]] == 1
    if "phase_sample_counts" in expected:
        assert projection.snapshot is not None
        snapshot = TypeAdapter(TransportSnapshotJson).validate_python(projection.snapshot)
        actual_counts = {name: value["count"] for name, value in snapshot["phase_summary"].items()}
        assert all(actual_counts.get(name, 0) == count for name, count in expected["phase_sample_counts"].items())
    # The public strict report also supports an exact JSON round trip.
    assert SnapshotEvidenceReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("routing", ["disabled", "enabled", "unavailable"])
def test_routing_observer_state_does_not_invent_layer_applicability(routing: str) -> None:
    data = scenario("normal_dense")
    data["capabilities"] = {"routing": routing}
    result = reconstruct(events_for(data))
    projection = project_legacy_snapshots(result, requests(result.metadata[0].run_id)[3:])[0]
    assert projection.evidence_report is not None
    entry = projection.evidence_report.entries[0]
    assert entry.evidence_status == "partial"
    assert entry.unknown_applicability == ["RoutingMetadataObserved", "RoutingMetadataPublished"]
    assert entry.missing_boundaries == []
    assert "applicability_unknown" in result.quality.codes


@pytest.mark.parametrize("delivery", [2, 3])
@pytest.mark.parametrize("ffnagent_count", [1, 2])
def test_complete_delivery_requires_retained_peer_topology(delivery: int, ffnagent_count: int) -> None:
    data = scenario("normal_dense")
    generation = fixture("metadata")["generations"][0]
    generation["ffnagent_count"] = ffnagent_count
    data["metadata_overrides"] = {"generations_replace": [generation]}
    for item in data["events"]:
        if item["producer"] == 5 and item["record"]["kind"] < 500:
            item["record"]["delivery"] = delivery
    result = reconstruct(events_for(data))
    projection = project_legacy_snapshots(result, requests(result.metadata[0].run_id)[3:])[0]
    assert projection.evidence_report is not None
    entry = projection.evidence_report.entries[0]
    partials = ["PartialReadyPublished", "PeerPartialsReadyObserved"]
    if ffnagent_count == 1:
        assert entry.evidence_status == "complete"
        assert entry.unknown_applicability == []
        assert all(
            f"events_ns.{name}" in entry.not_applicable_fields
            for name in (
                "partial_ready_published",
                "peer_partials_ready_observed",
            )
        )
        decisions = [s for s in entry.sources if s.field in [f"boundary.{kind}" for kind in partials]]
        assert len(decisions) == 2
        assert all(s.metadata is not None and s.metadata.path == "generations[0].ffnagent_count" for s in decisions)
    else:
        # Count alone cannot reveal this layer's selected group or local TP rank.
        assert entry.unknown_applicability == partials
        assert entry.evidence_status == "partial"
    assert not entry.missing_boundaries


def test_canonical_field_sources_are_bounded_and_event_order_independent() -> None:
    events = events_for(scenario("normal_dense"))
    event = next(e for e in events if e.record.kind == 100)
    duplicate = replace(
        event,
        record=event.record.model_copy(update={"event_sequence": 50}),
        source=replace(event.source, identity=replace(event.source.identity, event_sequence=50)),
    )
    metadata = (Metadata.model_validate(fixture("metadata")),)
    original = reconstruct(events, metadata=metadata)
    repeated = reconstruct(reversed((*events, duplicate, duplicate)), metadata=metadata)
    request = requests(metadata[0].run_id)[0:1]
    before = project_legacy_snapshots(original, request)[0]
    after = project_legacy_snapshots(repeated, request)[0]
    assert before.evidence_report == after.evidence_report
    assert len(repeated.event_sources[duplicate.source.identity]) == 2
    assert len(repeated.events) == len(original.events) + 1


@pytest.mark.parametrize("observer,producer,endpoint", [("transport", 3, (2, 1)), ("fabric", 5, None)])
def test_disabled_observer_has_no_separate_report(
    observer: str, producer: int, endpoint: tuple[int, int] | None
) -> None:
    data = fixture("metadata")
    data["capabilities"][observer] = "disabled"
    metadata = Metadata.model_validate(data)
    result = reconstruct((), metadata=(metadata,))
    projection = project_legacy_snapshots(
        result, (SnapshotRequest(metadata.run_id, producer, 1, 0, endpoint=endpoint),)
    )[0]
    assert projection.snapshot is None
    assert projection.evidence_report is None
    assert projection.records == ()


def test_entirely_unobserved_fabric_domain_keeps_report_without_guessing_pe() -> None:
    metadata = Metadata.model_validate(fixture("metadata"))
    result = reconstruct((), metadata=(metadata,))
    projection = project_legacy_snapshots(result, (SnapshotRequest(metadata.run_id, 5, 1, 2),))[0]
    assert projection.snapshot is None
    assert projection.records == ()
    assert projection.evidence_report is not None
    report = projection.evidence_report
    assert report.unobserved_operations == 2
    assert all(e.role is None and e.invocation is None and e.operation is None for e in report.entries)
    assert all(e.quality == ["operation_unobserved"] for e in report.entries)
    assert "legacy_fact_missing" in {f.code for f in projection.findings}


def test_closed_keeps_preceding_valid_intervals_without_assuming_lost_boundaries() -> None:
    case = next(c for c in snapshot_cases if c["id"] == "closed_does_not_prove_earlier_path")
    result = reconstruct(events_for(expand(case["input"])))
    own = [i for i in result.intervals if i.observation.role == 1]
    assert any(i.name == "request_staging" and i.duration_ns == 10 for i in own)
    assert any(i.name == "closed_total" and i.duration_ns == 20 for i in own)
    assert any(i.name == "request_publication" and i.reasons == ("applicability_unknown",) for i in own)


def test_reversal_between_phases_marks_evidence_without_discarding_valid_samples() -> None:
    events = tuple(
        replace(e, record=e.record.model_copy(update={"raw_time_ns": 115})) if e.record.kind == 107 else e
        for e in events_for(scenario("normal_dense"))
    )
    result = reconstruct(events)
    projection = project_legacy_snapshots(result, requests(result.metadata[0].run_id)[:1])[0]
    assert projection.evidence_report is not None
    assert projection.evidence_report.entries[0].quality == ["temporal_conflict"]
    assert projection.evidence_report.entries[0].evidence_status == "partial"
    assert "temporal_conflict" in result.quality.codes
    record = TypeAdapter(TransportSnapshotJson).validate_python(projection.snapshot)["records"][0]
    assert record["durations_ns"]["request_staging"] == 10
    assert record["durations_ns"]["output_copy"] == 205
