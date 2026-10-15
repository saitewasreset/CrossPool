"""Fixed causal fixtures verify observed relations and identity isolation."""

from dataclasses import replace

import pytest

from xpool.observability.errors import ObservationError
from xpool.observability.reconstruction import EntityIdentity, phase_interval, reconstruct
from xpool.observability.wire import EventKind, GraphError, PresenceBit

from .fixtures import events_for, fixture, scenario

event_cases = [
    case for case in fixture("causality")["cases"] if "events" in case["input"] or "base_case" in case["input"]
]


@pytest.mark.parametrize("case", event_cases, ids=[case["id"] for case in event_cases])
def test_causal_fixtures(case: dict) -> None:
    events = events_for(scenario(case["id"]))
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ObservationError) as caught:
            reconstruct(events)
        assert caught.value.code == expected["error"]
        if "producer" in expected:
            assert caught.value.producer == expected["producer"]
            assert caught.value.event_sequence == expected["event_sequence"]
        return
    result = reconstruct(events)
    if "operation_count" in expected:
        assert len(result.of_kind("operation")) == expected["operation_count"]
    if "invocation_count" in expected:
        assert len(result.of_kind("invocation")) == expected["invocation_count"]
    for relation, expected_key in (
        ("operation_to_invocation", "operation_to_invocation"),
        ("invocation_to_graph_activity", "invocation_to_graph_activity"),
    ):
        if expected_key in expected:
            actual = [
                [list(edge.origin.components), list(edge.destination.components)]
                for edge in result.edges
                if edge.relation == relation
                and isinstance(edge.origin, EntityIdentity)
                and isinstance(edge.destination, EntityIdentity)
            ]
            assert actual == expected[expected_key]
    if "graph_to_operations" in expected:
        actual = []
        for graph, operations in expected["graph_to_operations"]:
            actual_operations = sorted(
                list(edge.destination.components)
                for edge in result.edges
                if edge.relation == "graph_to_operation"
                and isinstance(edge.origin, EntityIdentity)
                and isinstance(edge.destination, EntityIdentity)
                and list(edge.origin.components) == graph
            )
            assert actual_operations == sorted(operations)
            actual.append(graph)
        assert len(actual) == len({edge.origin for edge in result.edges if edge.relation == "graph_to_operation"})
    assert set(expected.get("quality", [])) <= result.quality.codes
    for edge in result.edges:
        assert edge.sources
        for source in edge.sources:
            assert source.identity in result.events
            assert source.process_creation_id
    for interval in result.intervals:
        if interval.duration_ns is not None:
            assert interval.begin is not None and interval.end is not None
            assert interval.begin.record.clock_domain_id == interval.end.record.clock_domain_id
    if "operation_ids" in expected:
        assert sorted(list(entity.identity.components) for entity in result.of_kind("operation")) == sorted(
            expected["operation_ids"]
        )
    if "participant_ranks" in expected:
        assert (
            sorted(
                {
                    event.record.instance_rank
                    for event in result.events.values()
                    if event.record.has(PresenceBit.INSTANCE_RANK)
                }
            )
            == expected["participant_ranks"]
        )
    if "participant_pes" in expected:
        assert (
            sorted({event.record.pe for event in result.events.values() if event.record.has(PresenceBit.PE)})
            == expected["participant_pes"]
        )
    if "graph_executions" in expected:
        actual_graphs = sorted(
            list(entity.identity.components)
            for entity in result.of_kind("graph_execution")
            if entity.identity.components[0] == 1
        )
        assert actual_graphs == expected["graph_executions"]
    if "template_ids" in expected:
        actual_templates = sorted(
            list(entity.identity.components)
            for entity in result.of_kind("graph_template")
            if entity.identity.components[0] == 1
        )
        assert actual_templates == expected["template_ids"]
    if "dp_rank_payload_rows" in expected:
        actual_rows = {
            event.record.dp_rank_payload_rows
            for event in result.events.values()
            if event.record.has(PresenceBit.DP_RANK_PAYLOAD_ROWS)
        }
        rows = expected["dp_rank_payload_rows"]
        if isinstance(rows, list):
            assert actual_rows == set(rows)
        else:
            assert rows in actual_rows
            assert any(
                event.record.has(PresenceBit.DP_RANK_PAYLOAD_ROWS)
                and event.record.dp_rank_payload_rows == rows
                and event.record.forward_mode == 4
                for event in result.events.values()
            )
    assert all(
        interval.begin is None
        or interval.end is None
        or interval.begin.source.identity.producer == interval.end.source.identity.producer
        for interval in result.intervals
    )
    # SGLang Host return never supplies a Device activity endpoint.
    assert not any(
        interval.name == "device_activity" and interval.observation.entity.components[0] == 1
        for interval in result.intervals
    )


def test_input_order_does_not_create_relations() -> None:
    events = events_for(scenario("normal_dense"))
    forward = reconstruct(events)
    backward = reconstruct(reversed(events))
    assert forward.edges == backward.edges
    assert forward.intervals == backward.intervals


def test_duplicate_retains_every_location() -> None:
    event = events_for(scenario("normal_dense"))[0]
    duplicate = replace(event, source=replace(event.source, offset=999))
    result = reconstruct((event, duplicate))
    assert result.event_sources[event.source.identity] == (event.source, duplicate.source)
    assert len(result.events) == 1


def test_raw_zero_and_reversed_interval_are_not_hidden() -> None:
    events = list(events_for(scenario("normal_dense")))
    for index, event in enumerate(events):
        if event.record.kind == 100:
            events[index] = replace(event, record=event.record.model_copy(update={"raw_time_ns": 0}))
        if event.record.kind == 101:
            events[index] = replace(event, record=event.record.model_copy(update={"raw_time_ns": 500}))
    result = reconstruct(events)
    assert any(interval.name == "request_staging" and interval.duration_ns == 500 for interval in result.intervals)
    assert "temporal_conflict" in result.quality.codes


def test_graph_failure_does_not_become_successful_execution() -> None:
    event = next(event for event in events_for(scenario("normal_dense")) if event.record.kind == 505)
    failed = replace(event, record=event.record.model_copy(update={"kind": 506, "data4": 3}))
    result = reconstruct((failed,))
    assert len(result.intervals) == 1
    interval = result.intervals[0]
    assert interval.status == "unavailable" and interval.duration_ns is None
    assert interval.begin is None and interval.end == failed
    assert "graph_error" in result.quality.codes


@pytest.mark.parametrize("failed", [False, True])
def test_graph_interval_accepts_raw_event_kinds(failed: bool) -> None:
    events = tuple(
        event
        for event in events_for(scenario("normal_dense"))
        if event.record.kind in (EventKind.GRAPH_HOST_CALL_BEGIN, EventKind.GRAPH_HOST_CALL_END)
    )
    if failed:
        events = tuple(
            replace(
                event,
                record=event.record.model_copy(
                    update={"kind": int(EventKind.GRAPH_HOST_CALL_ERROR), "data4": int(GraphError.HOST_CALL_FAILED)}
                ),
            )
            if event.record.kind == EventKind.GRAPH_HOST_CALL_END
            else event
            for event in events
        )
    result = reconstruct(events)
    observation = next(iter(result.of_kind("graph_execution")[0].observations.values()))
    interval = phase_interval(
        observation, "host_call", int(EventKind.GRAPH_HOST_CALL_BEGIN), int(EventKind.GRAPH_HOST_CALL_END)
    )
    assert interval.status == ("unavailable" if failed else "complete")
    assert interval.begin is not None and interval.end is not None
    assert interval.end.record.kind == (EventKind.GRAPH_HOST_CALL_ERROR if failed else EventKind.GRAPH_HOST_CALL_END)
    assert interval.reasons == (("graph_error",) if failed else ())


def test_unavailable_graph_context_does_not_produce_normal_duration_or_edges() -> None:
    events = tuple(event for event in events_for(scenario("normal_dense")) if event.record.kind in (504, 505))
    events = tuple(
        replace(event, record=event.record.model_copy(update={"data4": 4})) if event.record.kind == 505 else event
        for event in events
    )
    result = reconstruct(events)
    assert "context_unavailable" in result.quality.codes
    assert result.intervals[0].status == "unavailable"
    assert result.intervals[0].duration_ns is None
    assert not any(edge.relation in ("graph_to_operation", "operation_to_invocation") for edge in result.edges)


def test_semantic_duplicates_are_distinct_from_boundary_conflicts() -> None:
    event = next(event for event in events_for(scenario("normal_dense")) if event.record.kind == 100)
    duplicate = replace(
        event,
        record=event.record.model_copy(update={"event_sequence": 20}),
        source=replace(event.source, identity=replace(event.source.identity, event_sequence=20)),
    )
    result = reconstruct((event, duplicate))
    assert "duplicate_event" in result.quality.codes
    assert len(result.events) == 2
    conflict = replace(duplicate, record=duplicate.record.model_copy(update={"raw_time_ns": 101}))
    with pytest.raises(ObservationError) as caught:
        reconstruct((event, conflict))
    assert caught.value.code == "event_conflict"
