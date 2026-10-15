"""Explicit-domain earliest-N projections of retained offline evidence."""

from dataclasses import dataclass

from xpool.devkit.fabric_observer import (
    FabricAtnAgentRecordJson,
    FabricCoordinatorRecordJson,
    FabricFfnAgentRecordJson,
    FabricRecordFieldsJson,
    FabricRecordJson,
    FabricSnapshotJson,
)
from xpool.devkit.transport_observer import (
    TransportRecordJson,
    TransportSnapshotJson,
    transport_phase_summary,
)
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import Finding, Source
from xpool.observability.reconstruction import Observation, Reconstruction, fabric_pairs, transport_pairs
from xpool.observability.wire import (
    Delivery,
    DpRowLayout,
    EventKind,
    EventRole,
    ForwardMode,
    OutputRequirement,
    ResultCode,
)

__all__ = ["LegacyProjection", "SnapshotRequest", "project_legacy_snapshots"]


@dataclass(frozen=True)
class SnapshotRequest:
    """Exact lifetime domain and original operation-attempt count supplied by its owner.

    An Endpoint selects Transport. Otherwise Generation selects Fabric. A null
    count requests an explicitly unavailable projection, never a guessed total.
    """

    run_id: str
    producer: int
    generation: int
    operations_attempted: int | None
    capacity: int = 8192
    endpoint: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.producer <= 0 or self.generation <= 0 or self.capacity <= 0:
            raise ValueError("snapshot domain and capacity must be positive")
        if self.operations_attempted is not None and not 0 <= self.operations_attempted < 2**64:
            raise ValueError("snapshot attempt count outside u64")
        if self.endpoint is not None and any(component <= 0 for component in self.endpoint):
            raise ValueError("Endpoint components must be positive")


@dataclass(frozen=True)
class LegacyProjection:
    """Unchanged legacy payload plus separately retained quality and provenance."""

    request: SnapshotRequest
    snapshot: TransportSnapshotJson | FabricSnapshotJson | None
    sources: dict[int, tuple[Source, ...]]
    findings: tuple[Finding, ...]
    unobserved_operations: int | None
    omitted_records: int


forward_names: dict[int, str] = {value: value.name.lower() for value in ForwardMode}
output_names: dict[int, str] = {value: value.name.lower() for value in OutputRequirement}
layout_names: dict[int, str] = {value: value.name.lower() for value in DpRowLayout}
delivery_names: dict[int, str] = {value: value.name.lower() for value in Delivery}
result_names: dict[int, str] = {value: value.name.lower() for value in ResultCode}
transport_times = (
    "request_staging_started",
    "request_staging_completed",
    "request_published",
    "request_observed",
    "execution_started",
    "execution_completed",
    "result_published",
    "result_observed",
    "output_copied",
    "result_acknowledged",
    "closed",
)
fabric_times = {
    EventRole.ATNAGENT: (
        "submission_prepared",
        "submission_published",
        "admission_observed",
        "input_ready_published",
        "output_commit_observed",
        "output_acknowledgement_published",
    ),
    EventRole.COORDINATOR: (
        "enqueued",
        "scheduled",
        "admission_published",
        "lane_execution_published",
        "ffnagent_completions_observed",
        "output_commit_published",
        "output_acknowledgements_observed",
        "lane_released",
    ),
    EventRole.FFNAGENT: (
        "lane_execution_observed",
        "input_ready_observed",
        "routing_metadata_published",
        "routing_metadata_observed",
        "compute_started",
        "compute_completed",
        "partial_ready_published",
        "peer_partials_ready_observed",
        "completion_published",
    ),
}


def legacy_durations(observation: Observation, findings: list[Finding]) -> dict[str, int]:
    pairs = (
        transport_pairs if observation.identity.entity.kind == "operation" else fabric_pairs[observation.identity.role]
    )
    durations: dict[str, int] = {}

    for name, start_kind, end_kind in pairs:
        start, end = observation.events.get(start_kind), observation.events.get(end_kind)
        if start is None or end is None:
            continue

        sources = (start.source, end.source)

        if start.record.clock_domain_id != end.record.clock_domain_id:
            findings.append(Finding("clock_domain_mismatch", f"legacy {name} unavailable", sources))
        elif start.record.raw_time_ns == 0 or end.record.raw_time_ns == 0:
            findings.append(Finding("legacy_zero_timestamp", f"legacy {name} unavailable", sources))
        elif end.record.raw_time_ns < start.record.raw_time_ns:
            findings.append(Finding("temporal_conflict", f"legacy {name} reversed", sources))
        else:
            durations[name] = end.record.raw_time_ns - start.record.raw_time_ns
    
    for event in observation.events.values():
        if event.record.raw_time_ns == 0:
            findings.append(
                Finding("legacy_zero_timestamp", "recorded raw zero projects as legacy zero", (event.source,))
            )
    
    return durations


def transport_record(observation: Observation, findings: list[Finding]) -> TransportRecordJson | None:
    facts = observation.facts

    required = (
        "local_trace_id",
        "payload_rows",
        "layer_ordinal",
        "forward_mode",
        "output_requirement",
        "dp_row_layout",
    )

    if any(name not in facts for name in required):
        findings.append(
            Finding("legacy_fact_missing", "Transport scalar facts unavailable", tuple(observation.sources))
        )
        return None
    
    times = {
        name: observation.events[kind].record.raw_time_ns if kind in observation.events else 0
        for kind, name in enumerate(transport_times, EventKind.REQUEST_STAGING_STARTED)
    }

    if "result_code" not in facts:
        findings.append(Finding("unknown_result", "site-local result unavailable", tuple(observation.sources)))
    
    return TransportRecordJson(
        trace_id=facts["local_trace_id"],
        payload_rows=facts["payload_rows"],
        layer_ordinal=facts["layer_ordinal"],
        forward_mode=forward_names[facts["forward_mode"]],
        output_requirement=output_names[facts["output_requirement"]],
        dp_row_layout=layout_names[facts["dp_row_layout"]],
        result_code=result_names[facts.get("result_code", ResultCode.PROTOCOL_MISMATCH)],
        request_staging_started=times["request_staging_started"],
        request_staging_completed=times["request_staging_completed"],
        request_published=times["request_published"],
        request_observed=times["request_observed"],
        execution_started=times["execution_started"],
        execution_completed=times["execution_completed"],
        result_published=times["result_published"],
        result_observed=times["result_observed"],
        output_copied=times["output_copied"],
        result_acknowledged=times["result_acknowledged"],
        closed=times["closed"],
        durations_ns=legacy_durations(observation, findings),
    )


def fabric_record(observation: Observation, scheduler: str, findings: list[Finding]) -> FabricRecordJson | None:
    facts = observation.facts
    role = observation.identity.role

    required = (
        "local_trace_id",
        "instance_index",
        "invocation_sequence",
        "layer_ordinal",
        "payload_rows",
        "output_requirement",
    )

    if role == EventRole.ATNAGENT:
        required += ("dp_rank_payload_rows", "forward_mode", "dp_row_layout")
    elif role == EventRole.COORDINATOR and scheduler == "fifo":
        required += ("ready_ticket",)
    elif role == EventRole.FFNAGENT:
        required += ("lane_index", "lease_sequence", "payload_row_capacity", "delivery")
    
    if any(name not in facts for name in required):
        findings.append(Finding("legacy_fact_missing", "Fabric scalar facts unavailable", tuple(observation.sources)))
        return None
    
    fields = FabricRecordFieldsJson(
        local_trace_id=facts["local_trace_id"],
        instance_index=facts["instance_index"],
        invocation_sequence=facts["invocation_sequence"],
        layer_ordinal=facts["layer_ordinal"],
        payload_rows=facts["payload_rows"],
        output_requirement=output_names[facts["output_requirement"]],
    )
    
    start = {
        EventRole.ATNAGENT: EventKind.SUBMISSION_PREPARED,
        EventRole.COORDINATOR: EventKind.ENQUEUED,
        EventRole.FFNAGENT: EventKind.LANE_EXECUTION_OBSERVED,
    }[role]
    
    events_ns = {
        name: observation.events[kind].record.raw_time_ns if kind in observation.events else 0
        for kind, name in enumerate(fabric_times[role], start)
    }
    
    durations_ns = legacy_durations(observation, findings)
    
    if role == EventRole.ATNAGENT:
        return FabricAtnAgentRecordJson(
            **fields,
            kind="atnagent",
            dp_row_layout=layout_names[facts["dp_row_layout"]],
            facts={
                "dp_rank_payload_rows": facts["dp_rank_payload_rows"],
                "forward_mode": forward_names[facts["forward_mode"]],
                "executor_lane_index": facts.get("lane_index"),
                "executor_lease_sequence": facts.get("lease_sequence"),
            },
            events_ns=events_ns,
            durations_ns=durations_ns,
        )
    
    if role == EventRole.COORDINATOR:
        return FabricCoordinatorRecordJson(
            **fields,
            kind="coordinator",
            dp_row_layout=None,
            facts={
                "executor_lane_index": facts.get("lane_index"),
                "executor_lease_sequence": facts.get("lease_sequence"),
                "scheduler": {"policy": "fifo", "ready_ticket": facts["ready_ticket"]}
                if scheduler == "fifo"
                else {"policy": "random"},
            },
            events_ns=events_ns,
            durations_ns=durations_ns,
        )
    
    return FabricFfnAgentRecordJson(
        **fields,
        kind="ffnagent",
        dp_row_layout=None,
        facts={
            "executor_lane_index": facts["lane_index"],
            "executor_lease_sequence": facts["lease_sequence"],
            "payload_row_capacity": facts["payload_row_capacity"],
            "delivery": delivery_names[facts["delivery"]],
        },
        events_ns=events_ns,
        durations_ns=durations_ns,
    )


def project_legacy_snapshots(
    result: Reconstruction, requests: tuple[SnapshotRequest, ...]
) -> tuple[LegacyProjection, ...]:
    """Project exact legacy JSON with separate loss, missing-fact and source data.

    Raises:
        ObservationError: If a domain is unknown or attempt totals contradict evidence.
    """
    projections: list[LegacyProjection] = []
    for request in requests:
        findings: list[Finding] = []
        documents = [m for m in result.metadata if m.run_id == request.run_id]
        producer = next((p for m in documents for p in m.producers if p.id == request.producer), None)
        generation = next((g for m in documents for g in m.generations if g.id == request.generation), None)
        if producer is None or generation is None:
            raise ObservationError(ErrorCode.METADATA_UNAVAILABLE, "snapshot Producer/Generation unavailable")
        if request.operations_attempted is None:
            finding = Finding(
                "snapshot_total_unknown",
                "original operation-attempt count unavailable",
                (),
                request.run_id,
                request.producer,
            )
            projections.append(LegacyProjection(request, None, {}, (finding,), None, 0))
            continue
        observations: dict[int, Observation] = {}
        for entity in result.entities.values():
            key = entity.identity
            eligible = (
                (key.kind == "operation" and key.components[:2] == request.endpoint)
                if request.endpoint
                else (key.kind == "invocation" and key.components[0] == request.generation)
            )
            if key.run_id != request.run_id or not eligible:
                continue
            for observation in entity.observations.values():
                if observation.identity.producer != request.producer:
                    continue
                local_id = observation.facts.get("local_trace_id")
                if local_id is None:
                    continue
                if local_id > request.operations_attempted:
                    raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "snapshot attempts precede observed local ID")
                if local_id in observations:
                    raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "local snapshot ID reused in its lifetime")
                if local_id <= request.capacity:
                    observations[local_id] = observation
        sources = {local_id: tuple(observation.sources) for local_id, observation in sorted(observations.items())}
        unobserved = min(request.capacity, request.operations_attempted) - len(observations)
        dropped = max(0, request.operations_attempted - request.capacity)
        if dropped:
            findings.append(Finding("snapshot_operations_dropped", f"{dropped} local IDs exceed retained prefix"))
        if unobserved:
            findings.append(
                Finding("snapshot_operations_unobserved", f"{unobserved} eligible operations lack evidence")
            )
        omitted = 0
        if request.endpoint:
            endpoint = next(
                (
                    e
                    for m in documents
                    for e in m.endpoints
                    if (e.owner_producer, e.endpoint_sequence) == request.endpoint
                ),
                None,
            )
            if endpoint is None or endpoint.generation != request.generation:
                raise ObservationError(ErrorCode.METADATA_UNAVAILABLE, "snapshot Endpoint unavailable")
            site = next((s.site for s in endpoint.sites if s.producer == request.producer), None)
            if site is None:
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "Producer does not observe snapshot Endpoint")
            records: list[TransportRecordJson] = []
            completed = closed = 0
            for observation in sorted(observations.values(), key=lambda item: item.facts["local_trace_id"]):
                record = transport_record(observation, findings)
                if record is None:
                    omitted += 1
                    continue
                records.append(record)
                terminal = EventKind.RESULT_ACKNOWLEDGED if site == "instance" else EventKind.RESULT_PUBLISHED
                completed += terminal in observation.events
                closed += EventKind.CLOSED in observation.events
                if terminal in observation.events and EventKind.CLOSED in observation.events:
                    raise ObservationError(
                        ErrorCode.IDENTITY_CONFLICT, "snapshot has both completed and closed terminals"
                    )
            snapshot: TransportSnapshotJson | FabricSnapshotJson = TransportSnapshotJson(
                pid=producer.process.pid,
                site=site,
                model_id=endpoint.model_id,
                rank=endpoint.instance_rank,
                arena_handle_suffix=endpoint.arena_handle_suffix,
                sequence=request.operations_attempted,
                dropped=dropped,
                phase_summary=transport_phase_summary(records),
                record_counts={
                    "retained": len(records),
                    "completed": completed,
                    "closed": closed,
                    "incomplete": len(records) - completed - closed,
                },
                records=records,
            )
        else:
            fabric_records: list[FabricRecordJson] = []
            pes = {observation.facts["pe"] for observation in observations.values() if "pe" in observation.facts}
            if len(pes) != 1:
                findings.append(Finding("legacy_fact_missing", "Fabric snapshot PE unavailable or conflicting"))
                projections.append(
                    LegacyProjection(request, None, sources, tuple(findings), unobserved, len(observations))
                )
                continue
            for observation in sorted(observations.values(), key=lambda item: item.facts["local_trace_id"]):
                fabric_value = fabric_record(observation, generation.scheduler, findings)
                if fabric_value is None:
                    omitted += 1
                else:
                    fabric_records.append(fabric_value)
            snapshot = FabricSnapshotJson(
                generation=generation.fabric_generation,
                pe=next(iter(pes)),
                atnagent_count=generation.atnagent_count,
                ffnagent_count=generation.ffnagent_count,
                model_topologies=[
                    {"atn_tp_size": t.atn_tp_size, "atn_dp_size": t.atn_dp_size} for t in generation.model_topologies
                ],
                sequence=request.operations_attempted,
                dropped=dropped,
                records=fabric_records,
            )
        projections.append(LegacyProjection(request, snapshot, sources, tuple(findings), unobserved, omitted))
    return tuple(projections)
