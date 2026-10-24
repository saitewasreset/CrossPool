"""Identity-based offline reconstruction, independent of calibrated time."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from xpool.observability.applicability import boundary_applicability
from xpool.observability.artifacts import ArtifactSet, AttachmentEvidence, resolve_attachments
from xpool.observability.codec import event_error, kind_names, presence_groups, validate_event
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import (
    CalibrationSample,
    Event,
    EventIdentity,
    Finding,
    Metadata,
    QualityCheckpoint,
    Source,
)
from xpool.observability.wire import (
    EventKind,
    EventRole,
    GraphError,
    PresenceBit,
    SampleFailure,
)

__all__ = [
    "CausalEdge",
    "Entity",
    "EntityIdentity",
    "Interval",
    "Observation",
    "QualityReport",
    "Reconstruction",
    "reconstruct",
]

type EntityKind = Literal["operation", "invocation", "graph_template", "graph_execution", "lease", "attachment"]


@dataclass(frozen=True, order=True)
class EntityIdentity:
    """Run-scoped contract identity; components never contain local trace IDs."""

    run_id: str
    kind: EntityKind
    components: tuple[int, ...]


@dataclass(frozen=True, order=True)
class ObservationIdentity:
    """A local site/role's observation, isolated from other participants."""

    entity: EntityIdentity
    producer: int
    role: EventRole


@dataclass
class Observation:
    """Retained boundaries and repeated facts at one actual observing site."""

    identity: ObservationIdentity
    facts: dict[str, int] = field(default_factory=dict)
    events: dict[int, Event] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    fact_sources: dict[str, Event] = field(default_factory=dict)

    def add(self, event: Event, findings: list[Finding], *, boundary: bool = True) -> None:
        """Merge equal repeated facts and retain distinct boundary evidence."""
        record = event.record
        previous = self.events.get(record.kind)

        if previous is not None:
            # event_sequence is transport identity, not a semantic fact.
            before = previous.record.model_dump(exclude={"event_sequence"})
            after = record.model_dump(exclude={"event_sequence"})

            if before != after:
                raise event_error(
                    ErrorCode.EVENT_CONFLICT,
                    "conflicting boundary at the same observation",
                    event,
                    sources=(previous.source, event.source),
                )

            findings.append(Finding("duplicate_event", "identical semantic boundary", (previous.source, event.source)))
        elif boundary:
            self.events[record.kind] = event

        for bit, names in presence_groups.items():
            if record.has(bit):
                for name in names:
                    value = getattr(record, name)
                    if name in self.facts and self.facts[name] != value:
                        raise event_error(
                            ErrorCode.IDENTITY_CONFLICT,
                            f"conflicting fact {name}",
                            event,
                            sources=(self.fact_sources[name].source, event.source),
                        )
                    self.facts[name] = value
                    previous_source = self.fact_sources.get(name)
                    if previous_source is None or event.source.identity < previous_source.source.identity:
                        self.fact_sources[name] = event

        self.sources.append(event.source)


@dataclass
class Entity:
    """One protocol or Graph identity with independently retained site observations."""

    identity: EntityIdentity
    observations: dict[ObservationIdentity, Observation] = field(default_factory=dict)
    facts: dict[str, int] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    fact_sources: dict[str, Event] = field(default_factory=dict)


@dataclass(frozen=True)
class CausalEdge:
    """An explicitly evidenced relation; source events explain why it exists."""

    relation: str
    origin: EntityIdentity | EventIdentity
    destination: EntityIdentity | EventIdentity
    sources: tuple[Source, ...]


@dataclass(frozen=True)
class Interval:
    """A same-domain observed interval, or an explicit unavailable boundary pair."""

    observation: ObservationIdentity
    name: str
    begin: Event | None
    end: Event | None
    status: Literal["complete", "partial", "conflict", "unavailable"]
    duration_ns: int | None
    reasons: tuple[str, ...]

    @property
    def sources(self) -> tuple[Source, ...]:
        """Return the concrete observed endpoints, including partial evidence."""
        return tuple(event.source for event in (self.begin, self.end) if event is not None)


@dataclass(frozen=True)
class QualityReport:
    """Latest cumulative checkpoints plus local findings, without invented totals."""

    findings: tuple[Finding, ...]
    checkpoints: dict[tuple[str, int], QualityCheckpoint]
    # None means no closed exact checkpoint proves final totals.
    final_counters: dict[tuple[str, int], dict[str, int] | None]

    @property
    def codes(self) -> frozenset[str]:
        """Return structured finding categories without parsing diagnostic text."""
        return frozenset(finding.code for finding in self.findings)


@dataclass(frozen=True)
class Reconstruction:
    """Evidence inventory consumed by offline export and snapshot projection."""

    events: dict[EventIdentity, Event]
    event_sources: dict[EventIdentity, tuple[Source, ...]]
    entities: dict[EntityIdentity, Entity]
    edges: tuple[CausalEdge, ...]
    intervals: tuple[Interval, ...]
    metadata: tuple[Metadata, ...]
    quality: QualityReport
    calibration_events: tuple[Event, ...]
    attachments: tuple[AttachmentEvidence, ...]
    calibration_samples: tuple[CalibrationSample, ...]

    def of_kind(self, kind: str) -> tuple[Entity, ...]:
        """Return all retained identities of a contract entity family."""
        return tuple(entity for key, entity in self.entities.items() if key.kind == kind)


# The observation artifact contract fixes event/phase ownership independently of input order.
transport_pairs = (
    ("request_staging", EventKind.REQUEST_STAGING_STARTED, EventKind.REQUEST_STAGING_COMPLETED),
    ("request_publication", EventKind.REQUEST_STAGING_COMPLETED, EventKind.REQUEST_PUBLISHED),
    ("request_processing", EventKind.REQUEST_OBSERVED, EventKind.RESULT_PUBLISHED),
    ("execution", EventKind.EXECUTION_STARTED, EventKind.EXECUTION_COMPLETED),
    ("output_copy", EventKind.RESULT_OBSERVED, EventKind.OUTPUT_COPIED),
    ("result_acknowledgement", EventKind.OUTPUT_COPIED, EventKind.RESULT_ACKNOWLEDGED),
    ("result_acknowledged_total", EventKind.REQUEST_STAGING_STARTED, EventKind.RESULT_ACKNOWLEDGED),
    ("closed_total", EventKind.REQUEST_STAGING_STARTED, EventKind.CLOSED),
)
fabric_pairs = {
    EventRole.ATNAGENT: (
        ("admission_wait", EventKind.SUBMISSION_PUBLISHED, EventKind.ADMISSION_OBSERVED),
        ("output_wait", EventKind.ADMISSION_OBSERVED, EventKind.OUTPUT_COMMIT_OBSERVED),
        ("total", EventKind.SUBMISSION_PREPARED, EventKind.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED),
    ),
    EventRole.COORDINATOR: (
        ("scheduling", EventKind.ENQUEUED, EventKind.SCHEDULED),
        ("execution", EventKind.LANE_EXECUTION_PUBLISHED, EventKind.FFNAGENT_COMPLETIONS_OBSERVED),
        ("active_total", EventKind.SCHEDULED, EventKind.LANE_RELEASED),
        ("total", EventKind.ENQUEUED, EventKind.LANE_RELEASED),
    ),
    EventRole.FFNAGENT: (
        ("input_wait", EventKind.LANE_EXECUTION_OBSERVED, EventKind.INPUT_READY_OBSERVED),
        ("compute", EventKind.COMPUTE_STARTED, EventKind.COMPUTE_COMPLETED),
        ("total", EventKind.LANE_EXECUTION_OBSERVED, EventKind.COMPLETION_PUBLISHED),
    ),
}
graph_pairs = (
    ("capture", EventKind.GRAPH_CAPTURE_BEGIN, EventKind.GRAPH_CAPTURE_END),
    ("installation", EventKind.GRAPH_INSTALL_BEGIN, EventKind.GRAPH_INSTALL_END),
    ("host_call", EventKind.GRAPH_HOST_CALL_BEGIN, EventKind.GRAPH_HOST_CALL_END),
    ("device_activity", EventKind.GRAPH_DEVICE_ACTIVITY_BEGIN, EventKind.GRAPH_DEVICE_ACTIVITY_END),
)
graph_failure_kinds: dict[int, EventKind] = {
    EventKind.GRAPH_CAPTURE_END: EventKind.GRAPH_CAPTURE_ERROR,
    EventKind.GRAPH_INSTALL_END: EventKind.GRAPH_INSTALL_ERROR,
    EventKind.GRAPH_HOST_CALL_END: EventKind.GRAPH_HOST_CALL_ERROR,
}


def phase_interval(observation: Observation, name: str, begin_kind: int, end_kind: int) -> Interval:
    begin = observation.events.get(begin_kind)
    end = observation.events.get(end_kind)

    failure_kind = graph_failure_kinds.get(end_kind)
    failure = observation.events.get(failure_kind) if failure_kind is not None else None

    if failure is not None:
        if end is not None:
            raise event_error(ErrorCode.EVENT_CONFLICT, "Graph has both success and failure boundaries", failure)

        return Interval(observation.identity, name, begin, failure, "unavailable", None, ("graph_error",))

    if observation.identity.entity.kind in ("graph_template", "graph_execution") and any(
        event is not None and event.record.data4 != GraphError.NONE for event in (begin, end)
    ):
        return Interval(observation.identity, name, begin, end, "unavailable", None, ("graph_error",))

    if begin is None or end is None:
        return Interval(observation.identity, name, begin, end, "partial", None, ("event_missing",))

    if begin.record.clock_domain_id != end.record.clock_domain_id:
        return Interval(observation.identity, name, begin, end, "unavailable", None, ("clock_domain_mismatch",))

    if end.record.raw_time_ns < begin.record.raw_time_ns:
        return Interval(observation.identity, name, begin, end, "conflict", None, ("temporal_conflict",))

    return Interval(
        observation.identity, name, begin, end, "complete", end.record.raw_time_ns - begin.record.raw_time_ns, ()
    )


def protocol_temporal_conflicts(observation: Observation) -> tuple[Finding, ...]:
    """Check local Hook precedence even when no legacy phase spans two Hooks."""
    role = observation.identity.role
    if observation.identity.entity.kind == "operation":
        order = (
            (
                EventKind.REQUEST_STAGING_STARTED,
                EventKind.REQUEST_STAGING_COMPLETED,
                EventKind.REQUEST_PUBLISHED,
                EventKind.RESULT_OBSERVED,
                EventKind.OUTPUT_COPIED,
                EventKind.RESULT_ACKNOWLEDGED,
                EventKind.CLOSED,
            )
            if role == EventRole.INSTANCE
            else (
                EventKind.REQUEST_OBSERVED,
                EventKind.EXECUTION_STARTED,
                EventKind.EXECUTION_COMPLETED,
                EventKind.RESULT_PUBLISHED,
            )
        )
    elif role == EventRole.ATNAGENT:
        order = tuple(range(EventKind.SUBMISSION_PREPARED, EventKind.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED + 1))
    elif role == EventRole.COORDINATOR:
        order = tuple(range(EventKind.ENQUEUED, EventKind.LANE_RELEASED + 1))
    else:
        # The publisher releases routing from inside compute; peers observe it
        # before compute. Enum ordinals alone do not express this partial order.
        order = (
            EventKind.LANE_EXECUTION_OBSERVED,
            EventKind.INPUT_READY_OBSERVED,
            EventKind.ROUTING_METADATA_OBSERVED,
            EventKind.COMPUTE_STARTED,
            EventKind.ROUTING_METADATA_PUBLISHED,
            EventKind.COMPUTE_COMPLETED,
            EventKind.PARTIAL_READY_PUBLISHED,
            EventKind.PEER_PARTIALS_READY_OBSERVED,
            EventKind.COMPLETION_PUBLISHED,
        )
    retained = [observation.events[k] for k in order if k in observation.events]
    return tuple(
        Finding("temporal_conflict", "local protocol Hook order reversed", (begin.source, end.source))
        for begin, end in pairwise(retained)
        if begin.record.clock_domain_id == end.record.clock_domain_id
        and begin.record.raw_time_ns > end.record.raw_time_ns
    )


def check_metadata_revisions(metadata: Iterable[Metadata]) -> tuple[Metadata, ...]:
    """Reject changed immutable references across revisions, processes and chunks."""
    retained: dict[tuple[str, int, int], Metadata] = {}
    definitions: dict[tuple[str, str, tuple[int, ...]], str] = {}

    for document in metadata:
        identity = (document.run_id, document.producer_id, document.revision)

        if identity in retained and retained[identity] != document:
            raise ObservationError(
                ErrorCode.IDENTITY_CONFLICT, "conflicting metadata revision", producer=document.producer_id
            )

        if identity in retained:
            continue

        retained[identity] = document

        for family, entries in (
            ("producer", [((x.id,), x.model_dump_json()) for x in document.producers]),
            ("generation", [((x.id,), x.model_dump_json()) for x in document.generations]),
            ("clock", [((x.id,), x.model_dump_json()) for x in document.clock_domains]),
            ("endpoint", [((x.owner_producer, x.endpoint_sequence), x.model_dump_json()) for x in document.endpoints]),
            (
                "template",
                [((x.owner_producer, x.template_sequence), x.model_dump_json()) for x in document.graph_templates],
            ),
            ("artifact", [((document.producer_id, x.id), x.model_dump_json()) for x in document.artifacts]),
            (
                "policy",
                [((document.producer_id,), document.model_dump_json(include={"capabilities", "configuration"}))],
            ),
        ):
            for components, value in entries:
                key = (document.run_id, family, components)
                if key in definitions and definitions[key] != value:
                    raise ObservationError(
                        ErrorCode.IDENTITY_CONFLICT,
                        f"changed immutable {family} {components}",
                        producer=document.producer_id,
                    )
                definitions[key] = value

    return tuple(retained.values())


def reconstruct(
    events: Iterable[Event] | ArtifactSet,
    *,
    metadata: Iterable[Metadata] = (),
    checkpoints: Iterable[QualityCheckpoint] = (),
) -> Reconstruction:
    """Reconstruct entities and causality exclusively from protocol identities.

    Missing evidence produces local findings. Malformed events, conflicting
    identities and conflicting repeated facts raise ObservationError.
    """
    artifact_set = events if isinstance(events, ArtifactSet) else None

    event_input: Iterable[Event]
    if isinstance(events, ArtifactSet):
        metadata = (*metadata, *events.metadata)
        checkpoints = (*checkpoints, *events.checkpoints)
        event_input = events.events
    else:
        event_input = events

    retained: dict[EventIdentity, Event] = {}
    locations: dict[EventIdentity, list[Source]] = defaultdict(list)
    documents = list(metadata)
    findings: list[Finding] = list(artifact_set.findings) if artifact_set else []
    attachments = artifact_set.attachments if artifact_set else ()

    for event in event_input:
        validate_event(event)
        identity = event.source.identity

        if (identity.run_id, identity.producer) != (event.metadata.run_id, event.metadata.producer_id):
            raise event_error(ErrorCode.IDENTITY_CONFLICT, "event provenance disagrees with metadata", event)

        if (
            identity.event_sequence != event.record.event_sequence
            or event.source.process_creation_id != event.metadata.process.creation_id
            or event.source.clock_domain_id != event.record.clock_domain_id
            or event.source.endpoint
            != (
                (event.record.endpoint_owner, event.record.endpoint_sequence)
                if event.record.has(PresenceBit.ENDPOINT)
                else None
            )
            or event.source.instance_rank
            != (event.record.instance_rank if event.record.has(PresenceBit.INSTANCE_RANK) else None)
            or event.source.pe != (event.record.pe if event.record.has(PresenceBit.PE) else None)
        ):
            raise event_error(ErrorCode.IDENTITY_CONFLICT, "source/event sequence mismatch", event)

        documents.append(event.metadata)
        previous = retained.get(identity)

        if previous is not None:
            if previous.record != event.record:
                raise event_error(
                    ErrorCode.EVENT_CONFLICT,
                    "event identity has different bytes",
                    event,
                    sources=(previous.source, event.source),
                )
            findings.append(Finding("duplicate_event", "repeated event identity", (previous.source, event.source)))
        else:
            retained[identity] = event

        locations[identity].append(event.source)

    documents_tuple = check_metadata_revisions(documents)

    if artifact_set is None:
        attachments, attachment_findings = resolve_attachments((), None, tuple(retained.values()), documents_tuple)
        findings.extend(attachment_findings)

    entities: dict[EntityIdentity, Entity] = {}
    edges: dict[tuple[str, EntityIdentity | EventIdentity, EntityIdentity | EventIdentity], CausalEdge] = {}
    samples: list[Event] = []

    def link(
        relation: str,
        origin: EntityIdentity | EventIdentity,
        destination: EntityIdentity | EventIdentity,
        sources: tuple[Source, ...],
    ) -> None:
        key = (relation, origin, destination)
        previous = edges.get(key)
        combined = (
            previous.sources + tuple(source for source in sources if source not in previous.sources)
            if previous
            else sources
        )
        edges[key] = CausalEdge(relation, origin, destination, combined)

    for event in sorted(retained.values(), key=lambda item: item.source.identity):
        r = event.record
        run_id = event.source.identity.run_id
        keys: dict[str, EntityIdentity] = {}

        identity_groups: tuple[tuple[PresenceBit, EntityKind, tuple[int, ...]], ...] = (
            (PresenceBit.OPERATION, "operation", (r.endpoint_owner, r.endpoint_sequence, r.operation_sequence)),
            (PresenceBit.INVOCATION, "invocation", (r.generation, r.instance_index, r.invocation_sequence)),
            (PresenceBit.GRAPH_TEMPLATE, "graph_template", (r.graph_owner, r.graph_template)),
            (PresenceBit.GRAPH_EXECUTION, "graph_execution", (r.graph_owner, r.graph_execution)),
            (PresenceBit.LEASE, "lease", (r.generation, r.lane_index, r.lease_sequence)),
            (PresenceBit.ATTACHMENT, "attachment", (r.attachment_owner, r.attachment_sequence)),
        )

        for bit, kind, components in identity_groups:
            if r.has(bit):
                identity = EntityIdentity(run_id, kind, components)
                keys[kind] = identity
                entity = entities.setdefault(identity, Entity(identity))
                entity.sources.append(event.source)
                stable = ("layer_ordinal", "payload_rows", "output_requirement") if kind == "invocation" else ()

                if kind == "operation":
                    stable += (
                        "generation",
                        "instance_rank",
                        "layer_ordinal",
                        "payload_rows",
                        "forward_mode",
                        "output_requirement",
                        "dp_row_layout",
                    )

                for name in stable:
                    if any(r.has(group) and name in names for group, names in presence_groups.items()):
                        value = getattr(r, name)
                        if name in entity.facts and entity.facts[name] != value:
                            local_observation = entity.observations.get(
                                ObservationIdentity(entity.identity, event.source.identity.producer, EventRole(r.role))
                            )
                            previous_fact = (
                                local_observation.fact_sources.get(name, entity.fact_sources[name])
                                if local_observation is not None
                                else entity.fact_sources[name]
                            )
                            raise event_error(
                                ErrorCode.IDENTITY_CONFLICT,
                                f"conflicting fact {name}",
                                event,
                                sources=(previous_fact.source, event.source),
                            )
                        entity.facts[name] = value
                        entity.fact_sources.setdefault(name, event)

        # Only the owning family mutates a local observation. Association facts
        # remain evidence for edges and cannot manufacture a Fabric boundary.
        family = (
            "operation"
            if EventKind.REQUEST_STAGING_STARTED <= r.kind <= EventKind.CLOSED
            else "invocation"
            if EventKind.SUBMISSION_PREPARED <= r.kind <= EventKind.COMPLETION_PUBLISHED
            else "graph_template"
            if r.kind
            in (
                EventKind.GRAPH_CAPTURE_BEGIN,
                EventKind.GRAPH_CAPTURE_END,
                EventKind.GRAPH_CAPTURE_ERROR,
                EventKind.GRAPH_TEMPLATE_INSTALLED,
                EventKind.GRAPH_TEMPLATE_RETIRED,
                EventKind.GRAPH_INSTALL_BEGIN,
                EventKind.GRAPH_INSTALL_END,
                EventKind.GRAPH_INSTALL_ERROR,
            )
            else "graph_execution"
            if EventKind.GRAPH_HOST_CALL_BEGIN <= r.kind <= EventKind.GRAPH_DEVICE_ACTIVITY_END
            else None
        )

        if family is not None:
            entity = entities[keys[family]]
            observation_id = ObservationIdentity(entity.identity, event.source.identity.producer, EventRole(r.role))
            observation = entity.observations.setdefault(observation_id, Observation(observation_id))
            observation.add(event, findings)

        # Auxiliary Graph evidence recovers only explicitly carried local facts.
        # It never enters the protocol boundary map or borrows another site's ID.
        if family == "graph_execution" and r.has(PresenceBit.LOCAL_TRACE_ID):
            auxiliary_family = "invocation" if r.role == EventRole.FFNAGENT else "operation"
            auxiliary_key = keys.get(auxiliary_family)
            if auxiliary_key is not None:
                local_site = auxiliary_family == "invocation" or any(
                    (e.owner_producer, e.endpoint_sequence) == auxiliary_key.components[:2]
                    and any(site.producer == event.source.identity.producer for site in e.sites)
                    for e in event.metadata.endpoints
                )
                if local_site:
                    entity = entities[auxiliary_key]
                    observation_id = ObservationIdentity(
                        auxiliary_key, event.source.identity.producer, EventRole(r.role)
                    )
                    observation = entity.observations.setdefault(observation_id, Observation(observation_id))
                    observation.add(event, findings, boundary=False)

        for relation, left, right, condition in (
            ("operation_to_invocation", "operation", "invocation", r.kind == EventKind.TRANSPORT_FABRIC_ASSOCIATED),
            ("graph_to_operation", "graph_execution", "operation", True),
            (
                "invocation_to_graph_activity",
                "invocation",
                "graph_execution",
                r.kind in (EventKind.GRAPH_DEVICE_ACTIVITY_BEGIN, EventKind.GRAPH_DEVICE_ACTIVITY_END),
            ),
            ("execution_of_template", "graph_execution", "graph_template", True),
            ("invocation_lease", "invocation", "lease", True),
            ("invocation_attachment", "invocation", "attachment", r.kind == EventKind.ROUTING_PUBLISHED),
        ):
            if condition and left in keys and right in keys:
                link(relation, keys[left], keys[right], (event.source,))

        if r.kind in (EventKind.CALIBRATION_SAMPLE_SUCCEEDED, EventKind.CALIBRATION_SAMPLE_FAILED):
            samples.append(event)
            if r.kind == EventKind.CALIBRATION_SAMPLE_FAILED:
                findings.append(Finding("sample_failed", "paired sampling attempt failed", (event.source,)))

        if EventKind.GRAPH_CAPTURE_BEGIN <= r.kind <= EventKind.GRAPH_INSTALL_ERROR and r.data4:
            findings.append(
                Finding(GraphError(r.data4).name.lower(), "Graph evidence unavailable or failed", (event.source,))
            )

    # An Operation or execution has one association target; repeated evidence
    # may add sources, but must not silently remap the same identity.
    targets: dict[tuple[str, EntityIdentity | EventIdentity], EntityIdentity | EventIdentity] = {}

    for edge in edges.values():
        if edge.relation in (
            "operation_to_invocation",
            "execution_of_template",
            "invocation_lease",
            "invocation_attachment",
        ):
            # An Invocation can legitimately have more than one Routing publisher.
            if edge.relation == "invocation_attachment":
                continue
            key = (edge.relation, edge.origin)
            if key in targets and targets[key] != edge.destination:
                source = edge.sources[0]
                raise ObservationError(
                    ErrorCode.IDENTITY_CONFLICT,
                    f"conflicting {edge.relation}",
                    path=source.path,
                    offset=source.offset,
                    producer=source.identity.producer,
                    event_sequence=source.identity.event_sequence,
                )
            targets[key] = edge.destination

    inverse_targets: dict[tuple[str, EntityIdentity | EventIdentity], EntityIdentity | EventIdentity] = {}
    for edge in edges.values():
        if edge.relation in ("graph_to_operation", "invocation_lease", "invocation_attachment"):
            key = (edge.relation, edge.destination)
            if key in inverse_targets and inverse_targets[key] != edge.origin:
                source = edge.sources[0]
                raise ObservationError(
                    ErrorCode.IDENTITY_CONFLICT,
                    f"reused {edge.relation} destination",
                    path=source.path,
                    offset=source.offset,
                    producer=source.identity.producer,
                    event_sequence=source.identity.event_sequence,
                )
            inverse_targets[key] = edge.origin

    intervals: list[Interval] = []
    for identity, entity in entities.items():
        if identity.kind == "operation":
            endpoint = next(
                e
                for m in documents_tuple
                if m.run_id == identity.run_id
                for e in m.endpoints
                if (e.owner_producer, e.endpoint_sequence) == identity.components[:2]
            )
            observed_producers = {observation.identity.producer for observation in entity.observations.values()}
            if any(site.producer not in observed_producers for site in endpoint.sites):
                findings.append(
                    Finding("event_missing", "Endpoint site observation unavailable", tuple(entity.sources))
                )
        if identity.kind == "graph_execution" and not entity.observations:
            findings.append(Finding("event_missing", "Graph activity boundaries unavailable", tuple(entity.sources)))
        if identity.kind == "operation" and ("operation_to_invocation", identity) not in targets:
            # A retained explicit rejection/closure can legitimately precede Fabric.
            rejected = any(EventKind.CLOSED in observation.events for observation in entity.observations.values())
            if not rejected:
                findings.append(
                    Finding("association_missing", "no retained Fabric-entry association", tuple(entity.sources))
                )
        for observation in entity.observations.values():
            if identity.kind == "operation":
                role = observation.identity.role
                pairs = tuple(
                    pair
                    for pair in transport_pairs
                    if (
                        role == EventRole.INSTANCE
                        and pair[1] not in (EventKind.REQUEST_OBSERVED, EventKind.EXECUTION_STARTED)
                    )
                    or (
                        role == EventRole.ATNAGENT
                        and pair[1] in (EventKind.REQUEST_OBSERVED, EventKind.EXECUTION_STARTED)
                    )
                )
                if "result_code" not in observation.facts:
                    findings.append(
                        Finding("unknown_result", "site-local result unavailable", tuple(observation.sources))
                    )
            elif identity.kind == "invocation":
                pairs = fabric_pairs[observation.identity.role]
            else:
                pairs = tuple(
                    pair
                    for pair in graph_pairs
                    if pair[1] in observation.events
                    or pair[2] in observation.events
                    or (
                        (failure_kind := graph_failure_kinds.get(pair[2])) is not None
                        and failure_kind in observation.events
                    )
                )
            if identity.kind in ("operation", "invocation"):
                findings.extend(protocol_temporal_conflicts(observation))
                applicability = boundary_applicability(observation, documents_tuple)
                missing = applicability.required - observation.events.keys()
                if missing:
                    findings.append(
                        Finding(
                            "event_missing",
                            "missing boundaries: " + ", ".join(kind_names[x] for x in sorted(missing)),
                            tuple(observation.sources),
                        )
                    )
                if applicability.unknown:
                    findings.append(
                        Finding(
                            "applicability_unknown", "conditional protocol path unresolved", tuple(observation.sources)
                        )
                    )
            for name, start, end in pairs:
                # Closed and normal acknowledged totals are alternative paths.
                if (
                    identity.kind == "operation"
                    and name == "closed_total"
                    and EventKind.CLOSED not in observation.events
                ):
                    continue
                if identity.kind in ("operation", "invocation") and (
                    start in applicability.inapplicable or end in applicability.inapplicable
                ):
                    continue
                interval = phase_interval(observation, name, start, end)
                if (
                    identity.kind in ("operation", "invocation")
                    and interval.reasons == ("event_missing",)
                    and (start in applicability.unknown or end in applicability.unknown)
                ):
                    interval = Interval(
                        interval.observation,
                        name,
                        interval.begin,
                        interval.end,
                        "partial",
                        None,
                        ("applicability_unknown",),
                    )
                intervals.append(interval)
                for reason in interval.reasons:
                    findings.append(Finding(reason, f"{name}: incomplete or invalid boundaries", interval.sources))

    # Cross-role protocol edges require the exact Invocation, matching Lease
    # where both sides carry one, and both actual observed boundaries.
    protocol_pairs = (
        (EventKind.SUBMISSION_PUBLISHED, EventKind.ENQUEUED),
        (EventKind.ADMISSION_PUBLISHED, EventKind.ADMISSION_OBSERVED),
        (EventKind.LANE_EXECUTION_PUBLISHED, EventKind.LANE_EXECUTION_OBSERVED),
        (EventKind.INPUT_READY_PUBLISHED, EventKind.INPUT_READY_OBSERVED),
        (EventKind.COMPLETION_PUBLISHED, EventKind.FFNAGENT_COMPLETIONS_OBSERVED),
        (EventKind.OUTPUT_COMMIT_PUBLISHED, EventKind.OUTPUT_COMMIT_OBSERVED),
        (EventKind.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED, EventKind.OUTPUT_ACKNOWLEDGEMENTS_OBSERVED),
    )
    for entity in entities.values():
        if entity.identity.kind != "invocation":
            continue
        by_kind: dict[int, list[Event]] = defaultdict(list)
        for observation in entity.observations.values():
            for kind, event in observation.events.items():
                by_kind[kind].append(event)
        for left, right in protocol_pairs:
            for before in by_kind[left]:
                for after in by_kind[right]:
                    a, b = before.record, after.record
                    if (
                        a.has(PresenceBit.LEASE)
                        and b.has(PresenceBit.LEASE)
                        and (a.lane_index, a.lease_sequence) != (b.lane_index, b.lease_sequence)
                    ):
                        continue
                    link("protocol", before.source.identity, after.source.identity, (before.source, after.source))

    latest: dict[tuple[str, int], QualityCheckpoint] = {}
    revisions: dict[tuple[str, int, int], QualityCheckpoint] = {}
    for checkpoint in checkpoints:
        key = (checkpoint.run_id, checkpoint.producer_id)
        rev_key = (*key, checkpoint.revision)
        if rev_key in revisions and revisions[rev_key] != checkpoint:
            raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "conflicting quality checkpoint", producer=key[1])
        revisions[rev_key] = checkpoint
    for checkpoint in sorted(revisions.values(), key=lambda item: item.revision):
        key = (checkpoint.run_id, checkpoint.producer_id)
        previous_checkpoint = latest.get(key)
        if previous_checkpoint is not None:
            if (
                checkpoint.host_time_ns < previous_checkpoint.host_time_ns
                or checkpoint.last_attempted_sequence < previous_checkpoint.last_attempted_sequence
            ) or any(
                value < previous_checkpoint.counters.model_dump()[name]
                for name, value in checkpoint.counters.model_dump().items()
            ):
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "cumulative checkpoint regressed", producer=key[1])
        latest[key] = checkpoint
    observed: dict[tuple[str, int], list[Event]] = defaultdict(list)
    for event in retained.values():
        observed[(event.source.identity.run_id, event.source.identity.producer)].append(event)
    registered = {(m.run_id, p.id) for m in documents_tuple for p in m.producers}
    for key in sorted(registered | latest.keys() | observed.keys()):
        checkpoint = latest.get(key)
        producer_events = sorted(observed[key], key=lambda item: item.record.event_sequence)
        if producer_events:
            sequences = [event.record.event_sequence for event in producer_events]
            if sequences[0] != 1 or any(b != a + 1 for a, b in pairwise(sequences)):
                findings.append(
                    Finding(
                        "event_missing",
                        "attempted sequence gaps; cause not inferred",
                        tuple(event.source for event in producer_events),
                        *key,
                    )
                )
            if (
                checkpoint is not None
                and checkpoint.tail == "complete"
                and checkpoint.exact_totals
                and checkpoint.collection_state in ("closed", "stopped")
                and checkpoint.export_state in ("closed", "failed")
            ):
                if sequences[-1] > checkpoint.last_attempted_sequence or len(sequences) > checkpoint.counters.exported:
                    raise ObservationError(
                        ErrorCode.IDENTITY_CONFLICT, "closed checkpoint contradicts retained events", producer=key[1]
                    )
                if sequences[-1] < checkpoint.last_attempted_sequence:
                    findings.append(
                        Finding("event_missing", "closing checkpoint proves a missing event suffix", (), *key)
                    )
        elif checkpoint is None:
            findings.append(
                Finding("process_artifact_missing", "no events or checkpoint for registered Producer", (), *key)
            )
        if checkpoint is None or checkpoint.tail != "complete":
            findings.append(Finding("tail_unknown", "no complete closing checkpoint", (), *key))
        if checkpoint is not None:
            if checkpoint.counter_exhausted:
                findings.append(
                    Finding("counter_exhausted", "checkpoint counters are saturated lower bounds", (), *key)
                )
            if not checkpoint.exact_totals:
                findings.append(Finding("totals_inexact", "checkpoint does not prove exact totals", (), *key))
            for counter in (
                "dropped_full",
                "abandoned",
                "post_commit_lost",
                "snapshot_operations_dropped",
                "attachment_capacity_exhausted",
                "attachment_payload_oversize",
                "attachment_not_collected",
                "attachment_publish_failed",
                "sample_failed",
                "context_unavailable",
            ):
                value = getattr(checkpoint.counters, counter)
                if value:
                    findings.append(Finding(counter, f"cumulative checkpoint count {value}", (), *key))
            if checkpoint.stop_reason:
                findings.append(Finding(checkpoint.stop_reason, "collection stopped", (), *key))
    final = {
        key: (
            checkpoint.counters.model_dump()
            if checkpoint.tail == "complete"
            and checkpoint.exact_totals
            and checkpoint.collection_state in ("closed", "stopped")
            and checkpoint.export_state in ("closed", "failed")
            else None
        )
        for key, checkpoint in latest.items()
    }
    for key in registered:
        final.setdefault(key, None)
    quality = QualityReport(tuple(findings), latest, final)
    calibrated_inputs: list[CalibrationSample] = []
    for event in samples:
        r = event.record
        domain = next(domain for domain in event.metadata.clock_domains if domain.id == r.data3)
        # validate_event established a Device domain with a positive Epoch.
        if domain.raw_epoch is None:
            raise event_error(ErrorCode.INVALID_EVENT, "sample Device Epoch unavailable", event)
        failures: dict[int, Literal["enqueue_failed", "completion_failed", "timed_out", "shutdown"]] = {
            SampleFailure.ENQUEUE_FAILED: "enqueue_failed",
            SampleFailure.COMPLETION_FAILED: "completion_failed",
            SampleFailure.TIMED_OUT: "timed_out",
            SampleFailure.SHUTDOWN: "shutdown",
        }
        succeeded = r.kind == EventKind.CALIBRATION_SAMPLE_SUCCEEDED
        calibrated_inputs.append(
            CalibrationSample(
                event.source,
                r.data4,
                r.data3,
                domain.raw_epoch,
                r.data0,
                r.data1 if succeeded else None,
                r.data2 if succeeded else None,
                "succeeded" if succeeded else "failed",
                None if succeeded else failures[r.data2],
            )
        )
    return Reconstruction(
        retained,
        {key: tuple(value) for key, value in locations.items()},
        entities,
        tuple(edges.values()),
        tuple(intervals),
        documents_tuple,
        quality,
        tuple(samples),
        attachments,
        tuple(calibrated_inputs),
    )
