"""Strict version-three artifact declarations and provenance values."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.wire import PresenceBit

type U64 = Annotated[int, Field(ge=0, le=2**64 - 1)]
type PositiveU64 = Annotated[int, Field(gt=0, le=2**64 - 1)]
type U32 = Annotated[int, Field(ge=0, le=2**32 - 1)]
type Role = Literal["instance", "atnagent", "coordinator", "ffnagent", "collector"]
type Capability = Literal["enabled", "disabled", "not_applicable", "unavailable"]
type AttachmentStatus = Literal[
    "published", "not_collected", "capacity_exhausted", "payload_oversize", "publish_failed", "missing"
]


class StrictModel(BaseModel):
    """An owned input rejects coercion and undeclared fields."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    @classmethod
    def parse_artifact(cls, data: bytes, path: Path, code: ErrorCode) -> Self:
        """Decode exact JSON without duplicate keys, coercion or non-finite values."""

        def unique_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
            result: dict[str, JsonValue] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON field {key}")
                result[key] = value
            return result

        def invalid_constant(value: str) -> JsonValue:
            raise ValueError(f"invalid JSON constant {value}")

        def finite_float(value: str) -> float:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError("JSON float exceeds finite representation")
            return parsed

        try:
            # JSON is a dynamic boundary; Pydantic establishes the concrete
            # schema immediately after duplicate and lexical validation.
            value = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=unique_fields,
                parse_constant=invalid_constant,
                parse_float=finite_float,
            )

            if isinstance(value, dict) and type(value.get("contract_version")) is not int:
                raise ValueError("contract_version must be an integer")

            if (
                isinstance(value, dict)
                and type(value.get("contract_version")) is int
                and value["contract_version"] != 3
            ):
                raise ObservationError(ErrorCode.UNSUPPORTED_VERSION, "unsupported artifact version", path=path)

            return cls.model_validate(value)
        except ObservationError:
            raise
        except (ValueError, UnicodeDecodeError) as exc:
            raise ObservationError(code, "invalid JSON artifact", path=path) from exc


class Process(StrictModel):
    creation_id: str
    pid: PositiveU64
    runtime_role: Role

    @model_validator(mode="after")
    def validate_uuid(self) -> Self:
        if str(UUID(self.creation_id)) != self.creation_id:
            raise ValueError("process creation UUID must be canonical")
        return self


class Producer(StrictModel):
    id: PositiveU64
    source: Literal["host", "device"]
    process: Process
    process_creation_id: str
    clock_domain_id: PositiveU64

    @model_validator(mode="after")
    def validate_process(self) -> Self:
        if self.process_creation_id != self.process.creation_id:
            raise ValueError("Producer process creation identity mismatch")
        return self


class ClockDomain(StrictModel):
    id: PositiveU64
    kind: Literal["host_monotonic", "device_globaltimer"]
    unit: Literal["ns"]
    host_creation_id: str | None = None
    process_creation_id: str | None = None
    physical_device_uuid: str | None = None
    logical_device: U32 | None = None
    raw_epoch: PositiveU64 | None = None

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        if self.kind == "host_monotonic":
            if self.host_creation_id is None or any(
                value is not None
                for value in (self.process_creation_id, self.physical_device_uuid, self.logical_device, self.raw_epoch)
            ):
                raise ValueError("invalid Host domain identity")
            UUID(self.host_creation_id)
        elif (
            any(
                value is None
                for value in (self.process_creation_id, self.physical_device_uuid, self.logical_device, self.raw_epoch)
            )
            or self.host_creation_id is not None
        ):
            raise ValueError("invalid Device domain identity")
        return self


class ModelTopology(StrictModel):
    atn_tp_size: PositiveU64
    atn_dp_size: PositiveU64
    ffn_tp_size: PositiveU64 | None = None


class GenerationModel(StrictModel):
    instance_index: U64
    model_id: str


class Generation(StrictModel):
    id: PositiveU64
    fabric_generation: str
    atnagent_count: PositiveU64
    ffnagent_count: PositiveU64
    model_topologies: list[ModelTopology]
    scheduler: Literal["fifo", "random"]
    models: list[GenerationModel]


class EndpointSite(StrictModel):
    producer: PositiveU64
    site: Literal["instance", "atnagent"]


class Endpoint(StrictModel):
    owner_producer: PositiveU64
    endpoint_sequence: PositiveU64
    generation: PositiveU64
    instance_index: U64
    instance_rank: U32
    model_id: str
    arena_handle_suffix: str
    sites: list[EndpointSite]
    supersedes_endpoint_sequence: PositiveU64 | None = None


class GraphTemplate(StrictModel):
    owner_producer: PositiveU64
    template_sequence: PositiveU64
    kind: str
    backend_class: str | None = None
    signature_index: U64 | None = None
    lane_index: U64 | None = None
    compute_templates: list[PositiveU64] = Field(default_factory=list)


class Artifact(StrictModel):
    id: PositiveU64
    kind: str
    status: Literal["published", "unavailable", "missing", "publish_failed"]
    path: str | None
    sha256: str | None


class Capabilities(StrictModel):
    transport: Capability
    fabric: Capability
    graph: Capability
    routing: Capability
    calibration: Capability


class Metadata(StrictModel):
    """One immutable, hash-addressed Producer metadata revision."""

    contract_version: Literal[3]
    run_id: str
    producer_id: PositiveU64
    revision: PositiveU64
    process: Process
    clock_domain_id: PositiveU64
    producers: list[Producer]
    generations: list[Generation]
    clock_domains: list[ClockDomain]
    endpoints: list[Endpoint]
    graph_templates: list[GraphTemplate]
    artifacts: list[Artifact]
    capabilities: Capabilities
    # Configuration is owned by xpool.config, not by the offline reader.
    configuration: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        """Establish reference uniqueness and ownership before admitting events."""
        if str(UUID(self.run_id)) != self.run_id:
            raise ValueError("Run UUID must be canonical")

        producers = {item.id: item for item in self.producers}
        domains = {item.id: item for item in self.clock_domains}
        generations = {item.id: item for item in self.generations}

        for entries, keys in (
            (self.producers, [item.id for item in self.producers]),
            (self.clock_domains, [item.id for item in self.clock_domains]),
            (self.generations, [item.id for item in self.generations]),
            (self.endpoints, [(item.owner_producer, item.endpoint_sequence) for item in self.endpoints]),
            (self.graph_templates, [(item.owner_producer, item.template_sequence) for item in self.graph_templates]),
            (self.artifacts, [item.id for item in self.artifacts]),
        ):
            if len(entries) != len(set(keys)):
                raise ValueError("duplicate metadata identity")

        own = producers.get(self.producer_id)

        if own is None or own.process != self.process or own.clock_domain_id != self.clock_domain_id:
            raise ValueError("metadata owner mismatch")

        for producer in self.producers:
            domain = domains.get(producer.clock_domain_id)
            if domain is None or (producer.source == "host") != (domain.kind == "host_monotonic"):
                raise ValueError("Producer clock domain mismatch")
            if producer.source == "device" and domain.process_creation_id != producer.process_creation_id:
                raise ValueError("Device domain process mismatch")

        for endpoint in self.endpoints:
            owner = producers.get(endpoint.owner_producer)
            if owner is None or owner.source != "device" or owner.process.runtime_role != "atnagent":
                raise ValueError("Endpoint owner is not an AtnAgent Device Producer")
            if endpoint.generation not in generations:
                raise ValueError("Endpoint Generation unavailable")
            for site in endpoint.sites:
                if site.producer not in producers or producers[site.producer].process.runtime_role != site.site:
                    raise ValueError("Endpoint site mismatch")

        templates = {(item.owner_producer, item.template_sequence) for item in self.graph_templates}
        for template in self.graph_templates:
            if template.owner_producer not in producers:
                raise ValueError("Graph owner unavailable")
            if any((template.owner_producer, child) not in templates for child in template.compute_templates):
                raise ValueError("Compute template unavailable")

        return self


class EventRecord(StrictModel):
    """Complete canonical wire record; presence controls semantic availability."""

    event_sequence: PositiveU64
    raw_time_ns: U64
    clock_domain_id: PositiveU64
    kind: U32
    presence: U32
    generation: U64
    endpoint_owner: U64
    endpoint_sequence: U64
    operation_sequence: U64
    graph_owner: U64
    graph_template: U64
    graph_execution: U64
    instance_index: U64
    invocation_sequence: U64
    lane_index: U64
    lease_sequence: U64
    attachment_owner: U64
    attachment_sequence: U64
    local_trace_id: U64
    layer_ordinal: U64
    payload_rows: U64
    dp_rank_payload_rows: U64
    payload_row_capacity: U64
    ready_ticket: U64
    forward_mode: U32
    output_requirement: U32
    dp_row_layout: U32
    delivery: U32
    result_code: U32
    role: U32
    instance_rank: U32
    pe: Annotated[int, Field(ge=-(2**31), lt=2**31)]
    data0: U64
    data1: U64
    data2: U64
    data3: U64
    data4: U64

    def has(self, bit: PresenceBit) -> bool:
        """Return whether a named contract presence group was recorded."""
        return bool(self.presence & (1 << bit))


class Counters(StrictModel):
    attempted: U64
    committed: U64
    dropped_full: U64
    abandoned: U64
    stale_token: U64
    duplicate_commit: U64
    copied: U64
    exported: U64
    post_commit_lost: U64
    snapshot_operations_attempted: U64
    snapshot_operations_dropped: U64
    attachment_attempted: U64
    attachment_capacity_exhausted: U64
    attachment_payload_oversize: U64
    attachment_not_collected: U64
    attachment_publish_failed: U64
    sample_attempted: U64
    sample_failed: U64
    sample_skipped_busy: U64
    context_unavailable: U64


class QualityCheckpoint(StrictModel):
    """Cumulative source counters with explicit finality and unknown-tail state."""

    contract_version: Literal[3]
    run_id: str
    producer_id: PositiveU64
    revision: PositiveU64
    host_time_ns: U64
    collection_state: Literal["active", "stopped", "closed"]
    export_state: Literal["active", "failed", "closed"]
    stop_reason: (
        Literal["disk_quota", "io_error", "metadata_unavailable", "identity_exhausted", "shutdown_deadline"] | None
    )
    last_attempted_sequence: U64
    counters: Counters
    pending: U64 | None
    tail: Literal["complete", "pending", "unknown"]
    counter_exhausted: bool
    exact_totals: bool

    @model_validator(mode="after")
    def validate_conservation(self) -> Self:
        """Reject contradictory exact accounting while preserving lower bounds."""
        if str(UUID(self.run_id)) != self.run_id:
            raise ValueError("Run UUID must be canonical")

        c = self.counters

        if self.exact_totals:
            if self.counter_exhausted:
                raise ValueError("saturated counters cannot be exact")
            if self.pending is not None and c.attempted != c.committed + c.dropped_full + c.abandoned + self.pending:
                raise ValueError("attempt conservation violated")
            if not c.exported <= c.copied <= c.committed or c.exported + c.post_commit_lost > c.committed:
                raise ValueError("export conservation violated")

        return self


class RoutingAttachment(StrictModel):
    """Final Routing index entry keyed by publication identity rather than slot."""

    owner: PositiveU64
    sequence: PositiveU64
    status: AttachmentStatus
    path: str | None = None
    sha256: str | None = None
    tensor_keys: list[str] = Field(default_factory=list)
    generation: PositiveU64
    instance_index: U64
    invocation_sequence: PositiveU64
    lane_index: U64
    lease_sequence: PositiveU64
    layer_ordinal: U64
    publisher: PositiveU64
    pe: U32
    clock_domain_id: PositiveU64
    raw_publication_time_ns: U64
    shape: list[U64]
    dtypes: list[str]

    @model_validator(mode="after")
    def validate_publication(self) -> Self:
        """Require usable paths and hashes exclusively for published attachments."""
        if self.status == "published":
            if self.path is None or self.sha256 is None or len(self.sha256) != 64:
                raise ValueError("published attachment requires path and SHA-256")
        elif self.path is not None or self.sha256 is not None:
            raise ValueError("unpublished attachment cannot have a usable path/hash")
        return self


@dataclass(frozen=True, order=True)
class EventIdentity:
    """One attempted event isolated by Run and registered Producer."""

    run_id: str
    producer: int
    event_sequence: int


@dataclass(frozen=True)
class Source:
    """The physical and protocol provenance of an observed event."""

    identity: EventIdentity
    path: Path
    offset: int
    process_creation_id: str
    clock_domain_id: int
    endpoint: tuple[int, int] | None
    instance_rank: int | None
    pe: int | None


@dataclass(frozen=True)
class Event:
    """Canonical record paired with its exact metadata and physical source."""

    record: EventRecord
    source: Source
    metadata: Metadata


@dataclass(frozen=True)
class Finding:
    """A local evidence limitation with optional event or Producer provenance."""

    code: str
    detail: str
    sources: tuple[Source, ...] = ()
    run_id: str | None = None
    producer: int | None = None


@dataclass(frozen=True)
class CalibrationSample:
    """A raw paired attempt for clock calibration; failed Device reads remain absent."""

    source: Source
    attempt_sequence: int
    device_clock_domain_id: int
    raw_epoch: int
    host_before_ns: int
    device_time_ns: int | None
    host_after_ns: int | None
    outcome: Literal["succeeded", "failed"]
    failure_code: Literal["enqueue_failed", "completion_failed", "timed_out", "shutdown"] | None
