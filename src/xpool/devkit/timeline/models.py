"""Versioned Timeline metadata shared by runtime writers and offline readers."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Grant",
    "GrantRequest",
    "ProcessRole",
    "Producer",
    "ProducerRequest",
    "Quality",
    "Reason",
    "SessionManifest",
]


type ProcessRole = Literal["daemon", "instance", "atnagent", "ffnagent"]


class Reason(StrEnum):
    """Independent collection quality reasons"""

    BUFFER_FULL = "buffer_full"
    RESERVATION_CONTENTION = "reservation_contention"
    DISK_LIMIT = "disk_limit"
    PRODUCER_FAULT = "producer_fault"
    TRACE_IO = "trace_io"
    FLUSH_TIMEOUT = "flush_timeout"
    ABNORMAL_EXIT = "abnormal_exit"
    RECORD_LOSS = "record_loss"
    UNPAIRED_BOUNDARY = "unpaired_boundary"
    MISSING_PRODUCER = "missing_producer"
    UNKNOWN_TAIL = "unknown_tail"
    IDENTITY_EXHAUSTED = "identity_exhausted"


class Metadata(BaseModel):
    """Strict external metadata boundary."""

    model_config = ConfigDict(extra="forbid")


class ProducerRequest(Metadata):
    """One startup registration; repeats preserve its Producer identity."""

    startup_id: UUID
    pid: int = Field(gt=0)
    create_time: float = Field(gt=0, allow_inf_nan=False)
    role: ProcessRole
    slot: str = Field(min_length=1, max_length=128)
    source: Literal["host", "device"]
    device_uuid: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_source(self) -> ProducerRequest:
        """Device Producers require a confirmed physical identity."""
        if (self.source == "device") != (self.device_uuid is not None):
            raise ValueError("device source and device_uuid must be paired")
        if self.role == "daemon" and self.source != "host":
            raise ValueError("daemon cannot initialize a Device Producer")
        return self


class Producer(ProducerRequest):
    """Daemon-authored Producer identity and exclusive publication directory."""

    producer_id: int = Field(gt=0, le=2**63 - 1)
    session_id: UUID
    directory: Path
    machine_id: str
    clock_domain: str


class GrantRequest(Metadata):
    """Sequential, retryable request for bounded file credit."""

    producer_id: int = Field(gt=0)
    startup_id: UUID
    sequence: int = Field(gt=0, le=2**63 - 1)
    bytes_requested: int = Field(gt=0, le=4 << 20)


class Grant(Metadata):
    """Reserved bytes; rename consumes no additional credit."""

    sequence: int
    bytes_granted: int = Field(ge=0)


class Quality(Metadata):
    """Reserved current state; counters do not recursively emit events."""

    attempted: int = Field(default=0, ge=0)
    committed: int = Field(default=0, ge=0)
    dropped: int = Field(default=0, ge=0)
    exported: int = Field(default=0, ge=0)
    lost_after_commit: int = Field(default=0, ge=0)
    contention: int = Field(default=0, ge=0)
    first_gap: int | None = None
    last_gap: int | None = None
    gap_coarsened: bool = True
    reasons: set[Reason] = Field(default_factory=set)
    window_clock: Literal["CLOCK_MONOTONIC"] = "CLOCK_MONOTONIC"
    window_begin_ns: int = Field(default=0, description="Host admission window start in CLOCK_MONOTONIC nanoseconds.")
    window_end_ns: int | None = Field(default=None, description="Host flush completion observation in the same clock.")
    closed: bool = False
    unknown_tail: bool = True


class SessionManifest(Metadata):
    """Session inventory with actual effective budgets and supported coverage."""

    schema_version: Literal[1] = 1
    session_id: UUID
    machine_id: str
    config: dict[str, int | bool | str | None]
    expected_slots: list[str]
    producers: list[Producer] = Field(default_factory=list)
    granted_bytes: int = 0
    capabilities: list[str] = Field(
        default_factory=lambda: [
            "host.lifecycle",
            "transport.shared_identity",
            "transport.local_boundaries",
            "fabric.lane_lease",
            "fabric.compute_protocol_bracket",
        ]
    )
    clock_alignment: Literal["unavailable"] = "unavailable"
    closed: bool = False
