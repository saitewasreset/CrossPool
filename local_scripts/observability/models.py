"""Archive contracts independent of runtime configuration installation."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

type Nonnegative = Annotated[int, Field(ge=0, le=2**63 - 1)]


class ArchiveModel(BaseModel):
    """Reject malformed metadata rather than repairing evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Participant(ArchiveModel):
    """A confirmed PE placement with the retained evidence source."""

    pe: Nonnegative
    device_uuid: str | None = None
    pid: Nonnegative | None = None
    process_created_at: str | None = None
    evidence: str | None = None

    @model_validator(mode="after")
    def require_evidence(self) -> Self:
        if self.device_uuid is not None and (not self.device_uuid.startswith("GPU-") or not self.evidence):
            raise ValueError("confirmed device UUID requires an evidence reference")
        return self


class RunContext(ArchiveModel):
    """Execution evidence supplied before archiving one isolated Attempt."""

    run_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    outcome: Literal["passed", "failed", "unknown"] = "unknown"
    command: list[str] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    time_scope: Literal["harness", "unknown"] = "unknown"
    returncode: int | None = None
    case: str = "serving-001"
    model_id: str = "Qwen/Qwen3-0.6B"
    graph_mode: str = "decode-full-prefill-breakable"
    record_capacity: Nonnegative = 32768
    participants: list[Participant] = Field(default_factory=list)
    software: dict[str, str | None] = Field(default_factory=dict)
    source: dict[str, JsonValue] = Field(default_factory=dict)
    metadata_issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_participants(self) -> Self:
        if len({participant.pe for participant in self.participants}) != len(self.participants):
            raise ValueError("duplicate PE metadata")
        return self


class SourceFile(ArchiveModel):
    """One byte-preserved input, relative to the archive directory."""

    path: str
    kind: Literal["fabric", "transport", "graph", "graph_events", "evidence"]
    size: Nonnegative
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def safe_path(self) -> Self:
        value = Path(self.path)
        if value.is_absolute() or ".." in value.parts or not value.parts or "\\" in self.path:
            raise ValueError("archive file path must be relative and contained")
        return self


class Manifest(ArchiveModel):
    """Complete archive inventory; capture time is deliberately unknown."""

    format_version: Literal[1] = 1
    context: RunContext
    archive_started_at: str
    archive_finished_at: str
    snapshot_capture_time: None = None
    files: list[SourceFile]
    missing: list[str]

    @model_validator(mode="after")
    def unique_files(self) -> Self:
        if len({source.path for source in self.files}) != len(self.files):
            raise ValueError("duplicate archive file path")
        return self


class QualityIssue(ArchiveModel):
    """A diagnostic whose scope preserves the uncertainty of its source."""

    kind: str
    detail: str
    source: str | None = None
    invocation: str | None = None


class QualityReport(ArchiveModel):
    """Conversion and representative-evidence verdicts remain separate."""

    run_id: str
    attempt: int
    incomplete: bool
    issues: list[QualityIssue]
    prefill_invocation: str | None
    decode_invocation: str | None
    lane_reused: bool
    representative: bool
    clock_calibrated: Literal[False] = False
