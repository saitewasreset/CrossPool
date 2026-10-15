"""Strict offline observation artifacts, identity-based reconstruction and legacy views."""

from xpool.observability.artifacts import (
    ArtifactInputs,
    ArtifactSet,
    AttachmentEvidence,
    AttachmentIndex,
    read_artifacts,
)
from xpool.observability.codec import Chunk, ChunkHeader, encode_chunk, parse_chunk
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import (
    CalibrationSample,
    Event,
    EventIdentity,
    EventRecord,
    Finding,
    Metadata,
    QualityCheckpoint,
    RoutingAttachment,
    Source,
)
from xpool.observability.reconstruction import (
    CausalEdge,
    Entity,
    EntityIdentity,
    Interval,
    Observation,
    QualityReport,
    Reconstruction,
    reconstruct,
)
from xpool.observability.snapshots import LegacyProjection, SnapshotRequest, project_legacy_snapshots

__all__ = [
    "ArtifactInputs",
    "ArtifactSet",
    "AttachmentEvidence",
    "AttachmentIndex",
    "CalibrationSample",
    "CausalEdge",
    "Chunk",
    "ChunkHeader",
    "Entity",
    "EntityIdentity",
    "ErrorCode",
    "Event",
    "EventIdentity",
    "EventRecord",
    "Finding",
    "Interval",
    "LegacyProjection",
    "Metadata",
    "Observation",
    "ObservationError",
    "QualityCheckpoint",
    "QualityReport",
    "Reconstruction",
    "RoutingAttachment",
    "SnapshotRequest",
    "Source",
    "encode_chunk",
    "parse_chunk",
    "project_legacy_snapshots",
    "read_artifacts",
    "reconstruct",
]
