"""Structured failures at observation artifact boundaries."""

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpool.observability.models import Source

__all__ = ["ErrorCode", "ObservationError"]


class ErrorCode(StrEnum):
    """Stable artifact error categories."""

    HEADER_TRUNCATED = "header_truncated"
    BAD_MAGIC = "bad_magic"
    UNSUPPORTED_VERSION = "unsupported_version"
    INVALID_LAYOUT = "invalid_layout"
    HEADER_CHECKSUM = "header_checksum"
    INVALID_LENGTH = "invalid_length"
    METADATA_UNAVAILABLE = "metadata_unavailable"
    METADATA_CHECKSUM = "metadata_checksum"
    PAYLOAD_CHECKSUM = "payload_checksum"
    INVALID_EVENT = "invalid_event"
    UNKNOWN_KIND = "unknown_kind"
    IDENTITY_CONFLICT = "identity_conflict"
    EVENT_CONFLICT = "event_conflict"


class ObservationError(ValueError):
    """An artifact failure with machine-readable category and source context."""

    def __init__(
        self,
        code: ErrorCode,
        detail: str,
        *,
        path: Path | None = None,
        offset: int | None = None,
        producer: int | None = None,
        event_sequence: int | None = None,
        sources: tuple["Source", ...] = (),
    ) -> None:
        self.code = code
        self.detail = detail
        self.path = path
        self.offset = offset
        self.producer = producer
        self.event_sequence = event_sequence
        self.sources = sources
        super().__init__(f"{code.value}: {detail}" + (f" ({path}:{offset})" if path else ""))
