"""Version-one little-endian raw Chunk serialization and verification."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from xpool.devkit.timeline.models import Producer

__all__ = ["HEADER_BYTES", "RECORD_BYTES", "Header", "Record", "read_chunk", "write_chunk"]

HEADER_BYTES = 256
RECORD_BYTES = 128
HEADER = struct.Struct("<8sII16s8Q32s")
RECORD = struct.Struct("<16Q")
MAGIC = b"XPTLINE1"


@dataclass(frozen=True, slots=True)
class Record:
    """Version-one record fields; zero kind is an explicitly aborted reservation."""

    sequence: int
    timestamp: int
    kind: int
    site: int
    generation_high: int
    generation_low: int
    endpoint_creator: int
    endpoint_index: int
    operation: int
    instance: int
    layer: int
    lane: int
    lease: int
    result: int
    rows: int
    reserved: int

    @classmethod
    def iter_payload(cls, payload: memoryview | bytes) -> Iterator[Record]:
        """Decode bounded version-one payloads at the owning wire boundary."""
        for fields in RECORD.iter_unpack(payload):
            yield cls(*fields)


@dataclass(frozen=True, slots=True)
class Header:
    """Published immutable inventory including source epoch and receipt window."""

    session_id: UUID
    producer_id: int
    sequence: int
    epoch: int
    count: int
    first_record: int
    last_record: int
    begin_ns: int
    end_ns: int
    checksum: bytes

    def encode(self) -> bytes:
        """Encode all fields and deterministic zero padding."""
        return HEADER.pack(
            MAGIC,
            1,
            RECORD_BYTES,
            self.session_id.bytes,
            self.producer_id,
            self.sequence,
            self.epoch,
            self.count,
            self.first_record,
            self.last_record,
            self.begin_ns,
            self.end_ns,
            self.checksum,
        ).ljust(HEADER_BYTES, b"\0")


def write_chunk(
    producer: Producer,
    sequence: int,
    epoch: int,
    begin_ns: int,
    end_ns: int,
    payload: memoryview,
) -> tuple[Path, int]:
    """Publish a unique file; caller already holds credit for its full maximum size.

    Keep failed partials as evidence. The memoryview remains leased throughout
    writing and is never copied into an unbounded Python collection.
    """
    if not payload or len(payload) % RECORD_BYTES:
        raise ValueError("timeline raw payload has invalid record length")
    sequences = (record.sequence for record in Record.iter_payload(payload))
    minimum = 2**64 - 1
    maximum = 0
    for value in sequences:
        minimum, maximum = min(minimum, value), max(maximum, value)
    header = Header(
        producer.session_id,
        producer.producer_id,
        sequence,
        epoch,
        len(payload) // RECORD_BYTES,
        minimum,
        maximum,
        begin_ns,
        end_ns,
        hashlib.sha256(payload).digest(),
    )
    path = producer.directory / f"chunk-{sequence:020d}.bin"
    partial = path.with_suffix(".partial")
    if path.exists():
        raise FileExistsError(f"timeline published chunk already exists: {path}")
    try:
        with partial.open("xb") as stream:
            stream.write(header.encode())
            stream.write(payload)
        # The sole Writer owns this Producer namespace and monotonic name.
        # Same-directory rename publishes only the fully closed contents.
        partial.rename(path)
    except OSError as error:
        raise OSError(error.errno, f"timeline chunk publication failed: {path}") from error
    return path, HEADER_BYTES + len(payload)


def read_chunk(path: Path, producer: Producer, chunk_bytes: int) -> tuple[Header, list[Record]]:
    """Validate one bounded Chunk before exposing any records."""
    size = path.stat().st_size
    if not HEADER_BYTES < size <= chunk_bytes:
        raise ValueError(f"timeline chunk length invalid: {path}")
    with path.open("rb") as stream:
        raw_header = stream.read(HEADER_BYTES)
        values = HEADER.unpack_from(raw_header)
        magic, version, record_bytes, session_id, *tail = values
        if magic != MAGIC or version != 1 or record_bytes != RECORD_BYTES or any(raw_header[HEADER.size :]):
            raise ValueError(f"timeline schema/header invalid: {path}")
        header = Header(UUID(bytes=session_id), *tail)
        if header.session_id != producer.session_id or header.producer_id != producer.producer_id:
            raise ValueError(f"timeline chunk identity conflict: {path}")
        if header.sequence <= 0 or header.epoch <= 0 or header.count * RECORD_BYTES + HEADER_BYTES != size:
            raise ValueError(f"timeline chunk count/epoch invalid: {path}")
        if header.end_ns < header.begin_ns:
            raise ValueError(f"timeline collection window invalid: {path}")
        payload = stream.read(chunk_bytes)
    if hashlib.sha256(payload).digest() != header.checksum:
        raise ValueError(f"timeline checksum mismatch: {path}")
    records = list(Record.iter_payload(payload))
    if (
        not records
        or min(r.sequence for r in records) != header.first_record
        or max(r.sequence for r in records) != header.last_record
    ):
        raise ValueError(f"timeline record range invalid: {path}")
    if any(r.sequence <= 0 or r.reserved != 0 for r in records):
        raise ValueError(f"timeline record identity/schema invalid: {path}")
    return header, records
