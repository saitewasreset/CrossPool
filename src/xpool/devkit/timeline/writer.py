"""Bounded local Writer; grants and files are absent from inference hooks."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import xpool.native
from xpool.devkit.timeline.format import HEADER_BYTES, Record, write_chunk
from xpool.devkit.timeline.models import Grant, GrantRequest, Producer, Quality, Reason
from xpool.devkit.timeline.session import PRODUCER_METADATA_LIMIT, publish_metadata
from xpool.service.errors import XpoolClientError, XpoolDaemonError

__all__ = ["Writer"]

logger = logging.getLogger(__name__)


class Writer:
    """One Producer's credit, publication and reserved quality metadata."""

    def __init__(self, producer: Producer, chunk_bytes: int, grant: Callable[[GrantRequest], Grant]) -> None:
        self.producer = producer
        self.chunk_bytes = chunk_bytes
        self.grant = grant
        self.credit = 0
        self.request_sequence = 1
        self.pending: GrantRequest | None = None
        self.quality = Quality(window_begin_ns=time.monotonic_ns())
        self.failed = False
        self.last_grant_warning = False
        self.quality_failure_reported = False

    def replenish(self) -> bool:
        """Retry the same outstanding request; no unbounded queue or credit guessing."""
        if self.credit >= self.chunk_bytes:
            return True
        if Reason.DISK_LIMIT in self.quality.reasons or self.failed:
            return False
        if self.pending is None:
            self.pending = GrantRequest(
                producer_id=self.producer.producer_id,
                startup_id=self.producer.startup_id,
                sequence=self.request_sequence,
                bytes_requested=4 * self.chunk_bytes,
            )
        try:
            grant = self.grant(self.pending)
        except (XpoolClientError, XpoolDaemonError) as error:
            if not error.is_recoverable:
                self.failed = True
                self.quality.reasons.add(Reason.PRODUCER_FAULT)
            if not self.last_grant_warning:
                logger.warning(
                    "timeline grant unavailable producer=%s detail=%s", self.producer.producer_id, str(error)
                )
                self.last_grant_warning = True
            return False
        if grant.sequence != self.pending.sequence or grant.bytes_granted > self.pending.bytes_requested:
            raise ValueError("timeline grant reply does not match pending request")
        self.credit += grant.bytes_granted
        self.request_sequence += 1
        self.pending = None
        self.last_grant_warning = False
        if grant.bytes_granted == 0:
            self.quality.reasons.add(Reason.DISK_LIMIT)
        return self.credit >= self.chunk_bytes

    def publish(self, chunk: xpool.native.devkit.timeline.Chunk) -> None:
        """Consume charged credit before creation, then release receipt ownership."""
        try:
            with memoryview(chunk) as payload:
                count = sum(record.kind != 0 for record in Record.iter_payload(payload))
                if self.failed or self.credit < self.chunk_bytes:
                    self.quality.lost_after_commit += count
                    return
                self.credit -= HEADER_BYTES + len(payload)
                try:
                    write_chunk(self.producer, chunk.sequence, chunk.epoch, chunk.begin_ns, chunk.end_ns, payload)
                except OSError as error:
                    self.quality.lost_after_commit += count
                    self.quality.reasons.add(Reason.TRACE_IO)
                    self.failed = True
                    logger.error(
                        "timeline export failed producer=%s errno=%s detail=%s",
                        self.producer.producer_id,
                        error.errno,
                        str(error),
                    )
                    return
                self.quality.exported += count
        finally:
            chunk.release()

    def checkpoint(self, counters: xpool.native.devkit.timeline.Counters, closed: bool) -> None:
        """Persist bounded counter state; disk faults fall back to existing logs."""
        self.quality.attempted = counters.attempted
        self.quality.committed = counters.committed
        self.quality.dropped = counters.dropped
        self.quality.contention = counters.contention
        if counters.dropped:
            self.quality.first_gap = 1
            self.quality.last_gap = counters.attempted
            self.quality.reasons.add(Reason.RECORD_LOSS)
        if counters.dropped > counters.contention:
            self.quality.reasons.add(Reason.BUFFER_FULL)
        if counters.contention:
            self.quality.reasons.add(Reason.RESERVATION_CONTENTION)
        if counters.exhausted:
            self.quality.reasons.add(Reason.IDENTITY_EXHAUSTED)
        if self.quality.lost_after_commit:
            self.quality.reasons.add(Reason.RECORD_LOSS)
        self.quality.closed = closed
        self.quality.unknown_tail = not closed
        if closed:
            self.quality.window_end_ns = time.monotonic_ns()
        if self.failed:
            if not self.quality_failure_reported:
                logger.error(
                    "timeline quality unavailable producer=%s reasons=%s",
                    self.producer.producer_id,
                    sorted(self.quality.reasons),
                )
                self.quality_failure_reported = True
            return
        try:
            publish_metadata(
                self.producer.directory / "summary.json",
                self.quality.model_dump_json().encode(),
                PRODUCER_METADATA_LIMIT,
            )
        except OSError as error:
            self.failed = True
            self.quality.reasons.add(Reason.TRACE_IO)
            logger.error("timeline summary failed producer=%s detail=%s", self.producer.producer_id, str(error))
