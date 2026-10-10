"""Daemon-only Session inventory, bounded metadata and idempotent disk grants."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from uuid import uuid4

from xpool.devkit.timeline.models import Grant, GrantRequest, Producer, ProducerRequest, SessionManifest

__all__ = ["Session", "publish_metadata"]

MANIFEST_LIMIT = 512 << 10
PRODUCER_METADATA_LIMIT = 16 << 10


def publish_metadata(path: Path, payload: bytes, limit: int) -> None:
    """Replace bounded current state, charging coexistence with its temporary file.

    The caller reserves twice ``limit``. A failed partial is retained and stops
    subsequent updates rather than accumulating unbudgeted retry files.
    """
    if len(payload) > limit:
        raise ValueError(f"timeline metadata_limit target={path} bytes={len(payload)} limit={limit}")
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
        os.replace(partial, path)
    except OSError as error:
        raise OSError(error.errno, f"timeline metadata publication failed: {path}") from error


class Session:
    """Serialized Session coordinator; no raw records cross this boundary."""

    def __init__(self, outdir: Path, config: dict[str, int | bool | str | None], expected_slots: list[str]) -> None:
        """Create exclusive directories only after validating metadata capacity."""
        self.lock = threading.RLock()
        self.session_id = uuid4()
        self.directory = outdir / str(self.session_id)
        self.chunk_bytes = int(config["chunk_bytes"] or 0)
        self.maximum = int(config["session_max_bytes"] or 0)
        self.reserve = int(config["metadata_reserve_bytes"] or 0)
        minimum = 2 * MANIFEST_LIMIT + len(expected_slots) * 2 * PRODUCER_METADATA_LIMIT
        host_budget = int(config["host_buffer_bytes"] or 0)
        host_metadata = max(65536, min(4 << 20, host_budget // 4))
        if minimum > host_metadata:
            raise ValueError(f"timeline expected Producers require {minimum} Host metadata bytes")
        if minimum > self.reserve:
            raise ValueError(f"timeline expected Producers require {minimum} metadata bytes")
        machine = Path("/etc/machine-id").read_bytes() + Path("/proc/sys/kernel/random/boot_id").read_bytes()
        machine_id = hashlib.sha256(machine).hexdigest()
        self.manifest = SessionManifest(
            session_id=self.session_id,
            machine_id=machine_id,
            config=config,
            expected_slots=expected_slots,
        )
        self.requests: dict[int, Grant] = {}
        self.request_sizes: dict[int, int] = {}
        self.failed = False
        self.directory.mkdir(parents=True, exist_ok=False)
        self.persist()

    def persist(self) -> None:
        """Atomically publish the bounded authoritative current inventory."""
        try:
            publish_metadata(
                self.directory / "manifest.json",
                self.manifest.model_copy(
                    update={
                        "producers": [
                            producer.model_copy(update={"directory": Path(producer.directory.name)})
                            for producer in self.manifest.producers
                        ]
                    }
                )
                .model_dump_json()
                .encode(),
                MANIFEST_LIMIT,
            )
        except (OSError, ValueError):
            self.failed = True
            raise

    def register(self, request: ProducerRequest) -> Producer:
        """Register an expected slot once, retaining creation identity across retries."""
        with self.lock:
            if self.failed or self.manifest.closed:
                raise ValueError("timeline Session is closed")
            key = f"{request.slot}:{request.source}"
            if key not in self.manifest.expected_slots:
                raise ValueError(f"timeline unexpected Producer slot={key}")
            for producer in self.manifest.producers:
                if producer.slot == request.slot and producer.source == request.source:
                    if (
                        ProducerRequest.model_validate(producer.model_dump(include=set(ProducerRequest.model_fields)))
                        != request
                    ):
                        raise ValueError(f"timeline Producer slot already has a creation identity: {key}")
                    return producer
            producer_id = len(self.manifest.producers) + 1
            directory = self.directory / f"producer-{producer_id}"
            producer = Producer(
                **request.model_dump(),
                producer_id=producer_id,
                session_id=self.session_id,
                directory=directory,
                machine_id=self.manifest.machine_id,
                clock_domain=f"{self.manifest.machine_id}:{request.device_uuid or 'CLOCK_MONOTONIC'}",
            )
            # The registration remains retained if publication fails: retries
            # cannot allocate a second identity or create uncharged namespaces.
            directory.mkdir(exist_ok=False)
            self.manifest.producers.append(producer)
            self.persist()
            return producer

    def grant(self, request: GrantRequest) -> Grant:
        """Reserve credit before use; retain the last reply for identical retries."""
        with self.lock:
            producer = next((p for p in self.manifest.producers if p.producer_id == request.producer_id), None)
            if producer is None or producer.startup_id != request.startup_id:
                raise ValueError("timeline grant Producer creation identity does not match")
            previous = self.requests.get(request.producer_id)
            if self.failed:
                raise ValueError("timeline Session metadata publication failed")
            if previous is not None and request.sequence == previous.sequence:
                if request.bytes_requested != self.request_sizes[request.producer_id]:
                    raise ValueError("timeline grant retry changed requested bytes")
                return previous
            expected = 1 if previous is None else previous.sequence + 1
            if request.sequence != expected or self.manifest.closed:
                raise ValueError("timeline grant request sequence is stale, skipped or Session closed")
            remaining = self.maximum - self.reserve - self.manifest.granted_bytes
            amount = min(request.bytes_requested, remaining)
            # Whole maximum-sized chunks permit the Writer to bound local credit.
            amount -= amount % self.chunk_bytes
            reply = Grant(sequence=request.sequence, bytes_granted=amount)
            self.requests[request.producer_id] = reply
            self.request_sizes[request.producer_id] = request.bytes_requested
            self.manifest.granted_bytes += amount
            self.persist()
            return reply

    def close(self) -> None:
        """Seal Session metadata; missing Producer summaries remain unknown tails."""
        with self.lock:
            if not self.manifest.closed:
                self.manifest.closed = True
                self.persist()
