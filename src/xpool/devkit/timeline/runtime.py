"""Process-local independent Collector/Writer scheduling and bounded retirement."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from uuid import uuid4

import torch

import xpool.native
from xpool.config import get_global_config
from xpool.devkit.timeline.models import Grant, GrantRequest, ProcessRole, Producer, ProducerRequest, Reason
from xpool.devkit.timeline.session import Session
from xpool.devkit.timeline.writer import Writer
from xpool.service.client import XpoolClient
from xpool.utils.procs import ProcUniqId

__all__ = ["Runtime", "start"]

logger = logging.getLogger(__name__)


class Pipeline:
    """One fixed-size receipt queue and two independently scheduled workers."""

    def __init__(
        self,
        producer: Producer,
        get_recorder: Callable[[], xpool.native.devkit.timeline.Recorder | None],
        grant: Callable[[GrantRequest], Grant],
    ) -> None:
        config = get_global_config().debug.timeline
        self.producer = producer
        self.get_recorder = get_recorder
        self.writer = Writer(producer, config.chunk_bytes, grant)
        self.interval = config.flush_interval_ms / 1000
        self.queue: queue.Queue[xpool.native.devkit.timeline.Chunk] = queue.Queue(maxsize=256)
        self.finish = threading.Event()
        self.collector_done = threading.Event()
        self.writer_done = threading.Event()
        self.stop_collecting = threading.Event()
        self.collector = threading.Thread(target=self.collect, name="xpool-timeline-collector", daemon=True)
        self.writer_thread = threading.Thread(target=self.write, name="xpool-timeline-writer", daemon=True)
        self.recorder: xpool.native.devkit.timeline.Recorder | None = None
        self.faulted = False
        self.collector.start()
        self.writer_thread.start()

    def collect(self) -> None:
        """Drive asynchronous receipt without touching filesystem or grant RPCs."""
        next_seal = 0.0
        try:
            while not self.stop_collecting.is_set():
                recorder = self.get_recorder()
                self.recorder = recorder
                if recorder is None:
                    if self.finish.is_set():
                        break
                    self.stop_collecting.wait(self.interval)
                    continue
                if self.finish.is_set() or self.writer.failed or Reason.DISK_LIMIT in self.writer.quality.reasons:
                    recorder.stop()
                if not self.queue.full():
                    seal = self.finish.is_set() or time.monotonic() >= next_seal
                    if seal:
                        next_seal = time.monotonic() + self.interval
                    chunk = recorder.collect(seal)
                    if chunk is not None:
                        self.queue.put_nowait(chunk)
                        del chunk
                if self.finish.is_set() and recorder.drained():
                    break
                self.stop_collecting.wait(min(self.interval, 0.001) if self.finish.is_set() else self.interval / 8)
        except Exception:
            self.faulted = True
            self.writer.quality.reasons.add(Reason.PRODUCER_FAULT)
            logger.exception("timeline collector failed producer=%s", self.producer.producer_id)
            if self.recorder is not None:
                self.recorder.stop()
        finally:
            self.collector_done.set()

    def write(self) -> None:
        """Publish leases with bounded credit, independently of CUDA collection."""
        last_checkpoint = 0.0
        try:
            while True:
                if not self.collector_done.is_set() or not self.queue.empty():
                    self.writer.replenish()
                try:
                    chunk = self.queue.get(timeout=self.interval)
                except queue.Empty:
                    if self.collector_done.is_set():
                        break
                else:
                    while (
                        self.writer.credit < self.writer.chunk_bytes
                        and not self.writer.failed
                        and Reason.DISK_LIMIT not in self.writer.quality.reasons
                        and not self.stop_collecting.is_set()
                    ):
                        if self.writer.replenish():
                            break
                        self.stop_collecting.wait(self.interval)
                    self.writer.publish(chunk)
                    del chunk  # Last lease view retired; receipt storage is reusable.
                recorder = self.recorder
                if recorder is not None and time.monotonic() - last_checkpoint >= 1:
                    self.writer.checkpoint(recorder.counters(), False)
                    last_checkpoint = time.monotonic()
            recorder = self.recorder
            if recorder is not None:
                closed = (
                    recorder.drained() and not self.faulted and Reason.FLUSH_TIMEOUT not in self.writer.quality.reasons
                )
                self.writer.checkpoint(recorder.counters(), closed)
            else:
                self.writer.quality.reasons.add(Reason.MISSING_PRODUCER)
        except Exception as error:
            self.faulted = True
            self.writer.failed = True
            self.writer.quality.reasons.add(Reason.PRODUCER_FAULT)
            logger.error("timeline writer failed producer=%s detail=%s", self.producer.producer_id, str(error))
        finally:
            self.writer_done.set()

    def close(self, deadline: float, production_quiesced: bool) -> None:
        """Bound waiting; retained threads/leases keep timed-out resources alive."""
        self.finish.set()
        recorder = self.recorder
        if recorder is not None:
            recorder.stop()
        self.collector.join(max(0.0, deadline - time.monotonic()))
        self.writer_thread.join(max(0.0, deadline - time.monotonic()))
        if self.collector.is_alive() or self.writer_thread.is_alive():
            self.writer.quality.reasons.add(Reason.FLUSH_TIMEOUT)
            self.stop_collecting.set()
            logger.warning("timeline flush timeout producer=%s", self.producer.producer_id)
            return
        if recorder is not None and production_quiesced and recorder.drained():
            recorder.close()


class Runtime:
    """Retain all Producers of one process and a separate background RPC client."""

    def __init__(self, role: ProcessRole, slot: str, device: int, session: Session | None = None) -> None:
        config = get_global_config().debug.timeline
        self.client = XpoolClient() if session is None else None
        if session is not None:
            register = session.register
            grant = session.grant
        elif self.client is not None:
            register = self.client.timeline_register
            grant = self.client.timeline_grant
        else:
            raise RuntimeError("timeline registration owner is missing")
        self.closed = False
        self.timeout = config.shutdown_flush_timeout_s
        self.pipelines: list[Pipeline] = []
        try:
            identity = ProcUniqId.current()
            startup = uuid4()
            request = ProducerRequest(
                startup_id=startup,
                pid=identity.pid,
                create_time=identity.create_time,
                role=role,
                slot=slot,
                source="host",
            )
            host = register(request)
            device_producer: Producer | None = None
            if device >= 0:
                device_producer = register(
                    request.model_copy(
                        update={
                            "source": "device",
                            "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
                        }
                    )
                )
            xpool.native.devkit.timeline.configure(
                host.producer_id, 0 if device_producer is None else device_producer.producer_id, device
            )
            self.pipelines = [Pipeline(host, xpool.native.devkit.timeline.host_recorder, grant)]
            if device_producer is not None:
                self.pipelines.append(Pipeline(device_producer, xpool.native.devkit.timeline.device_recorder, grant))
            recorder = xpool.native.devkit.timeline.host_recorder()
            if recorder is not None:
                recorder.record(
                    xpool.native.devkit.timeline.Record(
                        timestamp=time.monotonic_ns(), kind=1, site=threading.get_native_id()
                    )
                )
        except Exception:
            end = time.monotonic() + self.timeout
            for pipeline in self.pipelines:
                pipeline.finish.set()
            for pipeline in self.pipelines:
                pipeline.close(end, production_quiesced=True)
            if self.client is not None and all(p.writer_done.is_set() for p in self.pipelines):
                self.client.close()
            raise

    def mark_serving(self) -> None:
        """Record the daemon's confirmed serving-health edge in the Host clock."""
        recorder = xpool.native.devkit.timeline.host_recorder()
        if recorder is not None:
            recorder.record(
                xpool.native.devkit.timeline.Record(
                    timestamp=time.monotonic_ns(), kind=7, site=threading.get_native_id()
                )
            )

    def close(self, production_quiesced: bool, deadline: float | None = None) -> None:
        """Use the smaller of the Timeline and retained production deadlines."""
        if self.closed:
            return
        self.closed = True
        recorder = xpool.native.devkit.timeline.host_recorder()
        if recorder is not None:
            recorder.record(
                xpool.native.devkit.timeline.Record(
                    timestamp=time.monotonic_ns(), kind=2, site=threading.get_native_id()
                )
            )
        end = time.monotonic() + self.timeout
        if deadline is not None:
            end = min(end, deadline)
        # Close admission on every Producer before waiting for any one Writer.
        for pipeline in self.pipelines:
            pipeline.finish.set()
        for pipeline in self.pipelines:
            pipeline.close(end, production_quiesced)
        if self.client is not None and all(p.writer_done.is_set() for p in self.pipelines):
            self.client.close()


def start(role: ProcessRole, slot: str, device: int = -1, session: Session | None = None) -> Runtime | None:
    """Create Producers only when independent Timeline collection is enabled."""
    if not get_global_config().debug.timeline.enable:
        return None
    return Runtime(role, slot, device, session)
