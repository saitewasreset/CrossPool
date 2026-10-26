"""Deterministic capacity estimates; no CUDA initialization or hardware claims."""

from __future__ import annotations

import heapq
import math
from collections import deque
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from xpool.devkit.capacity_counter import Histogram

MIB = 1024 * 1024
EVENT_BYTES = 256


class Evidence(StrEnum):
    """Strength of a result, never inferred from an experiment label."""

    ESTIMATED = "estimated"
    INCOMPLETE = "incomplete"


class Service(BaseModel):
    """Observed service assumptions, including measured scheduling gaps."""

    model_config = ConfigDict(extra="forbid")
    copy_bytes_s: FiniteFloat = Field(gt=0)
    writer_bytes_s: FiniteFloat = Field(gt=0)
    copy_delay_s: FiniteFloat = Field(default=0, ge=0)
    writer_stall_s: FiniteFloat = Field(default=0, ge=0)
    host_other_bytes: int = Field(default=0, ge=0)
    disk_other_bytes: int = Field(default=0, ge=0)


class Capacity(BaseModel):
    """Demand from a conservative arrival replay, not a zero-loss guarantee."""

    interval_ms: int
    device_slots: int
    host_bytes: int
    doubled_device_slots: int
    doubled_host_bytes: int
    device_logical_bytes: int
    doubled_device_logical_bytes: int
    exceeds_default_event_slots: bool
    exceeds_default_device_logical_budget: bool
    exceeds_default_host_payload_budget: bool
    sustained_overload: bool


class Estimate(BaseModel):
    """Per-domain report; Graph target estimates remain explicitly separate."""

    evidence: Evidence
    duration_s: float
    events: int
    routing_payload_bytes: int
    event_bytes_s: float
    routing_payload_bytes_s: float
    peak_event_bytes_s: dict[str, float]
    peak_routing_payload_bytes_s: dict[str, float]
    graph_activity_estimate_events: int
    host_forward_boundary_events: int
    host_capture_boundary_events: int
    host_replay_boundary_events: int
    graph_structure_bytes: int
    estimated_graph_bytes_s: float
    estimated_total_file_bytes_s: float
    estimated_five_minute_file_bytes: int
    estimated_five_minute_file_bytes_with_margin: int
    estimated_disk_seconds: float | None
    disk_seconds_at_observed_average: float | None
    five_minute_payload_bytes: int
    capacities: list[Capacity]
    limitations: tuple[str, ...]


def rolling_peak(values: list[int], width: int, bucket_s: float = 0.01) -> float:
    """Maximum sliding window; pad short histories with zero, retaining window width."""
    if width < 1 or bucket_s <= 0:
        raise ValueError("window width and bucket duration must be positive")
    queue: deque[int] = deque()
    total = maximum = 0
    for value in values:
        if value < 0:
            raise ValueError("negative production")
        queue.append(value)
        total += value
        if len(queue) > width:
            total -= queue.popleft()
        maximum = max(maximum, total)
    return maximum / (width * bucket_s)


def simulate(events: list[int], payloads: list[int], interval_ms: int, service: Service) -> Capacity:
    """Replay earliest possible bucket arrivals and retain ownership through completion.

    Copies and Writer are serialized independently. Host storage is reserved at
    copy submission and held through publication. All arrivals in a bucket occur
    at its beginning; the extra boundary-bucket allowance covers unknown ordering.
    Device slots count fixed events only; Routing pool demand remains independent.
    """
    if len(events) != len(payloads) or interval_ms < 5 or interval_ms % 5:
        raise ValueError("matching buckets and a positive five-ms collection multiple required")
    if any(value < 0 for value in (*events, *payloads)):
        raise ValueError("negative production")
    pending_events = pending_bytes = device = host = peak_device = peak_host = 0
    copy_end = writer_end = 0.0
    completions: list[tuple[float, int, int]] = []
    step = interval_ms // 5
    copy_busy = writer_busy = 0.0

    def retire(now: float) -> None:
        nonlocal device, host
        while completions and completions[0][0] <= now:
            _, kind, amount = heapq.heappop(completions)
            if kind == 0:
                device -= amount
            else:
                host -= amount

    for index in range(len(events) * 2 + step + 1):
        now = index * 0.005
        retire(now)
        if index % 2 == 0 and index // 2 < len(events):
            bucket = index // 2
            pending_events += events[bucket]
            pending_bytes += events[bucket] * EVENT_BYTES + payloads[bucket]
            device += events[bucket]
            peak_device = max(peak_device, device)
        if index % step or not pending_bytes:
            continue
        # Each bounded file includes its header. Partial chunks are not free.
        chunks = math.ceil(pending_events / 4095)
        data = pending_bytes + chunks * 128
        copy_busy += service.copy_delay_s + pending_bytes / service.copy_bytes_s
        writer_busy += service.writer_stall_s + data / service.writer_bytes_s
        copy_end = max(now, copy_end) + service.copy_delay_s + pending_bytes / service.copy_bytes_s
        writer_end = max(copy_end, writer_end) + service.writer_stall_s + data / service.writer_bytes_s
        host += pending_bytes + chunks * 128
        peak_host = max(peak_host, host)
        heapq.heappush(completions, (copy_end, 0, pending_events))
        heapq.heappush(completions, (writer_end, 1, pending_bytes + chunks * 128))
        pending_events = pending_bytes = 0
    duration = max(len(events) * 0.01, 0.01)
    total_bytes = sum(events) * EVENT_BYTES + sum(payloads)
    boundary_events = max(events, default=0)
    boundary_bytes = max((n * EVENT_BYTES + b for n, b in zip(events, payloads, strict=True)), default=0)
    device_need = peak_device + boundary_events
    host_need = peak_host + boundary_bytes + service.host_other_bytes
    return Capacity(
        interval_ms=interval_ms,
        device_slots=device_need,
        host_bytes=host_need,
        doubled_device_slots=device_need * 2,
        doubled_host_bytes=host_need * 2,
        device_logical_bytes=device_need * 272 + 2888,
        doubled_device_logical_bytes=device_need * 544 + 2888,
        exceeds_default_event_slots=device_need > 16384,
        exceeds_default_device_logical_budget=device_need * 272 + 2888 > 8 * MIB,
        exceeds_default_host_payload_budget=host_need > 32 * MIB,
        sustained_overload=(
            total_bytes / duration > min(service.copy_bytes_s, service.writer_bytes_s)
            or copy_busy > duration
            or writer_busy > duration
        ),
    )


def estimate(
    histogram: Histogram, service: Service | None = None, *, disk_bytes: int = 1024 * MIB, metadata_bytes: int = 8 * MIB
) -> Estimate:
    """Estimate payload demand and quota lifetime; never hide missing overhead."""
    if disk_bytes <= metadata_bytes or metadata_bytes < 0:
        raise ValueError("disk quota must exceed nonnegative metadata reserve")
    events = [sum(bucket[:6]) for bucket in histogram.buckets]
    payloads = [bucket[6] for bucket in histogram.buckets]
    seconds = len(events) * 0.01
    total = sum(events) * EVENT_BYTES + sum(payloads)
    rate = total / seconds if seconds else 0
    reserve = metadata_bytes + (service.disk_other_bytes if service is not None else 0)
    available = max(0, disk_bytes - reserve - histogram.graph_structure_bytes)
    host_forward = sum(bucket[0] for bucket in histogram.host_graph_buckets)
    host_capture = sum(bucket[1] for bucket in histogram.host_graph_buckets)
    host_replay = sum(bucket[2] for bucket in histogram.host_graph_buckets)
    host_seconds = len(histogram.host_graph_buckets) * 0.01
    # Host forward boundaries proxy eager/Replay calls; replay is retained separately,
    # never added again. Each forward additionally estimates two Device activity points.
    host_graph_rate = (host_forward * 2 + host_capture) / host_seconds if host_seconds else 0
    native_graph_rate = sum(bucket[7] for bucket in histogram.buckets) / seconds if seconds else 0
    graph_rate = (host_graph_rate + native_graph_rate) * EVENT_BYTES
    routing_count = sum(bucket[5] for bucket in histogram.buckets)
    # One KiB per Routing attachment is an explicit adjustable sizing assumption,
    # not a measured serialization bound. Account one partially filled chunk per poll.
    projected_event_rate = sum(events) / seconds + host_graph_rate + native_graph_rate if seconds else 0
    event_headers_rate = (projected_event_rate / 4095 + 10) * 128 if seconds else 0
    headers_rate = (routing_count * 1024 / seconds if seconds else 0) + event_headers_rate
    file_rate = rate + graph_rate + headers_rate
    return Estimate(
        evidence=Evidence.INCOMPLETE
        if histogram.outside_window or histogram.host_outside_window or histogram.arithmetic_overflow or not seconds
        else Evidence.ESTIMATED,
        duration_s=seconds,
        events=sum(events),
        routing_payload_bytes=sum(payloads),
        event_bytes_s=sum(events) * EVENT_BYTES / seconds if seconds else 0,
        routing_payload_bytes_s=sum(payloads) / seconds if seconds else 0,
        peak_event_bytes_s={
            str(ms): rolling_peak([n * EVENT_BYTES for n in events], ms // 10) for ms in (10, 100, 1000)
        },
        peak_routing_payload_bytes_s={str(ms): rolling_peak(payloads, ms // 10) for ms in (10, 100, 1000)},
        graph_activity_estimate_events=sum(bucket[7] for bucket in histogram.buckets),
        host_forward_boundary_events=host_forward,
        host_capture_boundary_events=host_capture,
        host_replay_boundary_events=host_replay,
        graph_structure_bytes=histogram.graph_structure_bytes,
        estimated_graph_bytes_s=graph_rate,
        estimated_total_file_bytes_s=file_rate,
        estimated_five_minute_file_bytes=math.ceil(file_rate * 300) + histogram.graph_structure_bytes + reserve,
        estimated_five_minute_file_bytes_with_margin=math.ceil(file_rate * 600)
        + histogram.graph_structure_bytes
        + reserve,
        estimated_disk_seconds=available / file_rate if file_rate else None,
        disk_seconds_at_observed_average=available / rate if rate else None,
        five_minute_payload_bytes=math.ceil(rate * 300),
        capacities=[
            simulate([n + b[7] for n, b in zip(events, histogram.buckets, strict=True)], payloads, ms, service)
            for ms in (10, 25, 50, 100, 200)
        ]
        if service is not None
        and not histogram.outside_window
        and not histogram.host_outside_window
        and not histogram.arithmetic_overflow
        and seconds
        else [],
        limitations=(
            "Includes startup/warmup/drain; Device origin is not aligned with the client measurement.",
            "Graph activity uses Compute Hooks and Host forward proxies, not implemented target Device events.",
            "File overhead assumes 1024 bytes/Routing attachment plus full and partial 100-ms event chunk headers.",
            "Routing payload excludes attachment serialization headers and temporary-file overlap.",
            "Quota lifetime is a payload upper bound; five-minute size is a payload lower bound.",
            "Twice-demand capacities are candidates, not measured loss-free guarantees.",
            (
                "Capacity replay includes native Graph activity estimates; "
                "Host/eager Graph queues and tensor pools need extra budget."
            ),
            "Device logical sizing uses default Endpoint/Context/sampler counts; allocation alignment is additional.",
            "No cross-Device timestamp alignment or indefinite disk sustainability is established.",
        ),
    )
