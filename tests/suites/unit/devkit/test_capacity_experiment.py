from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.observability_capacity.analysis import Evidence, Service, estimate, rolling_peak, simulate

from xpool.config import CapacityCounterDebugConfig
from xpool.devkit.capacity_counter import Histogram


def histogram(counts: list[int], payload: int = 0) -> Histogram:
    return Histogram(
        pid=1,
        role="instance",
        device="GPU-test",
        origin_ns=100,
        last_ns=100 + max(0, len(counts) - 1) * 10_000_000,
        outside_window=0,
        arithmetic_overflow=0,
        allocation_bytes=1024,
        buckets=[(n, 0, 0, 0, 0, 0, payload, 0) for n in counts],
    )


def test_counter_config_requires_absolute_destination_and_bounded_storage(tmp_path: Path) -> None:
    assert not CapacityCounterDebugConfig().enable
    assert CapacityCounterDebugConfig(enable=True, outdir=tmp_path).bucket_count == 120000
    for kwargs in (
        {"enable": True},
        {"outdir": tmp_path},
        {"enable": True, "outdir": Path("relative")},
        {"bucket_count": 360001},
    ):
        with pytest.raises(ValidationError):
            CapacityCounterDebugConfig.model_validate(kwargs)


def test_quota_is_cumulative_and_attachment_payload_is_separate() -> None:
    value = estimate(histogram([1024] * 100, payload=1024))
    rate = 1024 * 256 * 100 + 1024 * 100
    assert value.event_bytes_s == 1024 * 256 * 100
    assert value.routing_payload_bytes_s == 1024 * 100
    assert value.disk_seconds_at_observed_average == pytest.approx((1016 * 1024**2) / rate)
    assert value.five_minute_payload_bytes == rate * 300


def test_same_average_does_not_imply_same_burst_capacity() -> None:
    service = Service(copy_bytes_s=1e9, writer_bytes_s=1e9)
    steady = simulate([10] * 100, [0] * 100, 100, service)
    burst = simulate([1000] + [0] * 99, [0] * 100, 100, service)
    assert burst.device_slots > steady.device_slots
    assert not steady.sustained_overload


def test_inflight_device_and_writer_ownership_are_independent() -> None:
    events = [100] * 100
    fast = simulate(events, [0] * 100, 25, Service(copy_bytes_s=1e9, writer_bytes_s=1e9))
    writer = simulate(events, [0] * 100, 25, Service(copy_bytes_s=1e9, writer_bytes_s=100))
    copy = simulate(events, [0] * 100, 25, Service(copy_bytes_s=100, writer_bytes_s=1e9))
    assert writer.device_slots == fast.device_slots
    assert writer.host_bytes > fast.host_bytes
    assert copy.device_slots > fast.device_slots
    assert writer.sustained_overload and copy.sustained_overload


def test_all_collection_intervals_and_incomplete_evidence() -> None:
    value = histogram([2] * 100)
    result = estimate(value, Service(copy_bytes_s=1e9, writer_bytes_s=1e9))
    assert [item.interval_ms for item in result.capacities] == [10, 25, 50, 100, 200]
    assert estimate(value.model_copy(update={"outside_window": 1})).evidence is Evidence.INCOMPLETE
    assert estimate(value.model_copy(update={"arithmetic_overflow": 1})).evidence is Evidence.INCOMPLETE
    assert rolling_peak([2, 0, 3], 2) == 150


def test_graph_estimates_keep_replay_separate_and_incomplete_data_has_no_candidates() -> None:
    value = histogram([1] * 100).model_copy(
        update={
            "host_origin_ns": 1,
            "host_graph_buckets": [(2, 0, 2)] * 100,
            "graph_structure_bytes": 4096,
        }
    )
    result = estimate(value)
    assert result.host_forward_boundary_events == 200
    assert result.host_replay_boundary_events == 200
    assert result.estimated_graph_bytes_s == 400 * 256
    assert result.estimated_five_minute_file_bytes > result.five_minute_payload_bytes
    incomplete = value.model_copy(update={"host_outside_window": 1})
    result = estimate(incomplete, Service(copy_bytes_s=1e9, writer_bytes_s=1e9))
    assert result.evidence is Evidence.INCOMPLETE
    assert result.capacities == []


def test_probe_service_evidence_rejects_reversed_or_nonfinite_timing() -> None:
    from scripts.observability_capacity.drain import Transfer

    sample = dict(
        sequence=0,
        submitted_s=1,
        completed_s=2,
        copy_device_ms=1,
        write_begin_s=2,
        published_s=3,
        synced_s=None,
        bytes=1024,
        crc32=0,
        host_wait_s=0,
        released_s=3,
    )
    assert Transfer.model_validate(sample).bytes == 1024
    for update in ({"completed_s": 0}, {"published_s": float("nan")}, {"released_s": 1}, {"synced_s": 4}):
        with pytest.raises(ValidationError):
            Transfer.model_validate(sample | update)
