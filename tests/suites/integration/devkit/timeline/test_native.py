import gc
import time
from concurrent.futures import ThreadPoolExecutor

import xpool.native
from xpool.devkit.timeline.format import RECORD, RECORD_BYTES


def test_named_record_preserves_version_one_wire_bytes() -> None:
    options = xpool.native.debug.TimelineOptions(True, 8 << 20, 32 << 20, 4096)
    recorder = xpool.native.devkit.timeline.Recorder(options)
    record = xpool.native.devkit.timeline.Record(
        sequence=999,
        timestamp=0x0807060504030201,
        kind=3,
        site=4,
        generation_high=2**63 + 5,
        generation_low=6,
        endpoint_creator=7,
        endpoint_index=8,
        operation=9,
        instance=10,
        layer=11,
        lane=12,
        lease=13,
        result=14,
        rows=15,
    )
    assert record.generation_high == 2**63 + 5
    assert record.reserved == 0
    assert recorder.record(record)
    recorder.stop()
    chunk = recorder.collect()
    assert chunk is not None
    with memoryview(chunk) as payload:
        assert len(payload) == RECORD_BYTES == 128
        # File version one owns this order; commit assigns the sequence even
        # when the caller supplies a different value in its immutable input.
        assert bytes(payload) == RECORD.pack(1, 0x0807060504030201, 3, 4, 2**63 + 5, *range(6, 16), 0)
    chunk.release()
    del chunk
    assert recorder.drained()
    recorder.close()


def test_host_pool_concurrency_and_receipt_lifetime() -> None:
    options = xpool.native.debug.TimelineOptions(True, 8 << 20, 32 << 20, 4096)
    recorder = xpool.native.devkit.timeline.Recorder(options)
    record = xpool.native.devkit.timeline.Record(timestamp=1, kind=1)
    with ThreadPoolExecutor(max_workers=8) as workers:
        accepted = list(workers.map(lambda _: recorder.record(record), range(1000)))
    counts = recorder.counters()
    assert counts.attempted == 1000
    assert counts.committed == sum(accepted)
    assert counts.committed + counts.dropped == counts.attempted
    lease = recorder.collect()
    assert lease is not None
    view = memoryview(lease)
    original = bytes(view)
    lease.release()
    del lease
    gc.collect()
    assert bytes(view) == original
    recorder.stop()
    assert not recorder.drained()
    view.release()
    del view
    while (chunk := recorder.collect()) is not None:
        chunk.release()
        del chunk
    assert recorder.drained()
    recorder.close()


def test_host_chunk_holds_pool_until_explicit_discard() -> None:
    options = xpool.native.debug.TimelineOptions(True, 8 << 20, 32 << 20, 4096)
    recorder = xpool.native.devkit.timeline.Recorder(options)
    assert recorder.record(xpool.native.devkit.timeline.Record(timestamp=time.monotonic_ns(), kind=1))
    chunk = recorder.collect()
    assert chunk is not None
    recorder.stop()
    assert not recorder.drained()
    chunk.release()
    del chunk
    assert recorder.drained()
    recorder.close()
