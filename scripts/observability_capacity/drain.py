"""Bounded synthetic D2H/publication probe; these are not timeline artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import queue
import shutil
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

type Seconds = Annotated[FiniteFloat, Field(ge=0)]


class Transfer(BaseModel):
    """Independent copy, publication and optional durability observations."""

    model_config = ConfigDict(extra="forbid")
    sequence: int = Field(ge=0)
    submitted_s: Seconds
    completed_s: Seconds
    copy_device_ms: Seconds
    write_begin_s: Seconds
    published_s: Seconds
    synced_s: Seconds | None
    bytes: int = Field(gt=128)
    crc32: int = Field(ge=0, le=2**32 - 1)
    host_wait_s: Seconds
    released_s: Seconds

    @model_validator(mode="after")
    def validate_ownership(self) -> Transfer:
        """Reject impossible ownership intervals rather than invent service rates."""
        if not self.submitted_s <= self.completed_s <= self.write_begin_s <= self.published_s <= self.released_s:
            raise ValueError("invalid transfer ownership ordering")
        if self.synced_s is not None and not self.write_begin_s <= self.synced_s <= self.published_s:
            raise ValueError("sync timestamp outside publication")
        return self


class DrainResult(BaseModel):
    """Finite-window capability evidence, never an indefinite disk guarantee."""

    model_config = ConfigDict(extra="forbid")
    device_uuid: str
    host_slots: int
    chunk_bytes: int
    origin_monotonic_s: Seconds
    elapsed_s: FiniteFloat = Field(gt=0)
    file_bytes: int = Field(ge=0)
    sync: bool
    transfers: list[Transfer]
    verified: bool

    @model_validator(mode="after")
    def validate_extent(self) -> DrainResult:
        """Require nonempty, sequential and byte-consistent verified evidence."""
        if not self.verified or not self.transfers:
            raise ValueError("probe lacks verified transfers")
        if [item.sequence for item in self.transfers] != list(range(len(self.transfers))):
            raise ValueError("probe transfer sequence has a gap or duplicate")
        if self.file_bytes != sum(item.bytes for item in self.transfers):
            raise ValueError("probe byte accounting disagrees")
        if any(
            item.released_s > self.elapsed_s
            or item.bytes != self.chunk_bytes
            or (item.synced_s is not None) != self.sync
            for item in self.transfers
        ):
            raise ValueError("probe timing, size or sync policy disagrees")
        return self


class DrainSettings(BaseModel):
    """Admission limits checked before any probe storage is allocated."""

    seconds: float = Field(default=30, gt=0, le=300)
    max_bytes: int = Field(default=256 * 1024 * 1024, ge=1024 * 1024, le=8 * 1024**3)
    host_slots: int = Field(default=2, ge=1, le=32)
    chunk_bytes: int = Field(default=1024 * 1024, ge=384, le=1024 * 1024)
    sync: bool = False


def publish(path: Path, payload: memoryview, sequence: int, *, sync: bool) -> tuple[int, float, float | None]:
    """CRC-protected synthetic file, exact 128-byte header, same-directory rename.

    EXPDRAIN deliberately differs from XPOOLC03: payload bytes are synthetic,
    not semantic events. The I/O shape matches a bounded raw publication.
    """
    checksum = zlib.crc32(payload)
    header = bytearray(128)
    struct.pack_into("<8sQQII", header, 0, b"EXPDRAIN", sequence, len(payload), checksum, 0)
    struct.pack_into("<I", header, 28, zlib.crc32(header))
    temporary = path.with_suffix(".partial")
    synced = None
    with temporary.open("xb", buffering=0) as handle:
        for part in (memoryview(header), payload):
            remaining = part
            while remaining:
                count = handle.write(remaining)
                if count is None or count <= 0:
                    raise OSError(f"short write without progress: {temporary}")
                remaining = remaining[count:]
        if sync:
            os.fdatasync(handle.fileno())
            synced = time.monotonic()
    temporary.replace(path)
    return checksum, time.monotonic(), synced


def verify(path: Path) -> None:
    """Verify the complete synthetic file, including exact payload extent."""
    with path.open("rb") as handle:
        header = handle.read(128)
        if len(header) != 128:
            raise ValueError(f"truncated probe header: {path}")
        magic, _, size, checksum, header_checksum = struct.unpack_from("<8sQQII", header)
        checked_header = bytearray(header)
        struct.pack_into("<I", checked_header, 28, 0)
        if zlib.crc32(checked_header) != header_checksum:
            raise ValueError(f"invalid probe header CRC: {path}")
        payload = handle.read()
    if magic != b"EXPDRAIN" or len(payload) != size or zlib.crc32(payload) != checksum:
        raise ValueError(f"invalid probe file: {path}")


def run(output: Path, settings: DrainSettings) -> DrainResult:
    """Overlap a bounded pinned pool with one Writer; propagate worker failures."""
    import torch

    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < settings.max_bytes + 1024**3:
        raise OSError("insufficient free space for probe reservation and safety reserve")
    source = torch.full((settings.chunk_bytes - 128,), 165, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    # This initialization belongs only to the probe's private CUDA context.
    torch.cuda.current_stream().synchronize()
    hosts = [torch.empty_like(source, device="cpu", pin_memory=True) for _ in range(settings.host_slots)]
    available: queue.Queue[int] = queue.Queue(settings.host_slots)
    for index in range(settings.host_slots):
        available.put_nowait(index)
    work: queue.Queue[tuple[int, int, float, float, float, float] | None] = queue.Queue(settings.host_slots)
    failures: queue.Queue[BaseException] = queue.Queue()
    transfers: list[Transfer] = []
    origin = time.monotonic()

    def writer() -> None:
        while (item := work.get()) is not None:
            index, sequence, submitted, completed, device_ms, host_wait = item
            try:
                begin = time.monotonic()
                checksum, published, synced = publish(
                    output / f"{sequence:08d}.bin", memoryview(hosts[index].numpy()), sequence, sync=settings.sync
                )
                transfers.append(
                    Transfer(
                        sequence=sequence,
                        submitted_s=submitted - origin,
                        completed_s=completed - origin,
                        copy_device_ms=device_ms,
                        write_begin_s=begin - origin,
                        published_s=published - origin,
                        synced_s=synced - origin if synced is not None else None,
                        bytes=settings.chunk_bytes,
                        crc32=checksum,
                        host_wait_s=host_wait,
                        released_s=time.monotonic() - origin,
                    )
                )
            except BaseException as error:
                failures.put(error)
            finally:
                available.put_nowait(index)

    thread = threading.Thread(target=writer, name="capacity-probe-writer", daemon=False)
    thread.start()
    try:
        sequence = 0
        while (
            time.monotonic() - origin < settings.seconds and (sequence + 1) * settings.chunk_bytes <= settings.max_bytes
        ):
            if not failures.empty():
                raise failures.get_nowait()
            waiting = time.monotonic()
            try:
                index = available.get(timeout=0.1)
            except queue.Empty:
                continue
            submitted = time.monotonic()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                begin.record()
                hosts[index].copy_(source, non_blocking=True)
                end.record()
            while not end.query():
                time.sleep(0.001)
            work.put((index, sequence, submitted, time.monotonic(), begin.elapsed_time(end), submitted - waiting))
            sequence += 1
    finally:
        work.put(None)
        thread.join()
    if not failures.empty():
        raise failures.get_nowait()
    elapsed = time.monotonic() - origin
    # Validate copied content as well as the I/O publication and exact file extent.
    expected_checksum = zlib.crc32(bytes([165]) * (settings.chunk_bytes - 128))
    if any(item.crc32 != expected_checksum for item in transfers):
        raise ValueError("probe Device-to-Host payload differs from the initialized pattern")
    hashes: dict[str, str] = {}
    for path in sorted(output.glob("*.bin")):
        verify(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    import json

    (output / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
    result = DrainResult(
        device_uuid=torch.cuda.get_device_properties(0).uuid,
        host_slots=settings.host_slots,
        chunk_bytes=settings.chunk_bytes,
        origin_monotonic_s=origin,
        elapsed_s=elapsed,
        file_bytes=sum(item.bytes for item in transfers),
        sync=settings.sync,
        transfers=transfers,
        verified=True,
    )
    (output / "drain.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


def main() -> None:
    """Run one process on the operator-selected visible Device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    run(args.output, DrainSettings(seconds=args.seconds, max_bytes=args.max_bytes, sync=args.sync))


if __name__ == "__main__":
    main()
