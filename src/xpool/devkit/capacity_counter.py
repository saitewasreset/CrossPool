"""Experimental bounded production histograms, exported after local quiescence."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from xpool.native import RuntimeRole

installed = False
install_lock = Lock()

runtime_roles = frozenset({RuntimeRole.INSTANCE, RuntimeRole.ATNAGENT, RuntimeRole.FFNAGENT})
type Nonnegative = Annotated[int, Field(ge=0, le=2**64 - 1)]
type Bucket = tuple[
    Nonnegative, Nonnegative, Nonnegative, Nonnegative, Nonnegative, Nonnegative, Nonnegative, Nonnegative
]
COLUMNS = (
    "transport",
    "atnagent",
    "coordinator",
    "ffnagent",
    "association",
    "routing_published",
    "routing_bytes",
    "graph_activity_estimate",
)


class Histogram(BaseModel):
    """One independent process/Device domain, including startup and drain activity."""

    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    pid: int = Field(gt=0)
    role: Literal["instance", "atnagent", "ffnagent"]
    device: str
    origin_ns: Nonnegative
    last_ns: Nonnegative
    outside_window: Nonnegative
    arithmetic_overflow: Nonnegative
    allocation_bytes: Nonnegative
    bucket_ns: int = Field(default=10_000_000, ge=10_000_000, le=10_000_000)
    buckets: list[Bucket]
    host_counter_bytes_estimate: Nonnegative = 0
    host_origin_ns: Nonnegative = 0
    host_outside_window: Nonnegative = 0
    host_graph_buckets: list[tuple[Nonnegative, Nonnegative, Nonnegative]] = Field(default_factory=list)
    graph_structure_bytes: Nonnegative = 0

    @model_validator(mode="after")
    def validate_clock(self) -> Histogram:
        """Reject reversed clocks and impossible bucket coverage."""
        if (self.host_origin_ns == 0) != (not self.host_graph_buckets):
            raise ValueError("Host counter origin and buckets disagree")
        if self.last_ns < self.origin_ns:
            raise ValueError("last Device time precedes origin")
        if (self.origin_ns == 0) != (not self.buckets):
            raise ValueError("empty activity and Device origin disagree")
        if self.buckets and len(self.buckets) > (self.last_ns - self.origin_ns) // self.bucket_ns + 1:
            raise ValueError("histogram extends beyond last observed activity")
        return self


def install() -> None:
    """Install experiment wrappers once in the current process."""
    global installed
    with install_lock:
        if installed:
            return
        _install()
        installed = True


def _install() -> None:
    """Export only at established local-production retirement boundaries.

    Export failures propagate instead of being converted into missing evidence.
    Counter storage is released only by native finish after local quiescence.
    """
    import xpool.native
    from xpool.bootstrap import get_runtime_role
    from xpool.config import get_global_config

    config = get_global_config().debug.capacity_counter
    outdir = config.outdir
    if outdir is None:
        raise RuntimeError("xpool capacity counter requires outdir")
    outdir.mkdir(parents=True, exist_ok=True)
    role = get_runtime_role()
    exported = False
    host_origin = 0
    host_counts: list[list[int]] = []

    host_outside = 0
    host_lock = Lock()

    def count_graph(column: int) -> None:
        nonlocal host_origin, host_outside
        with host_lock:
            now = time.monotonic_ns()
            if host_origin == 0:
                host_origin = now
            bucket = (now - host_origin) // 10_000_000
            if bucket >= config.bucket_count:
                host_outside += 1
                return
            while len(host_counts) <= bucket:
                host_counts.append([0, 0, 0])
            host_counts[bucket][column] += 1

    def wrap_graph[R](original: Callable[..., R], column: int) -> Callable[..., R]:
        @wraps(original)
        def observed(*args: object, **kwargs: object) -> R:
            count_graph(column)
            try:
                return original(*args, **kwargs)
            finally:
                count_graph(column)

        return observed

    if role is RuntimeRole.INSTANCE:
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner

        # Forward counts cover eager calls as well as calls containing Replay.
        setattr(ModelRunner, "forward", wrap_graph(ModelRunner.forward, 0))
        for runner in (DecodeCudaGraphRunner, PrefillCudaGraphRunner):
            setattr(runner, "capture", wrap_graph(runner.capture, 1))
            setattr(runner, "execute", wrap_graph(runner.execute, 2))

    def export(structure_bytes: int = 0) -> None:
        nonlocal exported
        if exported:
            return
        snapshot = xpool.native.devkit.capacity_counter.finish()
        import torch

        value = Histogram(
            pid=os.getpid(),
            role=TypeAdapter(Literal["instance", "atnagent", "ffnagent"]).validate_python(role.name.lower()),
            device=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            origin_ns=snapshot.origin_ns,
            last_ns=snapshot.last_ns,
            outside_window=snapshot.outside_window,
            arithmetic_overflow=snapshot.arithmetic_overflow,
            allocation_bytes=snapshot.allocation_bytes,
            buckets=TypeAdapter(list[Bucket]).validate_python(snapshot.buckets),
            graph_structure_bytes=structure_bytes,
            host_counter_bytes_estimate=sys.getsizeof(host_counts)
            + sum(sys.getsizeof(bucket) + sum(sys.getsizeof(value) for value in bucket) for bucket in host_counts),
            host_origin_ns=host_origin,
            host_outside_window=host_outside,
            host_graph_buckets=[(bucket[0], bucket[1], bucket[2]) for bucket in host_counts],
        )
        write_histogram(outdir / f"counter.{os.getpid()}.json", value)
        exported = True

    if role is RuntimeRole.INSTANCE:
        from xpool.runtime.instance import InstanceRankRuntime

        original = InstanceRankRuntime.detach_arena

        @wraps(original)
        def detach(runtime: InstanceRankRuntime) -> None:
            attached = runtime.arena_handle is not None
            original(runtime)
            # Native detach synchronizes the Instance Device before closing IPC.
            # Failed detach leaves counters owned and no safe readout is claimed.
            if attached:
                export()

        setattr(InstanceRankRuntime, "detach_arena", detach)
    else:
        from xpool.fabric import FabricParticipantPhase
        from xpool.runtime.agent import Agent

        advance = Agent.advance_fabric_lifecycle

        @wraps(advance)
        def drained(agent: Agent) -> None:
            advance(agent)
            if (
                agent.participant_report is not None
                and agent.participant_report.phase is FabricParticipantPhase.DRAINED
            ):
                structure_bytes = 0
                if not exported and role is RuntimeRole.FFNAGENT:
                    from xpool.devkit.graph_observer import graph_snapshot
                    from xpool.runtime.ffnagent import FfnAgent

                    if not isinstance(agent, FfnAgent):
                        raise RuntimeError("xpool graph counter received an unexpected Agent role")
                    graph = xpool.native.devkit.graph_observer.read()
                    if graph is None:
                        raise RuntimeError("xpool counter missing installed native Graph structure")
                    payload = graph_snapshot(agent, graph).model_dump_json().encode()
                    path = outdir / f"graph.{os.getpid()}.json"
                    temporary = path.with_suffix(".partial")
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                    structure_bytes = len(payload)
                export(structure_bytes)

        setattr(Agent, "advance_fabric_lifecycle", drained)


def write_histogram(path: Path, value: Histogram) -> None:
    """Publish one complete histogram atomically; I/O failures remain visible."""
    temporary = path.with_suffix(".partial")
    temporary.write_text(value.model_dump_json() + "\n", encoding="utf-8")
    temporary.replace(path)
