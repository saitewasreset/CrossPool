"""Opt-in Host operation boundaries for diagnosing blocked native calls."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from xpool.config import get_global_config

logger = logging.getLogger(__name__)
state = threading.local()


@contextmanager
def operation(name: str, *, device: int = -1, producer: int = 0) -> Iterator[None]:
    """Sample successful pairs every five seconds; retain exceptions and slow exits.

    Native tracking independently reports stalled calls even between samples.
    Python-only waits retain their existing timeout/exception ownership.
    """
    if not get_global_config().debug.timeline.diagnostics:
        yield
        return
    begin = time.monotonic_ns()
    identity = (os.getpid(), threading.get_native_id(), device, producer, name, begin)
    message = "timeline diagnostic pid=%s tid=%s device=%s producer=%s operation=%s begin_ns=%s edge=%s elapsed_ns=%s"
    samples: dict[tuple[str, int, int], int]
    if not hasattr(state, "samples"):
        state.samples = {}
    samples = state.samples
    key = (name, device, producer)
    previous = samples.get(key)
    sampled = previous is None or begin - previous >= 5_000_000_000
    if sampled:
        samples[key] = begin
        logger.info(message, *identity, "enter", 0)
    try:
        yield
    except BaseException:
        logger.info(message, *identity, "exception", time.monotonic_ns() - begin)
        raise
    else:
        elapsed = time.monotonic_ns() - begin
        if sampled or elapsed >= 1_000_000_000:
            logger.info(message, *identity, "exit", elapsed)
