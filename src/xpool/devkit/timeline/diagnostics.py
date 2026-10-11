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


@contextmanager
def operation(name: str, *, device: int = -1, producer: int = 0) -> Iterator[None]:
    """Emit paired edges only when diagnostics are enabled; preserve exceptions.

    Entry precedes the observed call. Missing exit narrows the blocked interval,
    but does not prove which internal dependency prevented progress.
    """
    if not get_global_config().debug.timeline.diagnostics:
        yield
        return
    begin = time.monotonic_ns()
    identity = (os.getpid(), threading.get_native_id(), device, producer, name, begin)
    message = "timeline diagnostic pid=%s tid=%s device=%s producer=%s operation=%s begin_ns=%s edge=%s elapsed_ns=%s"
    logger.info(message, *identity, "enter", 0)
    try:
        yield
    except BaseException:
        logger.info(message, *identity, "exception", time.monotonic_ns() - begin)
        raise
    else:
        logger.info(message, *identity, "exit", time.monotonic_ns() - begin)
