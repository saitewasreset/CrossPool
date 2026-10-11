from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest

import xpool.devkit.timeline.diagnostics
from xpool.devkit.timeline.diagnostics import operation
from xtest.harness.support.config import install_test_config, reset_global_config, synthetic_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.mark.parametrize("enabled", [False, True])
def test_operation_edges_and_exception_propagation(enabled: bool, caplog: pytest.LogCaptureFixture) -> None:
    config = synthetic_config()
    config = config.model_copy(
        update={
            "debug": config.debug.model_copy(
                update={"timeline": config.debug.timeline.model_copy(update={"diagnostics": enabled})}
            )
        }
    )
    install_test_config(config=config)
    failure = RuntimeError("diagnostic probe")
    with caplog.at_level(logging.INFO):
        with operation("probe", device=2, producer=7):
            if enabled:
                assert "edge=enter" in caplog.records[-1].getMessage()
        with pytest.raises(RuntimeError) as raised:
            with operation("failed_probe", device=2, producer=7):
                raise failure
    assert raised.value is failure
    messages = [record.getMessage() for record in caplog.records]
    if enabled:
        assert [message.split("edge=")[1].split()[0] for message in messages] == ["enter", "exit", "enter", "exception"]
        assert all(f"pid={os.getpid()}" in message and "device=2 producer=7" in message for message in messages)
        assert all("begin_ns=" in message and "elapsed_ns=" in message for message in messages)
    else:
        assert messages == []


def test_successful_poll_pairs_are_sampled(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    config = synthetic_config()
    config = config.model_copy(
        update={
            "debug": config.debug.model_copy(
                update={"timeline": config.debug.timeline.model_copy(update={"diagnostics": True})}
            )
        }
    )
    install_test_config(config=config)
    samples = iter([0, 10_000_000, 20_000_000, 30_000_000, 5_000_000_000, 5_010_000_000])
    monkeypatch.setattr(xpool.devkit.timeline.diagnostics, "time", SimpleNamespace(monotonic_ns=lambda: next(samples)))
    with caplog.at_level(logging.INFO):
        for poll in range(3):
            with operation("sampled_python_probe"):
                pass
    assert len(caplog.records) == 4
    assert [record.getMessage().split("edge=")[1].split()[0] for record in caplog.records] == [
        "enter",
        "exit",
        "enter",
        "exit",
    ]
