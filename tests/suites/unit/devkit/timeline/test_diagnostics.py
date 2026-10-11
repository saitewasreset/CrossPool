from __future__ import annotations

import logging
import os

import pytest

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
