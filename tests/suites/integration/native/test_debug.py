"""Typed native debug-option binding contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
import xtest
from xpool.config import DebugConfig, TimelineDebugConfig, TransportObserverDebugConfig
from xpool.native import RuntimeRole
from xtest.harness.native.case import run_native_case


def isolated_default_debug_options_are_frozen() -> None:
    device = torch.cuda.current_device()
    xpool.native.initialize(RuntimeRole.INSTANCE, device, None)
    xpool.native.initialize(RuntimeRole.INSTANCE, device, None)
    with pytest.raises(RuntimeError, match="debug options differ"):
        xpool.native.initialize(
            RuntimeRole.INSTANCE,
            device,
            DebugConfig(
                transport_observer=TransportObserverDebugConfig(enable=True, outdir=Path.cwd())
            ).native_options(),
        )


def test_debug_options_are_read_only() -> None:
    options = DebugConfig().native_options()
    with pytest.raises(AttributeError):
        setattr(options.transport_observer, "enable", True)


@xtest.requirements(device_count=1)
def test_debug_defaults_are_frozen_by_first_initialization(tmp_path: Path) -> None:
    run_native_case(isolated_default_debug_options_are_frozen, workdir=tmp_path / "case")


def isolated_fabric_failure_diagnostics() -> None:
    xpool.native.initialize(
        RuntimeRole.ATNAGENT,
        torch.cuda.current_device(),
        DebugConfig(timeline=TimelineDebugConfig(diagnostics=True)).native_options(),
    )
    with pytest.raises(RuntimeError, match="outside a joined generation"):
        xpool.native.fabric.failure()


@xtest.requirements(device_count=1)
def test_fabric_failure_diagnostics_preserve_validation_error(tmp_path: Path) -> None:
    workdir = tmp_path / "case"
    run_native_case(isolated_fabric_failure_diagnostics, workdir=workdir)
    output = (workdir / "case.log").read_text()
    assert "operation=fabric_failure_lock edge=enter" in output
    assert "operation=fabric_failure_lock edge=exit" in output
    assert "operation=fabric_failure edge=exception" in output
