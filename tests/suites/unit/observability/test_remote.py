import json
from pathlib import Path

import pytest
from local_scripts.observability.remote import execution_environment, read_participants, restricted_path


def test_remote_visibility_and_configuration_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPOOL_DEBUG_FABRIC_OBSERVER_ENABLE", "0")
    monkeypatch.setenv("UV_ENV_FILE", "unrelated.env")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/other-controller")
    environment = execution_environment(Path("/tmp/prototype.toml"), ("GPU-a", "GPU-b"))
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert environment["XPOOL_CONFIG"] == "/tmp/prototype.toml"
    assert "XPOOL_DEBUG_FABRIC_OBSERVER_ENABLE" not in environment
    assert "CUDA_MPS_PIPE_DIRECTORY" not in environment
    assert "UV_ENV_FILE" not in environment


def test_registration_confirms_launch_placement_and_pid(tmp_path: Path) -> None:
    launch = tmp_path / "launch"
    readiness = launch / "readiness"
    readiness.mkdir(parents=True)
    (launch / "xpool.toml").write_text(
        "[scheduler]\nslo={ttft_ms=1000,tbt_ms=50}\n"
        '[atn]\ndevices=[0]\n[ffn]\ndevices=[1]\n[vendor]\nmodel_base_uri="/tmp/models"\n'
        '[[models]]\nid="Qwen/Qwen3-0.6B"\n'
    )
    payload = {
        "ready": False,
        "generation": None,
        "fabric_phase": None,
        "fabric_invocation_failure": None,
        "fabric_owner_failure": None,
        "fabric_control_failure": None,
        "transport_ready": False,
        "instances_initialized": False,
        "mps_status": "online",
        "devices": [0, 1],
        "atnagents": [{"device": 0, "pid": 123, "status": "online"}],
        "ffnagents": [{"device": 1, "pid": 456, "status": "online"}],
        "instances": [],
    }
    evidence = {
        "name": "agent registration",
        "url": "http://localhost/ready",
        "last_status_code": 200,
        "last_response_excerpt": json.dumps(payload),
    }
    path = readiness / "agent-registration.json"
    path.write_text(json.dumps(evidence))
    participants = read_participants(tmp_path, ("GPU-a", "GPU-b"))
    assert [(item.pe, item.pid, item.device_uuid) for item in participants] == [(0, 123, "GPU-a"), (1, 456, "GPU-b")]
    payload["atnagents"][0]["status"] = "offline"
    evidence["last_response_excerpt"] = json.dumps(payload)
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="registration"):
        read_participants(tmp_path, ("GPU-a", "GPU-b"))


def test_forbidden_directory_alias_is_rejected(tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to("/home/incoming")
    with pytest.raises(ValueError, match="forbidden"):
        restricted_path(alias / "anything")
