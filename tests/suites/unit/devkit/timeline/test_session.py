from pathlib import Path
from uuid import uuid4

import pytest

from xpool.config import TimelineDebugConfig
from xpool.devkit.timeline.models import Grant, GrantRequest, ProducerRequest, SessionManifest
from xpool.devkit.timeline.session import Session, publish_metadata
from xpool.devkit.timeline.writer import Writer
from xpool.service.errors import XpoolClientError


def session(tmp_path: Path) -> Session:
    config = TimelineDebugConfig(enable=True, outdir=tmp_path).model_dump(mode="json")
    return Session(tmp_path, config, ["daemon:host"])


def test_retries_do_not_duplicate_identity_or_credit(tmp_path: Path) -> None:
    owner = session(tmp_path)
    request = ProducerRequest(startup_id=uuid4(), pid=1, create_time=1, role="daemon", slot="daemon", source="host")
    producer = owner.register(request)
    assert owner.register(request) == producer
    grant = GrantRequest(
        producer_id=producer.producer_id, startup_id=request.startup_id, sequence=1, bytes_requested=4 << 20
    )
    assert owner.grant(grant) == owner.grant(grant)
    manifest = SessionManifest.model_validate_json((owner.directory / "manifest.json").read_bytes())
    assert manifest.granted_bytes == 4 << 20
    assert len(manifest.producers) == 1
    with pytest.raises(ValueError, match="creation identity"):
        owner.register(request.model_copy(update={"startup_id": uuid4()}))
    with pytest.raises(ValueError, match="stale, skipped"):
        owner.grant(grant.model_copy(update={"sequence": 3}))


def test_quota_keeps_grants_after_close(tmp_path: Path) -> None:
    owner = session(tmp_path)
    producer = owner.register(
        ProducerRequest(startup_id=uuid4(), pid=1, create_time=1, role="daemon", slot="daemon", source="host")
    )
    owner.maximum = owner.reserve + owner.chunk_bytes
    request = GrantRequest(
        producer_id=producer.producer_id, startup_id=producer.startup_id, sequence=1, bytes_requested=4 << 20
    )
    assert owner.grant(request).bytes_granted == owner.chunk_bytes
    assert owner.grant(request.model_copy(update={"sequence": 2})).bytes_granted == 0
    owner.close()
    assert owner.manifest.granted_bytes == owner.chunk_bytes
    with pytest.raises(ValueError, match="closed"):
        owner.grant(request.model_copy(update={"sequence": 3}))


def test_metadata_failure_preserves_partial_and_original(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    publish_metadata(path, b"original", 10)
    partial = path.with_suffix(".json.partial")
    partial.write_bytes(b"failed")
    with pytest.raises(OSError, match="publication failed"):
        publish_metadata(path, b"new", 10)
    assert path.read_bytes() == b"original"
    assert partial.read_bytes() == b"failed"
    with pytest.raises(ValueError, match="metadata_limit"):
        publish_metadata(tmp_path / "large.json", b"too large", 2)


def test_invalid_budget_and_expected_inventory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="set or unset"):
        TimelineDebugConfig(enable=True)
    with pytest.raises(ValueError, match="buffers require"):
        TimelineDebugConfig(host_buffer_bytes=4096)
    config = TimelineDebugConfig(enable=True, outdir=tmp_path).model_dump(mode="json")
    with pytest.raises(ValueError, match="metadata bytes"):
        Session(tmp_path, config, [f"slot-{i}:host" for i in range(300)])
    assert not list(tmp_path.iterdir())


def test_lost_grant_reply_retries_without_double_charging(tmp_path: Path) -> None:
    owner = session(tmp_path)
    producer = owner.register(
        ProducerRequest(startup_id=uuid4(), pid=1, create_time=1, role="daemon", slot="daemon", source="host")
    )
    first = True

    def lose_reply(request: GrantRequest) -> Grant:
        nonlocal first
        reply = owner.grant(request)
        if first:
            first = False
            raise XpoolClientError("transport", "reply lost after daemon reservation")
        return reply

    writer = Writer(producer, owner.chunk_bytes, lose_reply)
    assert not writer.replenish()
    assert writer.credit == 0
    assert writer.replenish()
    assert writer.credit == owner.manifest.granted_bytes == 4 * owner.chunk_bytes
    assert writer.request_sequence == 2


def test_host_budget_includes_reserved_metadata_before_pool_geometry() -> None:
    with pytest.raises(ValueError, match="reserved metadata, control and two chunks"):
        TimelineDebugConfig(host_buffer_bytes=131072 + 2 * (1 << 20))
    assert TimelineDebugConfig(host_buffer_bytes=3 << 20).host_buffer_bytes == 3 << 20
