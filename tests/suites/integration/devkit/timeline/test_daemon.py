from http import HTTPStatus
from pathlib import Path
from uuid import uuid4

import pytest

from xpool.config import TimelineDebugConfig
from xpool.devkit.timeline.models import GrantRequest, Producer, ProducerRequest
from xpool.devkit.timeline.session import Session
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.config import reset_global_config, synthetic_config
from xtest.harness.support.service.daemon import create_app, deterministic_daemon_dependencies, request

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


def test_daemon_api_checks_creation_identity_and_deduplicates_credit(tmp_path: Path) -> None:
    app = create_app(synthetic_config())
    options = TimelineDebugConfig(enable=True, outdir=tmp_path)
    owner = Session(tmp_path, options.model_dump(mode="json"), ["daemon:host"])
    app.state.control_plane.timeline_session = owner
    identity = ProcUniqId.current()
    registration = ProducerRequest(
        startup_id=uuid4(),
        pid=identity.pid,
        create_time=identity.create_time,
        role="daemon",
        slot="daemon",
        source="host",
    )
    payload = registration.model_dump(mode="json")
    first = request(app, "POST", "/timeline/register", json=payload)
    assert first.status_code == HTTPStatus.OK
    assert request(app, "POST", "/timeline/register", json=payload).json() == first.json()
    producer = Producer.model_validate(first.json())
    wrong_creation = {**payload, "create_time": identity.create_time + 1}
    assert request(app, "POST", "/timeline/register", json=wrong_creation).status_code == HTTPStatus.CONFLICT
    grant = GrantRequest(
        producer_id=producer.producer_id, startup_id=producer.startup_id, sequence=1, bytes_requested=4 << 20
    ).model_dump(mode="json")
    first_grant = request(app, "POST", "/timeline/grant", json=grant)
    assert first_grant.status_code == HTTPStatus.OK
    assert request(app, "POST", "/timeline/grant", json=grant).json() == first_grant.json()
    assert owner.manifest.granted_bytes == 4 << 20
    assert (
        request(app, "POST", "/timeline/grant", json={**grant, "startup_id": str(uuid4())}).status_code
        == HTTPStatus.CONFLICT
    )
    assert (
        request(
            app, "POST", "/timeline/register", json={**payload, "source": "device", "device_uuid": "GPU-test"}
        ).status_code
        == HTTPStatus.UNPROCESSABLE_ENTITY
    )
    owner.close()
