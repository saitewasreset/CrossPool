import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from local_scripts.observability.remote import stream_command


@pytest.mark.parametrize("returncode", [0, 7])
def test_command_mirrors_both_streams_and_retains_exit_code(
    tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes], returncode: int
) -> None:
    log = tmp_path / "command.log"
    command = [
        sys.executable,
        "-u",
        "-c",
        "import os,sys; os.write(1,b'out\\xff'); os.write(2,b'err\\n'); sys.exit(int(sys.argv[1]))",
        str(returncode),
    ]
    result = stream_command(command, log)
    captured = capfdbinary.readouterr()
    assert result.returncode == returncode
    assert log.read_bytes() == captured.out == b"out\xfferr\n"
    assert captured.err == b""


def test_checked_failure_keeps_output(tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]) -> None:
    log = tmp_path / "failed.log"
    with pytest.raises(subprocess.CalledProcessError) as error:
        stream_command(
            [sys.executable, "-c", "import sys; print('failure',file=sys.stderr); sys.exit(4)"], log, check=True
        )
    assert error.value.returncode == 4
    assert log.read_bytes() == capfdbinary.readouterr().out == b"failure\n"


def test_unterminated_output_is_visible_before_child_exit(
    tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    log = tmp_path / "live.log"
    release = tmp_path / "release"
    code = (
        "import sys,time; from pathlib import Path; print('live',end=''); "
        "deadline=time.monotonic()+5\n"
        "while not Path(sys.argv[1]).exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "sys.exit(0 if Path(sys.argv[1]).exists() else 3)"
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(stream_command, [sys.executable, "-c", code, str(release)], log)
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and (not log.exists() or log.read_bytes() != b"live"):
                time.sleep(0.01)
            assert log.read_bytes() == b"live"
            assert not future.done()
            assert capfdbinary.readouterr().out == b"live"
        finally:
            release.touch()
        assert future.result(timeout=5).returncode == 0


@pytest.mark.parametrize(
    "script,sync_status,run_status",
    [
        ("prepare.sh", 0, 0),
        ("prepare.sh", 5, 0),
        ("prepare.sh", 0, 7),
        ("run.sh", 0, 0),
        ("run.sh", 0, 7),
    ],
)
def test_wrappers_tee_output_and_preserve_failures(
    tmp_path: Path, script: str, sync_status: int, run_status: int
) -> None:
    root = tmp_path / "checkout"
    directory = root / "local_scripts/observability"
    directory.mkdir(parents=True)
    shutil.copyfile(Path("local_scripts/observability") / script, directory / script)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "stdout %s\\n" "$1"\nprintf "stderr %s\\n" "$1" >&2\n'
        'if [ "$1" = sync ]; then exit "$XPOOL_TEST_SYNC_STATUS"; fi\n'
        'exit "$XPOOL_TEST_RUN_STATUS"\n'
    )
    uv.chmod(0o700)
    result = subprocess.run(
        ["bash", str(directory / script)],
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "XPOOL_TEST_SYNC_STATUS": str(sync_status),
            "XPOOL_TEST_RUN_STATUS": str(run_status),
        },
        capture_output=True,
        check=False,
    )
    assert result.returncode == (sync_status if script == "prepare.sh" and sync_status else run_status)
    logs = list((root / ".xpool-cache").glob("observer-*.log"))
    assert len(logs) == 1
    content = logs[0].read_bytes()
    assert content and result.stdout.endswith(content)
    assert result.stderr == b""
    assert b"stdout" in content and b"stderr" in content
    if sync_status and script == "prepare.sh":
        assert b"stdout run" not in content
