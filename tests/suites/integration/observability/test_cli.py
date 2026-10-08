import json
import subprocess
import sys
from pathlib import Path

import pytest
from local_scripts.observability.__main__ import main
from local_scripts.observability.models import RunContext


def test_cli_retains_incomplete_run_and_distinguishes_acceptance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "attempt"
    source.mkdir()
    context = tmp_path / "context.json"
    context.write_text(RunContext(run_id="failed-run", attempt=1, outcome="failed").model_dump_json())
    archive = tmp_path / "archive"
    assert main(["archive", "--source", str(source), "--context", str(context), "--output", str(archive)]) == 0
    derived = tmp_path / "derived"
    assert (
        main(
            [
                "convert",
                "--manifest",
                str(archive / "run_manifest.json"),
                "--output",
                str(derived),
                "--require-representative",
            ]
        )
        == 1
    )
    quality = json.loads((derived / "quality.json").read_text())
    assert quality["incomplete"] and not quality["representative"]
    assert (derived / "timeline.sqlite").exists()
    assert (
        main(["query", "--database", str(derived / "timeline.sqlite"), "--generation", "1" * 32, "--sequence", "1"])
        == 2
    )
    assert "not found" in capsys.readouterr().err


def test_module_entry_reports_bad_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "local_scripts.observability",
            "convert",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "derived"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid archive manifest" in result.stderr
    assert not (tmp_path / "derived").exists()
