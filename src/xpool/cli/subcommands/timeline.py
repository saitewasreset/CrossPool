"""Artifact-only independent Timeline verification and Perfetto export."""

from __future__ import annotations

import argparse
from pathlib import Path

from xpool.cli.command import CliCommandGroup, OfflineCliCommand
from xpool.devkit.timeline.offline import export, verify


class TimelineCommand(CliCommandGroup):
    """Independent raw Timeline tools."""

    name = "timeline"
    help = "verify and export raw Timeline Sessions"
    subparser_dest = "timeline_command"


class TimelineVerifyCommand(OfflineCliCommand):
    """Read-only structural validation and declared-coverage assessment."""

    name = "verify"
    parent = "timeline"
    help = "verify a raw Timeline Session"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--session", required=True, type=Path, help="raw Session directory")

    def run(self, args: argparse.Namespace) -> int:
        report = verify(args.session)
        print(
            f"timeline {'complete' if report.complete else 'degraded'} chunks={report.chunks} records={report.records}"
        )
        for issue in report.issues:
            print(f"reason={issue.reason} producer={issue.producer_id} sequence={issue.sequence} detail={issue.detail}")
        return 0 if report.complete else 1


class TimelineExportCommand(OfflineCliCommand):
    """Explicit separate-domain Perfetto projection, independent of runtime credit."""

    name = "export"
    parent = "timeline"
    help = "export each clock domain separately for Perfetto"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--session", required=True, type=Path, help="raw Session directory")
        parser.add_argument("--output", required=True, type=Path, help="new directory outside the Session")

    def run(self, args: argparse.Namespace) -> int:
        report = export(args.session, args.output)
        print(f"timeline {'complete' if report.complete else 'degraded'} output={args.output}")
        return 0 if report.complete else 1
