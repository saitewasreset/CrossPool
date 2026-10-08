"""Explicit-path commands for the disposable Observer prototype."""

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from local_scripts.observability.archive import archive_attempt
from local_scripts.observability.models import RunContext
from local_scripts.observability.timeline import convert, query


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CrossPool offline Observer prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    archive_parser = commands.add_parser("archive")
    archive_parser.add_argument("--source", type=Path, required=True)
    archive_parser.add_argument("--context", type=Path, required=True)
    archive_parser.add_argument("--output", type=Path, required=True)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("--manifest", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--require-representative", action="store_true")
    query_parser = commands.add_parser("query")
    query_parser.add_argument("--database", type=Path, required=True)
    query_parser.add_argument("--generation", required=True)
    query_parser.add_argument("--instance", type=int, default=0)
    query_parser.add_argument("--sequence", type=int, required=True)
    options = parser.parse_args(arguments)
    try:
        match options.command:
            case "archive":
                context = RunContext.model_validate_json(options.context.read_bytes())
                print(archive_attempt(options.source, options.output, context))
            case "convert":
                report = convert(options.manifest, options.output)
                print(report.model_dump_json(indent=2))
                if options.require_representative and not report.representative:
                    return 1
            case "query":
                print(
                    json.dumps(
                        query(options.database, options.generation, options.instance, options.sequence), indent=2
                    )
                )
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"xpool observer prototype {options.command} failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
