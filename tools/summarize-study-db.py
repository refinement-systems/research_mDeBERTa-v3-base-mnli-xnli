#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from replication_cpu_final import (
    SummaryRow,
    compute_frontier_flags,
    fetch_backend_names,
    fetch_candidate_rows,
    fetch_dataset_rows,
    fetch_reference_rows,
    summarize_dataset_backend,
    summarize_study_db_to_prefix,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCRATCHPAD_ROOT = REPO_ROOT / "scratchpad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize stored study evaluations by comparing each artifact against the "
            "backend-specific float reference and computing size-vs-fidelity frontiers."
        )
    )
    parser.add_argument(
        "--scratchpad-root",
        default=str(DEFAULT_SCRATCHPAD_ROOT),
        help=f"Scratchpad root directory (default: {DEFAULT_SCRATCHPAD_ROOT})",
    )
    parser.add_argument(
        "--db-path",
        default="",
        help="Optional path to the study SQLite database. Defaults to <scratchpad>/db.sqlite3.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Dataset name to summarize. Repeat to add more. Defaults to all datasets for the selected roles.",
    )
    parser.add_argument(
        "--role",
        dest="roles",
        action="append",
        default=[],
        choices=["calibration", "fidelity_validation", "fidelity_test", "stress_test", "smoke"],
        help="Dataset role to summarize. Repeat to add more. Defaults to fidelity_validation only.",
    )
    parser.add_argument(
        "--backend",
        dest="backends",
        action="append",
        default=[],
        choices=["cpu", "coreml"],
        help="Backend to summarize. Repeat to add more. Defaults to all backends present in the DB.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional output prefix. Defaults to <scratchpad>/reports/study-summary.",
    )
    return parser.parse_args()


def resolve_output_prefix(args: argparse.Namespace, scratchpad_root: pathlib.Path) -> pathlib.Path:
    if args.output_prefix:
        return pathlib.Path(args.output_prefix)
    return scratchpad_root / "reports" / "study-summary"


def main() -> int:
    args = parse_args()
    scratchpad_root = pathlib.Path(args.scratchpad_root).resolve()
    output_prefix = resolve_output_prefix(args, scratchpad_root)
    summarize_study_db_to_prefix(
        scratchpad_root,
        db_path=pathlib.Path(args.db_path).resolve() if args.db_path else None,
        dataset_names=list(args.datasets),
        roles=list(args.roles),
        backends=list(args.backends),
        output_prefix=output_prefix,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
