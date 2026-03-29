#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that multiple NLI TSV slices do not overlap by dataset/config/split/row_idx "
            "or, if unavailable, by id."
        )
    )
    parser.add_argument(
        "--tsv",
        dest="tsv_paths",
        action="append",
        default=[],
        help="TSV file to include in the overlap check. Repeat to add more.",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Optional directory to glob for TSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.tsv",
        help="Glob pattern used with --input-dir (default: *.tsv).",
    )
    parser.add_argument(
        "--show-overlaps",
        type=int,
        default=5,
        help="Maximum overlapping row keys to print per file pair (default: 5).",
    )
    return parser.parse_args()


def discover_paths(args: argparse.Namespace) -> list[pathlib.Path]:
    paths = [pathlib.Path(path) for path in args.tsv_paths]
    if args.input_dir:
        paths.extend(sorted(pathlib.Path(args.input_dir).glob(args.pattern)))
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    if len(unique_paths) < 2:
        raise RuntimeError("Provide at least two TSV files via --tsv or --input-dir")
    return unique_paths


def row_identity(row: dict[str, str]) -> tuple[str, ...]:
    dataset = (row.get("dataset") or "").strip()
    config = (row.get("config") or "").strip()
    split = (row.get("split") or "").strip()
    row_idx = (row.get("row_idx") or "").strip()
    if dataset and config and split and row_idx:
        return ("dataset", dataset, config, split, row_idx)

    example_id = (row.get("id") or "").strip()
    if example_id:
        return ("id", example_id)

    return (
        "text",
        (row.get("label") or "").strip(),
        (row.get("premise") or "").strip(),
        (row.get("hypothesis") or "").strip(),
    )


def read_identities(path: pathlib.Path) -> set[tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"premise", "hypothesis"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"TSV must include premise and hypothesis columns: {path}")
        return {row_identity(row) for row in reader}


def main() -> int:
    args = parse_args()
    paths = discover_paths(args)
    identities_by_path = {path: read_identities(path) for path in paths}

    found_overlap = False
    for left_path, right_path in itertools.combinations(paths, 2):
        overlap = identities_by_path[left_path] & identities_by_path[right_path]
        if not overlap:
            print(f"ok: {left_path} vs {right_path} -> disjoint")
            continue

        found_overlap = True
        print(
            f"overlap: {left_path} vs {right_path} -> {len(overlap)} shared rows",
            file=sys.stderr,
        )
        for item in sorted(overlap)[: args.show_overlaps]:
            print(f"  {item}", file=sys.stderr)

    return 1 if found_overlap else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
