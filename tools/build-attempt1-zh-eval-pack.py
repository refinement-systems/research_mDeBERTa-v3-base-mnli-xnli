#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/nli/attempt1-zh-sensitive-validation.tsv"
DEFAULT_SOURCE_TSVS = [
    REPO_ROOT / "benchmarks/nli/xnli-zh-validation-search-validation-skip32-32-per-label.tsv",
]
DEFAULT_FINAL_GATES = [
    REPO_ROOT / "benchmarks/nli/mnli-validation_matched-200-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/mnli-validation_mismatched-200-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-de-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-en-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-es-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-fr-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-zh-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/hf-probe-set.tsv",
    REPO_ROOT / "benchmarks/nli/hf-core-probe.tsv",
]
OUTPUT_FIELDS = [
    "benchmark",
    "id",
    "label",
    "premise",
    "hypothesis",
    "dataset",
    "config",
    "split",
    "row_idx",
    "selection_reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a held-out Chinese-sensitive evaluation/validation TSV for attempt1 "
            "from one or more local zh slices, and verify it does not overlap with the final gates."
        )
    )
    parser.add_argument(
        "--source-tsv",
        dest="source_tsvs",
        action="append",
        default=[],
        help="Input TSV to include in the pack. Repeat to add more. Defaults to the zh search-validation slice.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output TSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--benchmark-name",
        default="attempt1-zh-sensitive-validation.tsv",
        help="Benchmark name stored in the output TSV (default: attempt1-zh-sensitive-validation.tsv).",
    )
    parser.add_argument(
        "--selection-reason",
        default="zh_sensitive_validation",
        help="selection_reasons tag stored in generated rows (default: zh_sensitive_validation).",
    )
    parser.add_argument(
        "--skip-disjointness-check",
        action="store_true",
        help="Skip overlap checks against the final gate TSVs.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"premise", "hypothesis"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"TSV must include premise and hypothesis columns: {path}")
        return list(reader)


def read_identities(path: pathlib.Path) -> set[tuple[str, ...]]:
    return {row_identity(row) for row in read_rows(path)}


def build_pack_rows(
    source_paths: list[pathlib.Path],
    benchmark_name: str,
    selection_reason: str,
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()

    for path in source_paths:
        for row in read_rows(path):
            identity = row_identity(row)
            if identity in seen:
                continue
            seen.add(identity)
            output_rows.append(
                {
                    "benchmark": benchmark_name,
                    "id": (row.get("id") or "").strip(),
                    "label": (row.get("label") or row.get("gold_label") or "").strip(),
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "dataset": (row.get("dataset") or "").strip(),
                    "config": (row.get("config") or "").strip(),
                    "split": (row.get("split") or "").strip(),
                    "row_idx": (row.get("row_idx") or "").strip(),
                    "selection_reasons": selection_reason,
                }
            )

    if not output_rows:
        raise RuntimeError("No rows were loaded for the zh-sensitive evaluation pack")
    return output_rows


def verify_disjoint(output_rows: list[dict[str, str]], final_gate_paths: list[pathlib.Path]) -> None:
    output_identities = {row_identity(row) for row in output_rows}
    for path in final_gate_paths:
        if not path.is_file():
            raise RuntimeError(f"Final gate TSV not found: {path}")
        overlap = output_identities & read_identities(path)
        if overlap:
            sample = ", ".join(repr(item) for item in sorted(overlap)[:5])
            raise RuntimeError(f"Overlap detected with {path}: {sample}")


def write_rows(path: pathlib.Path, rows: list[dict[str, str]], force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError(f"Output already exists: {path}. Re-run with --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    source_paths = [pathlib.Path(path) for path in (args.source_tsvs or DEFAULT_SOURCE_TSVS)]
    for path in source_paths:
        if not path.is_file():
            raise RuntimeError(f"Source TSV not found: {path}")

    rows = build_pack_rows(
        source_paths=source_paths,
        benchmark_name=args.benchmark_name,
        selection_reason=args.selection_reason,
    )
    if not args.skip_disjointness_check:
        verify_disjoint(rows, DEFAULT_FINAL_GATES)

    output_path = pathlib.Path(args.output)
    write_rows(output_path, rows, force=args.force)

    print(f"generated: {output_path}")
    print(f"  rows: {len(rows)}")
    print(f"  sources: {[str(path) for path in source_paths]}")
    print(f"  benchmark_name: {args.benchmark_name}")
    print(f"  selection_reason: {args.selection_reason}")
    print(f"  disjoint_from_final_gates: {not args.skip_disjointness_check}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
