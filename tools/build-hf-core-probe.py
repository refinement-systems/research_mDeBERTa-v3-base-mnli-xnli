#!/usr/bin/env python3

import argparse
import csv
import pathlib
import sys
from collections import Counter


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "benchmarks/nli/hf-probe-set.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/nli/hf-core-probe.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a smaller deterministic core probe from hf-probe-set.tsv while preserving "
            "finalist disagreements, source diversity, language diversity, and top-drift cases."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Input hard-probe TSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output core-probe TSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=25,
        help="Target number of rows if the required set fits (default: 25)",
    )
    parser.add_argument(
        "--top-drift-per-candidate",
        type=int,
        default=3,
        help="Highest-drift rows to preserve per finalist candidate (default: 3)",
    )
    return parser.parse_args()


def language_for_benchmark(benchmark: str) -> str | None:
    if not benchmark.startswith("xnli-"):
        return None
    parts = benchmark.split("-")
    if len(parts) < 3:
        return None
    return parts[1]


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def has_reason(row: dict[str, str], needle: str) -> bool:
    return needle in (row.get("selection_reasons") or "")


def combined_drift(row: dict[str, str]) -> float:
    return max(
        parse_float(row.get("attention_only_max_abs_logit_delta_vs_hf", "")),
        parse_float(row.get("attention_proj_only_max_abs_logit_delta_vs_hf", "")),
    )


def row_sort_key(row: dict[str, str]) -> tuple[float, str, str]:
    return (-combined_drift(row), row["benchmark"], row["id"])


def add_row(
    selected: dict[tuple[str, str], dict[str, str]],
    reasons: dict[tuple[str, str], list[str]],
    row: dict[str, str],
    reason: str,
) -> None:
    key = (row["benchmark"], row["id"])
    if key not in selected:
        selected[key] = row
        reasons[key] = []
    if reason not in reasons[key]:
        reasons[key].append(reason)


def pick_best(rows: list[dict[str, str]], predicate) -> dict[str, str] | None:
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return None
    return sorted(candidates, key=row_sort_key)[0]


def main() -> int:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not input_path.is_file():
        raise RuntimeError(f"Input TSV not found: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
        fieldnames = list(rows[0].keys()) if rows else []

    if not rows:
        raise RuntimeError(f"No rows found in input TSV: {input_path}")

    selected: dict[tuple[str, str], dict[str, str]] = {}
    core_reasons: dict[tuple[str, str], list[str]] = {}

    finalist_diff_rows = [
        row for row in rows if has_reason(row, "finalists:label_disagreement")
    ]
    for row in sorted(finalist_diff_rows, key=lambda item: (item["benchmark"], item["id"])):
        add_row(selected, core_reasons, row, "finalists:label_disagreement")

    benchmarks = sorted({row["benchmark"] for row in rows})
    for benchmark in benchmarks:
        if any(key[0] == benchmark for key in selected):
            continue
        best = pick_best(rows, lambda row, benchmark=benchmark: row["benchmark"] == benchmark)
        if best:
            add_row(selected, core_reasons, best, "source_coverage")

    languages = sorted(
        {
            language
            for language in (language_for_benchmark(row["benchmark"]) for row in rows)
            if language
        }
    )
    for language in languages:
        if any(language_for_benchmark(key[0]) == language for key in selected):
            continue
        best = pick_best(
            rows,
            lambda row, language=language: language_for_benchmark(row["benchmark"]) == language,
        )
        if best:
            add_row(selected, core_reasons, best, "language_coverage")

    if not any(language_for_benchmark(key[0]) == "zh" for key in selected):
        zh_best = pick_best(rows, lambda row: language_for_benchmark(row["benchmark"]) == "zh")
        if zh_best:
            add_row(selected, core_reasons, zh_best, "zh_coverage")

    for candidate_name in ("attention_only", "attention_proj_only"):
        drift_column = f"{candidate_name}_max_abs_logit_delta_vs_hf"
        top_rows = sorted(
            rows,
            key=lambda row, drift_column=drift_column: (
                -parse_float(row.get(drift_column, "")),
                row["benchmark"],
                row["id"],
            ),
        )[: args.top_drift_per_candidate]
        for rank, row in enumerate(top_rows, start=1):
            add_row(selected, core_reasons, row, f"{candidate_name}:core_top_drift:{rank}")

    remaining = [
        row
        for row in rows
        if (row["benchmark"], row["id"]) not in selected
    ]
    for row in sorted(remaining, key=row_sort_key):
        if len(selected) >= args.target_size:
            break
        add_row(selected, core_reasons, row, "fill_high_drift")

    selected_rows = [
        selected[key] | {"core_selection_reasons": ";".join(core_reasons[key])}
        for key in sorted(selected)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fieldnames = fieldnames + ["core_selection_reasons"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames, delimiter="\t")
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({field: row.get(field, "") for field in output_fieldnames})

    benchmark_counts = Counter(row["benchmark"] for row in selected_rows)
    language_counts = Counter(
        language
        for language in (language_for_benchmark(row["benchmark"]) for row in selected_rows)
        if language
    )

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"target_size: {args.target_size}")
    print(f"rows: {len(selected_rows)}")
    print(
        "finalist_label_differences_kept: "
        f"{sum('finalists:label_disagreement' in row['core_selection_reasons'] for row in selected_rows)}"
    )
    print(f"benchmarks: {dict(sorted(benchmark_counts.items()))}")
    print(f"languages: {dict(sorted(language_counts.items()))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
