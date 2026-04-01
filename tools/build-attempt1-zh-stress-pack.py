#!/usr/bin/env python3

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "benchmarks/nli/hf-finalist-full.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/nli/attempt1-zh-stress-pack.tsv"
PREFERRED_CANDIDATES = [
    "attention_only",
    "attention_proj_only",
    "nncf_fidelity_attention_proj_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact zh-only stress pack from benchmark JSON by preserving "
            "candidate/HF disagreements, candidate/gold errors, finalist label disagreements, "
            "and top logit-drift rows."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Input benchmark JSON with per_example rows (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output TSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        default=[],
        help="Candidate name to include. Repeat to add more.",
    )
    parser.add_argument(
        "--benchmark-substring",
        default="xnli-zh",
        help="Only include benchmarks whose name contains this substring (default: xnli-zh)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=48,
        help="Target number of rows after required selections (default: 48)",
    )
    parser.add_argument(
        "--top-drift-per-candidate",
        type=int,
        default=8,
        help="Highest-drift rows to preserve per candidate (default: 8)",
    )
    return parser.parse_args()


def parse_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def append_reason(reason_map: dict[tuple[str, str], list[str]], key: tuple[str, str], reason: str) -> None:
    if reason not in reason_map[key]:
        reason_map[key].append(reason)


def resolve_candidates(
    requested: list[str],
    per_example: dict[str, list[dict[str, object]]],
) -> list[str]:
    if requested:
        return requested

    available = set(per_example)
    preferred = [candidate for candidate in PREFERRED_CANDIDATES if candidate in available]
    if preferred:
        return preferred

    return sorted(candidate for candidate in per_example if candidate != "float")


def main() -> int:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.is_file():
        raise RuntimeError(f"Input JSON not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    per_example = data.get("per_example", {})
    if not per_example:
        raise RuntimeError(f"No per_example payload found in: {input_path}")

    candidates = resolve_candidates(args.candidates, per_example)
    for candidate in candidates:
        if candidate not in per_example:
            raise RuntimeError(f"Candidate not found in per_example: {candidate}")

    exemplar_source = "float" if "float" in per_example else candidates[0]
    exemplar_rows = {}
    for row in per_example[exemplar_source]:
        benchmark = str(row.get("benchmark", ""))
        if args.benchmark_substring not in benchmark:
            continue
        key = (benchmark, str(row.get("id", "")))
        exemplar_rows[key] = row

    if not exemplar_rows:
        raise RuntimeError(
            f"No rows matched benchmark substring {args.benchmark_substring!r} in {input_path}"
        )

    rows_by_candidate: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
    for candidate in candidates:
        filtered = {}
        for row in per_example[candidate]:
            benchmark = str(row.get("benchmark", ""))
            if args.benchmark_substring not in benchmark:
                continue
            key = (benchmark, str(row.get("id", "")))
            filtered[key] = row
        rows_by_candidate[candidate] = filtered

    float_rows = {}
    if "float" in per_example:
        for row in per_example["float"]:
            benchmark = str(row.get("benchmark", ""))
            if args.benchmark_substring not in benchmark:
                continue
            key = (benchmark, str(row.get("id", "")))
            float_rows[key] = row

    for candidate, rows in rows_by_candidate.items():
        missing = sorted(set(exemplar_rows) - set(rows))
        if missing:
            raise RuntimeError(
                f"Candidate {candidate} is missing {len(missing)} filtered rows; first missing key: {missing[0]}"
            )

    reason_map: dict[tuple[str, str], list[str]] = defaultdict(list)

    def combined_drift(key: tuple[str, str]) -> float:
        drifts = [
            parse_float(rows_by_candidate[candidate][key].get("max_abs_logit_delta_vs_hf"))
            for candidate in candidates
        ]
        if key in float_rows:
            drifts.append(parse_float(float_rows[key].get("max_abs_logit_delta_vs_hf")))
        return max(drifts, default=0.0)

    for candidate in candidates:
        rows = list(rows_by_candidate[candidate].values())

        for row in rows:
            key = (str(row["benchmark"]), str(row["id"]))
            if row.get("model_label") != row.get("hf_label"):
                append_reason(reason_map, key, f"{candidate}:hf_disagreement")
            if row.get("model_label") != row.get("gold_label"):
                append_reason(reason_map, key, f"{candidate}:gold_error")

        top_drift = sorted(
            rows,
            key=lambda row: (
                -parse_float(row.get("max_abs_logit_delta_vs_hf")),
                str(row.get("benchmark", "")),
                str(row.get("id", "")),
            ),
        )[: args.top_drift_per_candidate]
        for rank, row in enumerate(top_drift, start=1):
            key = (str(row["benchmark"]), str(row["id"]))
            append_reason(reason_map, key, f"{candidate}:top_drift:{rank}")

    if len(candidates) >= 2:
        base_rows = rows_by_candidate[candidates[0]]
        other_maps = [rows_by_candidate[candidate] for candidate in candidates[1:]]
        for key, base_row in base_rows.items():
            if any(other_rows[key].get("model_label") != base_row.get("model_label") for other_rows in other_maps):
                append_reason(reason_map, key, "finalists:label_disagreement")

    selected = set(reason_map)
    remaining_keys = [key for key in exemplar_rows if key not in selected]
    for key in sorted(remaining_keys, key=lambda item: (-combined_drift(item), item[0], item[1])):
        if len(selected) >= args.target_size:
            break
        append_reason(reason_map, key, "fill_high_drift")
        selected.add(key)

    candidate_columns = []
    for candidate in candidates:
        candidate_columns.extend(
            [
                f"{candidate}_label",
                f"{candidate}_max_abs_logit_delta_vs_hf",
                f"{candidate}_mean_abs_logit_delta_vs_hf",
            ]
        )

    fieldnames = [
        "benchmark",
        "id",
        "label",
        "premise",
        "hypothesis",
        "selection_reasons",
        "hf_label",
        "float_label",
        "float_max_abs_logit_delta_vs_hf",
        "combined_max_abs_logit_delta_vs_hf",
    ] + candidate_columns

    ordered_keys = sorted(selected, key=lambda key: (-combined_drift(key), key[0], key[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for key in ordered_keys:
            exemplar = exemplar_rows[key]
            row = {
                "benchmark": exemplar["benchmark"],
                "id": exemplar["id"],
                "label": exemplar["gold_label"],
                "premise": exemplar["premise"],
                "hypothesis": exemplar["hypothesis"],
                "selection_reasons": ";".join(reason_map[key]),
                "hf_label": exemplar["hf_label"],
                "float_label": float_rows.get(key, {}).get("model_label", exemplar.get("float_label", "")),
                "float_max_abs_logit_delta_vs_hf": (
                    f"{parse_float(float_rows.get(key, {}).get('max_abs_logit_delta_vs_hf')):.6f}"
                    if key in float_rows
                    else ""
                ),
                "combined_max_abs_logit_delta_vs_hf": f"{combined_drift(key):.6f}",
            }
            for candidate in candidates:
                candidate_row = rows_by_candidate[candidate][key]
                row[f"{candidate}_label"] = candidate_row.get("model_label", "")
                row[f"{candidate}_max_abs_logit_delta_vs_hf"] = (
                    f"{parse_float(candidate_row.get('max_abs_logit_delta_vs_hf')):.6f}"
                )
                row[f"{candidate}_mean_abs_logit_delta_vs_hf"] = (
                    f"{parse_float(candidate_row.get('mean_abs_logit_delta_vs_hf')):.6f}"
                )
            writer.writerow(row)

    benchmark_counts = Counter(key[0] for key in ordered_keys)
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"benchmark_substring: {args.benchmark_substring}")
    print(f"candidates: {', '.join(candidates)}")
    print(f"rows: {len(ordered_keys)}")
    print(f"benchmarks: {dict(sorted(benchmark_counts.items()))}")
    for candidate in candidates:
        rows = rows_by_candidate[candidate]
        hf_disagreements = sum(1 for row in rows.values() if row.get("model_label") != row.get("hf_label"))
        gold_errors = sum(1 for row in rows.values() if row.get("model_label") != row.get("gold_label"))
        print(f"{candidate}_hf_disagreements: {hf_disagreements}")
        print(f"{candidate}_gold_errors: {gold_errors}")
    finalist_label_diffs = sum(
        1 for reasons in reason_map.values() if "finalists:label_disagreement" in reasons
    )
    print(f"finalist_label_differences: {finalist_label_diffs}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
