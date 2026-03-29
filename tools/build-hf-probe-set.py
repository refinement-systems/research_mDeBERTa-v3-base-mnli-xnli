#!/usr/bin/env python3

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "benchmarks/nli/hf-finalist-full.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/nli/hf-probe-set.tsv"
DEFAULT_CANDIDATES = ["attention_only", "attention_proj_only"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed probe TSV from the HF-vs-ONNX finalist benchmark by taking "
            "HF disagreements, highest-drift examples, and finalist label disagreements."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Input benchmark JSON (default: {DEFAULT_INPUT})",
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
        help="Candidate name to include. Repeat to add more. Defaults to attention_only and attention_proj_only.",
    )
    parser.add_argument(
        "--top-drift",
        type=int,
        default=20,
        help="Top drift examples to take per candidate (default: 20)",
    )
    parser.add_argument(
        "--include-finalist-label-diffs",
        action="store_true",
        help="Include examples where the selected finalists predict different labels.",
    )
    parser.add_argument(
        "--include-float-top-drift",
        type=int,
        default=0,
        help="Optional number of float top-drift examples to include (default: 0)",
    )
    return parser.parse_args()


def append_reason(reason_map: dict[tuple[str, str], list[str]], key: tuple[str, str], reason: str) -> None:
    if reason not in reason_map[key]:
        reason_map[key].append(reason)


def main() -> int:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not input_path.is_file():
        raise RuntimeError(f"Input JSON not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    per_example = data.get("per_example", {})
    candidates = args.candidates or DEFAULT_CANDIDATES
    for candidate in candidates:
        if candidate not in per_example:
            raise RuntimeError(f"Candidate not found in per_example: {candidate}")

    rows_by_candidate = {
        candidate: {
            (row["benchmark"], row["id"]): row
            for row in per_example[candidate]
        }
        for candidate in candidates
    }

    exemplar_rows = {
        (row["benchmark"], row["id"]): row
        for row in next(iter(per_example.values()))
    }

    reason_map: dict[tuple[str, str], list[str]] = defaultdict(list)

    for candidate in candidates:
        rows = list(rows_by_candidate[candidate].values())
        disagreements = [row for row in rows if row["model_label"] != row["hf_label"]]
        for row in disagreements:
            append_reason(
                reason_map,
                (row["benchmark"], row["id"]),
                f"{candidate}:hf_disagreement",
            )

        top_drift = sorted(
            rows,
            key=lambda row: row["max_abs_logit_delta_vs_hf"],
            reverse=True,
        )[: args.top_drift]
        for rank, row in enumerate(top_drift, start=1):
            append_reason(
                reason_map,
                (row["benchmark"], row["id"]),
                f"{candidate}:top_drift:{rank}",
            )

    if args.include_finalist_label_diffs and len(candidates) >= 2:
        base_rows = rows_by_candidate[candidates[0]]
        other_row_maps = [rows_by_candidate[candidate] for candidate in candidates[1:]]
        for key, base_row in base_rows.items():
            if any(other_rows[key]["model_label"] != base_row["model_label"] for other_rows in other_row_maps):
                append_reason(reason_map, key, "finalists:label_disagreement")

    if args.include_float_top_drift > 0 and "float" in per_example:
        float_top = sorted(
            per_example["float"],
            key=lambda row: row["max_abs_logit_delta_vs_hf"],
            reverse=True,
        )[: args.include_float_top_drift]
        for rank, row in enumerate(float_top, start=1):
            append_reason(
                reason_map,
                (row["benchmark"], row["id"]),
                f"float:top_drift:{rank}",
            )

    selected_keys = sorted(reason_map.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "benchmark",
        "id",
        "label",
        "premise",
        "hypothesis",
        "selection_reasons",
        "hf_label",
        "float_label",
        "attention_only_label",
        "attention_proj_only_label",
        "attention_only_max_abs_logit_delta_vs_hf",
        "attention_proj_only_max_abs_logit_delta_vs_hf",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for key in selected_keys:
            exemplar = exemplar_rows[key]
            attention_only = rows_by_candidate.get("attention_only", {}).get(key)
            attention_proj_only = rows_by_candidate.get("attention_proj_only", {}).get(key)
            writer.writerow(
                {
                    "benchmark": exemplar["benchmark"],
                    "id": exemplar["id"],
                    "label": exemplar["gold_label"],
                    "premise": exemplar["premise"],
                    "hypothesis": exemplar["hypothesis"],
                    "selection_reasons": ";".join(reason_map[key]),
                    "hf_label": exemplar["hf_label"],
                    "float_label": exemplar["float_label"],
                    "attention_only_label": attention_only["model_label"] if attention_only else "",
                    "attention_proj_only_label": attention_proj_only["model_label"] if attention_proj_only else "",
                    "attention_only_max_abs_logit_delta_vs_hf": (
                        f"{attention_only['max_abs_logit_delta_vs_hf']:.6f}" if attention_only else ""
                    ),
                    "attention_proj_only_max_abs_logit_delta_vs_hf": (
                        f"{attention_proj_only['max_abs_logit_delta_vs_hf']:.6f}"
                        if attention_proj_only
                        else ""
                    ),
                }
            )

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"candidates: {', '.join(candidates)}")
    print(f"rows: {len(selected_keys)}")
    for candidate in candidates:
        candidate_rows = rows_by_candidate[candidate]
        disagreement_count = sum(
            1 for row in candidate_rows.values() if row["model_label"] != row["hf_label"]
        )
        print(f"{candidate}_hf_disagreements: {disagreement_count}")
    if args.include_finalist_label_diffs and len(candidates) >= 2:
        finalist_diff_count = sum(
            1 for reasons in reason_map.values() if "finalists:label_disagreement" in reasons
        )
        print(f"finalist_label_differences: {finalist_diff_count}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
