#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the attempt4 CPU deployment-study report by joining study summaries, "
            "development gates, and CPU runtime/RSS benchmarks."
        )
    )
    parser.add_argument("--datasets-manifest", required=True)
    parser.add_argument("--validation-summary-json", required=True)
    parser.add_argument("--validation-runtime-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--test-summary-json", default="")
    parser.add_argument("--stress-summary-json", default="")
    parser.add_argument("--cold-benchmark-csv", default="")
    parser.add_argument("--report-markdown", default="")
    return parser.parse_args()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Summary JSON does not contain a row list: {path}")
    return [dict(row) for row in rows]


def read_benchmark_rows(path: pathlib.Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["candidate"]: row for row in reader}


def optional_float(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def percent_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def ms_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f} ms"


def mib_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / (1024.0 * 1024.0):.1f} MiB"


def lower_name(value: str) -> str:
    return value.lower()


def is_mnli_dataset(name: str) -> bool:
    return lower_name(name).startswith("mnli-")


def is_anli_dev_dataset(name: str) -> bool:
    lowered = lower_name(name)
    return lowered.startswith("anli-") and "-dev-" in lowered


def is_anli_test_dataset(name: str) -> bool:
    lowered = lower_name(name)
    return lowered.startswith("anli-") and "-test-" in lowered


def is_xnli_dataset(name: str) -> bool:
    return lower_name(name).startswith("xnli-")


def xnli_language(name: str) -> str | None:
    if not is_xnli_dataset(name):
        return None
    parts = name.split("-")
    if len(parts) < 2:
        return None
    return parts[1]


def aggregate_required_rows(
    rows: list[dict[str, Any]],
    required_datasets: list[str],
) -> dict[str, dict[str, Any]]:
    required_dataset_set = set(required_datasets)
    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in required_dataset_set:
            continue
        quantization = str(row["quantization"])
        item = grouped.setdefault(
            quantization,
            {
                "quantization": quantization,
                "artifact_path": row["artifact_path"],
                "size_bytes": int(row["size_bytes"]),
                "smooth_quant_disabled": row.get("smooth_quant_disabled"),
                "retry_reason": row.get("retry_reason", ""),
                "datasets": {},
                "example_count": 0,
                "labeled_example_count": 0,
                "correct_prediction_count": 0,
                "disagreement_count": 0,
                "max_abs_logit_delta": 0.0,
                "_weighted_mean_sum": 0.0,
            },
        )
        item["datasets"][dataset] = row
        item["example_count"] += int(row["example_count"])
        item["labeled_example_count"] += int(row.get("labeled_example_count", 0))
        item["correct_prediction_count"] += int(row.get("correct_prediction_count", 0))
        item["disagreement_count"] += int(row["disagreement_count"])
        item["max_abs_logit_delta"] = max(
            float(item["max_abs_logit_delta"]),
            float(row["max_abs_logit_delta"]),
        )
        item["_weighted_mean_sum"] += float(row["mean_abs_logit_delta"]) * int(row["example_count"])

    aggregated: dict[str, dict[str, Any]] = {}
    for quantization, item in grouped.items():
        present_dataset_set = set(item["datasets"].keys())
        complete = present_dataset_set == required_dataset_set
        example_count = int(item["example_count"])
        labeled_example_count = int(item["labeled_example_count"])
        aggregated[quantization] = {
            "quantization": quantization,
            "artifact_path": item["artifact_path"],
            "size_bytes": int(item["size_bytes"]),
            "smooth_quant_disabled": item["smooth_quant_disabled"],
            "retry_reason": item["retry_reason"],
            "complete": complete,
            "dataset_names": sorted(present_dataset_set),
            "datasets": item["datasets"],
            "example_count": example_count,
            "labeled_example_count": labeled_example_count,
            "correct_prediction_count": int(item["correct_prediction_count"]),
            "gold_accuracy": (
                float(item["correct_prediction_count"]) / float(labeled_example_count)
                if labeled_example_count
                else None
            ),
            "float_label_agreement": (
                float(example_count - int(item["disagreement_count"])) / float(example_count)
                if example_count
                else None
            ),
            "mean_abs_logit_delta": (
                float(item["_weighted_mean_sum"]) / float(example_count)
                if example_count
                else None
            ),
            "max_abs_logit_delta": float(item["max_abs_logit_delta"]),
            "disagreement_count": int(item["disagreement_count"]),
        }
    return aggregated


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def validation_gate(
    quantization: str,
    aggregated_validation: dict[str, dict[str, Any]],
    runtime_rows: dict[str, dict[str, str]],
) -> tuple[bool, list[str], dict[str, float | None]]:
    candidate = aggregated_validation[quantization]
    reference = aggregated_validation.get("reference")
    reasons: list[str] = []
    metrics: dict[str, float | None] = {
        "mnli_macro_accuracy": None,
        "mnli_macro_accuracy_drop": None,
        "anli_macro_accuracy": None,
        "anli_macro_accuracy_drop": None,
        "validation_float_label_agreement": candidate["float_label_agreement"],
        "peak_rss_ratio_vs_reference": None,
    }

    if not candidate["complete"]:
        reasons.append("missing validation datasets")
        return False, reasons, metrics
    if reference is None or not reference["complete"]:
        reasons.append("reference validation summary is incomplete")
        return False, reasons, metrics

    candidate_runtime = runtime_rows.get(quantization)
    reference_runtime = runtime_rows.get("reference")
    if candidate_runtime is None:
        reasons.append("missing CPU persistent benchmark row")
    if reference_runtime is None:
        reasons.append("missing reference CPU persistent benchmark row")

    candidate_dataset_rows = candidate["datasets"]
    reference_dataset_rows = reference["datasets"]

    candidate_mnli_values = [
        float(row["gold_accuracy"])
        for dataset_name, row in candidate_dataset_rows.items()
        if is_mnli_dataset(dataset_name) and row.get("gold_accuracy") is not None
    ]
    reference_mnli_values = [
        float(row["gold_accuracy"])
        for dataset_name, row in reference_dataset_rows.items()
        if is_mnli_dataset(dataset_name) and row.get("gold_accuracy") is not None
    ]
    candidate_anli_values = [
        float(row["gold_accuracy"])
        for dataset_name, row in candidate_dataset_rows.items()
        if is_anli_dev_dataset(dataset_name) and row.get("gold_accuracy") is not None
    ]
    reference_anli_values = [
        float(row["gold_accuracy"])
        for dataset_name, row in reference_dataset_rows.items()
        if is_anli_dev_dataset(dataset_name) and row.get("gold_accuracy") is not None
    ]

    candidate_mnli_macro = average(candidate_mnli_values)
    reference_mnli_macro = average(reference_mnli_values)
    candidate_anli_macro = average(candidate_anli_values)
    reference_anli_macro = average(reference_anli_values)

    metrics["mnli_macro_accuracy"] = candidate_mnli_macro
    metrics["anli_macro_accuracy"] = candidate_anli_macro

    if candidate_mnli_macro is None or reference_mnli_macro is None:
        reasons.append("missing MNLI development accuracy")
    else:
        metrics["mnli_macro_accuracy_drop"] = reference_mnli_macro - candidate_mnli_macro
        if reference_mnli_macro - candidate_mnli_macro > 0.005:
            reasons.append("MNLI macro accuracy drop exceeds 0.5 points")

    if candidate_anli_macro is None or reference_anli_macro is None:
        reasons.append("missing ANLI development accuracy")
    else:
        metrics["anli_macro_accuracy_drop"] = reference_anli_macro - candidate_anli_macro
        if reference_anli_macro - candidate_anli_macro > 0.010:
            reasons.append("ANLI macro accuracy drop exceeds 1.0 point")

    for dataset_name, row in candidate_dataset_rows.items():
        if not (is_mnli_dataset(dataset_name) or is_anli_dev_dataset(dataset_name)):
            continue
        reference_row = reference_dataset_rows.get(dataset_name)
        if reference_row is None:
            reasons.append(f"missing reference row for {dataset_name}")
            continue
        candidate_accuracy = row.get("gold_accuracy")
        reference_accuracy = reference_row.get("gold_accuracy")
        if candidate_accuracy is None or reference_accuracy is None:
            reasons.append(f"missing gold accuracy for {dataset_name}")
            continue
        if float(reference_accuracy) - float(candidate_accuracy) > 0.015:
            reasons.append(f"{dataset_name} accuracy drop exceeds 1.5 points")

    if candidate["float_label_agreement"] is None or float(candidate["float_label_agreement"]) < 0.98:
        reasons.append("aggregate float-label agreement is below 98.0%")

    if candidate_runtime is not None and reference_runtime is not None:
        candidate_peak_rss = optional_float(candidate_runtime, "peak_rss_after_timed_runs_median_bytes")
        reference_peak_rss = optional_float(reference_runtime, "peak_rss_after_timed_runs_median_bytes")
        if candidate_peak_rss is None or reference_peak_rss is None:
            reasons.append("missing peak RSS for CPU persistent benchmark")
        else:
            metrics["peak_rss_ratio_vs_reference"] = (
                candidate_peak_rss / reference_peak_rss if reference_peak_rss else None
            )
            if reference_peak_rss and candidate_peak_rss > reference_peak_rss * 1.25:
                reasons.append("peak RSS exceeds reference by more than 25%")

    return not reasons, reasons, metrics


def dominates_final(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["quantization"] == right["quantization"]:
        return False

    left_values = (
        left.get("size_bytes"),
        left.get("cpu_persistent_warm_median_ms"),
        left.get("cpu_persistent_resident_after_warmup_bytes"),
    )
    right_values = (
        right.get("size_bytes"),
        right.get("cpu_persistent_warm_median_ms"),
        right.get("cpu_persistent_resident_after_warmup_bytes"),
    )
    if any(value is None for value in left_values + right_values):
        return False
    if any(float(left_value) > float(right_value) for left_value, right_value in zip(left_values, right_values)):
        return False
    return any(float(left_value) < float(right_value) for left_value, right_value in zip(left_values, right_values))


def choose_recommendation(frontier_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not frontier_rows:
        return None
    chosen = min(
        frontier_rows,
        key=lambda row: (
            float(row["size_bytes"]),
            float(row.get("cpu_persistent_warm_median_ms") or float("inf")),
            float(row.get("cpu_persistent_resident_after_warmup_bytes") or float("inf")),
            -float(row.get("final_test_gold_accuracy") or 0.0),
            row["quantization"],
        ),
    )
    return {
        "quantization": chosen["quantization"],
        "artifact_path": chosen["artifact_path"],
    }


def build_candidate_rows(
    validation_aggregates: dict[str, dict[str, Any]],
    test_aggregates: dict[str, dict[str, Any]],
    stress_aggregates: dict[str, dict[str, Any]],
    runtime_rows: dict[str, dict[str, str]],
    cold_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    candidate_names = sorted(
        set(validation_aggregates) | set(test_aggregates) | set(stress_aggregates) | set(runtime_rows) | set(cold_rows)
    )
    rows: list[dict[str, Any]] = []
    for candidate_name in candidate_names:
        validation = validation_aggregates.get(candidate_name, {})
        test = test_aggregates.get(candidate_name, {})
        stress = stress_aggregates.get(candidate_name, {})
        runtime = runtime_rows.get(candidate_name, {})
        cold = cold_rows.get(candidate_name, {})
        rows.append(
            {
                "quantization": candidate_name,
                "artifact_path": validation.get("artifact_path", test.get("artifact_path", "")),
                "size_bytes": validation.get("size_bytes", test.get("size_bytes")),
                "smooth_quant_disabled": validation.get("smooth_quant_disabled"),
                "retry_reason": validation.get("retry_reason", ""),
                "validation_complete": validation.get("complete", False),
                "validation_gold_accuracy": validation.get("gold_accuracy"),
                "validation_float_label_agreement": validation.get("float_label_agreement"),
                "final_test_complete": test.get("complete", False),
                "final_test_gold_accuracy": test.get("gold_accuracy"),
                "final_test_float_label_agreement": test.get("float_label_agreement"),
                "stress_complete": stress.get("complete", False),
                "stress_gold_accuracy": stress.get("gold_accuracy"),
                "stress_float_label_agreement": stress.get("float_label_agreement"),
                "cpu_persistent_load_median_ms": optional_float(runtime, "load_median_ms"),
                "cpu_persistent_warm_median_ms": optional_float(runtime, "warm_median_ms"),
                "cpu_persistent_resident_after_warmup_bytes": optional_float(
                    runtime, "resident_after_warmup_median_bytes"
                ),
                "cpu_persistent_peak_rss_bytes": optional_float(
                    runtime, "peak_rss_after_timed_runs_median_bytes"
                ),
                "cpu_cold_load_median_ms": optional_float(cold, "load_median_ms"),
                "cpu_cold_warm_median_ms": optional_float(cold, "warm_median_ms"),
            }
        )
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def relative_link(target: pathlib.Path, source_dir: pathlib.Path) -> str:
    return os.path.relpath(target, start=source_dir)


def repo_relative_label(target: pathlib.Path) -> str:
    try:
        return target.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return target.name


def markdown_link(target: pathlib.Path, report_dir: pathlib.Path) -> str:
    return f"[{repo_relative_label(target)}]({relative_link(target, report_dir)})"


def main() -> int:
    args = parse_args()
    datasets_manifest_path = pathlib.Path(args.datasets_manifest).resolve()
    validation_summary_path = pathlib.Path(args.validation_summary_json).resolve()
    validation_runtime_path = pathlib.Path(args.validation_runtime_csv).resolve()
    output_prefix = pathlib.Path(args.output_prefix).resolve()
    report_markdown_path = (
        pathlib.Path(args.report_markdown).resolve()
        if args.report_markdown
        else output_prefix.with_suffix(".md")
    )
    test_summary_path = pathlib.Path(args.test_summary_json).resolve() if args.test_summary_json else None
    stress_summary_path = pathlib.Path(args.stress_summary_json).resolve() if args.stress_summary_json else None
    cold_benchmark_path = pathlib.Path(args.cold_benchmark_csv).resolve() if args.cold_benchmark_csv else None

    dataset_manifest = read_json(datasets_manifest_path)
    validation_summary_rows = read_summary_rows(validation_summary_path)
    test_summary_rows = read_summary_rows(test_summary_path) if test_summary_path else []
    stress_summary_rows = read_summary_rows(stress_summary_path) if stress_summary_path else []
    validation_runtime_rows = read_benchmark_rows(validation_runtime_path)
    cold_rows = read_benchmark_rows(cold_benchmark_path) if cold_benchmark_path else {}

    validation_datasets = list(dataset_manifest["validation_datasets"])
    test_datasets = list(dataset_manifest["test_datasets"])
    stress_datasets = list(dataset_manifest["stress_datasets"])

    validation_aggregates = aggregate_required_rows(validation_summary_rows, validation_datasets)
    test_aggregates = aggregate_required_rows(test_summary_rows, test_datasets) if test_summary_rows else {}
    stress_aggregates = (
        aggregate_required_rows(stress_summary_rows, stress_datasets) if stress_summary_rows else {}
    )

    candidate_rows = build_candidate_rows(
        validation_aggregates,
        test_aggregates,
        stress_aggregates,
        validation_runtime_rows,
        cold_rows,
    )
    candidate_rows_by_name = {row["quantization"]: row for row in candidate_rows}

    locked_quantizations: list[str] = []
    for candidate_name in sorted(validation_aggregates):
        gate_pass, reasons, gate_metrics = validation_gate(
            candidate_name,
            validation_aggregates,
            validation_runtime_rows,
        )
        row = candidate_rows_by_name.setdefault(
            candidate_name,
            {
                "quantization": candidate_name,
                "artifact_path": validation_aggregates[candidate_name]["artifact_path"],
                "size_bytes": validation_aggregates[candidate_name]["size_bytes"],
            },
        )
        row["validation_gate_pass"] = gate_pass
        row["validation_gate_reasons"] = reasons
        row["validation_gate_reason_text"] = "; ".join(reasons)
        row.update(gate_metrics)
        if candidate_name == "reference" or gate_pass:
            locked_quantizations.append(candidate_name)

    final_frontier: list[dict[str, Any]] = []
    recommendation = None
    if test_aggregates:
        locked_rows = []
        for candidate_name in locked_quantizations:
            candidate_row = candidate_rows_by_name[candidate_name]
            if not candidate_row.get("final_test_complete"):
                continue
            locked_rows.append(candidate_row)
        for row in locked_rows:
            row["final_frontier"] = not any(dominates_final(other, row) for other in locked_rows)
            if row["final_frontier"]:
                final_frontier.append(row)
        recommendation = choose_recommendation(final_frontier)
    else:
        for candidate_name in locked_quantizations:
            candidate_rows_by_name[candidate_name]["final_frontier"] = False

    per_dataset_rows: list[dict[str, Any]] = []
    for phase_name, rows in (
        ("validation", validation_summary_rows),
        ("test", test_summary_rows),
        ("stress", stress_summary_rows),
    ):
        for row in rows:
            enriched = dict(row)
            enriched["phase"] = phase_name
            enriched["language"] = xnli_language(str(row["dataset"]))
            per_dataset_rows.append(enriched)

    per_language_rows = [
        row
        for row in per_dataset_rows
        if row["language"] is not None and row["phase"] == "test"
    ]

    candidate_summary_rows = sorted(candidate_rows_by_name.values(), key=lambda row: row["quantization"])
    candidate_csv_path = output_prefix.with_suffix(".csv")
    candidate_json_path = output_prefix.with_suffix(".json")
    per_dataset_csv_path = output_prefix.parent / f"{output_prefix.name}-per-dataset.csv"
    per_dataset_json_path = output_prefix.parent / f"{output_prefix.name}-per-dataset.json"
    per_language_csv_path = output_prefix.parent / f"{output_prefix.name}-per-language.csv"
    per_language_json_path = output_prefix.parent / f"{output_prefix.name}-per-language.json"

    write_csv(
        candidate_csv_path,
        candidate_summary_rows,
        [
            "quantization",
            "artifact_path",
            "size_bytes",
            "smooth_quant_disabled",
            "retry_reason",
            "validation_complete",
            "validation_gate_pass",
            "validation_gate_reason_text",
            "validation_gold_accuracy",
            "validation_float_label_agreement",
            "mnli_macro_accuracy",
            "mnli_macro_accuracy_drop",
            "anli_macro_accuracy",
            "anli_macro_accuracy_drop",
            "peak_rss_ratio_vs_reference",
            "cpu_persistent_load_median_ms",
            "cpu_persistent_warm_median_ms",
            "cpu_persistent_resident_after_warmup_bytes",
            "cpu_persistent_peak_rss_bytes",
            "cpu_cold_load_median_ms",
            "cpu_cold_warm_median_ms",
            "final_test_complete",
            "final_test_gold_accuracy",
            "final_test_float_label_agreement",
            "stress_complete",
            "stress_gold_accuracy",
            "stress_float_label_agreement",
            "final_frontier",
        ],
    )
    candidate_json_path.write_text(
        json.dumps(
            {
                "locked_quantizations": locked_quantizations,
                "recommendation": recommendation,
                "candidates": candidate_summary_rows,
                "final_frontier": [row["quantization"] for row in final_frontier],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        per_dataset_csv_path,
        per_dataset_rows,
        [
            "phase",
            "dataset",
            "role",
            "backend",
            "quantization",
            "language",
            "gold_accuracy",
            "float_label_agreement",
            "mean_abs_logit_delta",
            "max_abs_logit_delta",
            "example_count",
            "labeled_example_count",
            "correct_prediction_count",
            "disagreement_count",
            "pareto_frontier",
        ],
    )
    per_dataset_json_path.write_text(
        json.dumps({"rows": per_dataset_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        per_language_csv_path,
        per_language_rows,
        [
            "phase",
            "dataset",
            "language",
            "quantization",
            "gold_accuracy",
            "float_label_agreement",
            "example_count",
        ],
    )
    per_language_json_path.write_text(
        json.dumps({"rows": per_language_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = report_markdown_path.parent
    lines = [
        "# Attempt4 CPU Deployment Study",
        "",
        "## Locked Quantizations",
        "",
        f"- Locked after development gates: {', '.join(f'`{name}`' for name in locked_quantizations)}",
        f"- Recommendation: `{recommendation['quantization']}`" if recommendation else "- Recommendation: pending locked-final test",
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Gate | Dev Acc | Dev Float Agree | CPU Warm | CPU Steady RSS | CPU Peak RSS | Cold Load |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidate_summary_rows:
        gate_text = "pass" if row.get("validation_gate_pass") else "fail"
        lines.append(
            "| "
            f"`{row['quantization']}` | "
            f"{gate_text} | "
            f"{percent_text(row.get('validation_gold_accuracy'))} | "
            f"{percent_text(row.get('validation_float_label_agreement'))} | "
            f"{ms_text(row.get('cpu_persistent_warm_median_ms'))} | "
            f"{mib_text(row.get('cpu_persistent_resident_after_warmup_bytes'))} | "
            f"{mib_text(row.get('cpu_persistent_peak_rss_bytes'))} | "
            f"{ms_text(row.get('cpu_cold_load_median_ms'))} |"
        )

    if test_aggregates:
        lines.extend(
            [
                "",
                "## Locked Final Frontier",
                "",
                "| Candidate | Size | Final Acc | Final Float Agree | CPU Warm | CPU Steady RSS | Frontier |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in candidate_summary_rows:
            if row["quantization"] not in locked_quantizations or not row.get("final_test_complete"):
                continue
            lines.append(
                "| "
                f"`{row['quantization']}` | "
                f"{mib_text(row.get('size_bytes'))} | "
                f"{percent_text(row.get('final_test_gold_accuracy'))} | "
                f"{percent_text(row.get('final_test_float_label_agreement'))} | "
                f"{ms_text(row.get('cpu_persistent_warm_median_ms'))} | "
                f"{mib_text(row.get('cpu_persistent_resident_after_warmup_bytes'))} | "
                f"{'frontier' if row.get('final_frontier') else '-'} |"
            )

    if per_language_rows:
        lines.extend(
            [
                "",
                "## XNLI Per-Language Rows",
                "",
                f"- {markdown_link(per_language_csv_path, report_dir)}",
                f"- {markdown_link(per_language_json_path, report_dir)}",
            ]
        )

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- {markdown_link(candidate_csv_path, report_dir)}",
            f"- {markdown_link(candidate_json_path, report_dir)}",
            f"- {markdown_link(per_dataset_csv_path, report_dir)}",
            f"- {markdown_link(per_dataset_json_path, report_dir)}",
            f"- {markdown_link(validation_summary_path, report_dir)}",
            f"- {markdown_link(validation_runtime_path, report_dir)}",
            f"- {markdown_link(datasets_manifest_path, report_dir)}",
        ]
    )
    if test_summary_path:
        lines.append(f"- {markdown_link(test_summary_path, report_dir)}")
    if stress_summary_path:
        lines.append(f"- {markdown_link(stress_summary_path, report_dir)}")
    if cold_benchmark_path:
        lines.append(f"- {markdown_link(cold_benchmark_path, report_dir)}")
    report_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {candidate_csv_path}")
    print(f"Wrote {candidate_json_path}")
    print(f"Wrote {per_dataset_csv_path}")
    print(f"Wrote {per_dataset_json_path}")
    print(f"Wrote {per_language_csv_path}")
    print(f"Wrote {per_language_json_path}")
    print(f"Wrote {report_markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
