#!/usr/bin/env python3

import argparse
import csv
import json
import pathlib
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "benchmarks/nli"

DEFAULT_FULL_CSV = BENCHMARK_DIR / "hf-finalist-full.csv"
DEFAULT_HARD_CSV = BENCHMARK_DIR / "hf-probe-benchmark.csv"
DEFAULT_CORE_CSV = BENCHMARK_DIR / "hf-core-probe-benchmark.csv"
DEFAULT_CPU_COLD_CSV = BENCHMARK_DIR / "runtime-cpu-core-probe.csv"
DEFAULT_COREML_COLD_CSV = BENCHMARK_DIR / "runtime-coreml-core-probe.csv"
DEFAULT_CPU_PERSISTENT_CSV = BENCHMARK_DIR / "runtime-cpu-core-probe-persistent.csv"
DEFAULT_COREML_PERSISTENT_CSV = BENCHMARK_DIR / "runtime-coreml-core-probe-persistent.csv"

DEFAULT_DASHBOARD_JSON = BENCHMARK_DIR / "quantization-dashboard.json"
DEFAULT_DASHBOARD_CSV = BENCHMARK_DIR / "quantization-dashboard.csv"
DEFAULT_DASHBOARD_MD = BENCHMARK_DIR / "quantization-dashboard.md"
DEFAULT_RECOMMENDATION_MD = REPO_ROOT / "QUANTIZATION_RECOMMENDATION_1.md"

CANDIDATE_PATHS = {
    "float": "models/mdeberta/onnx/model.onnx",
    "attention_only": (
        "models/mdeberta/onnx/candidates/family_search/"
        "dynamic_qint8_matmul_attention_only.onnx"
    ),
    "attention_proj_only": (
        "models/mdeberta/onnx/candidates/family_search/"
        "dynamic_qint8_matmul_attention_proj_only.onnx"
    ),
}
PREFERRED_ORDER = ["float", "attention_only", "attention_proj_only"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a consolidated quantization dashboard from the current quality and "
            "runtime benchmark CSVs, and emit a shipping recommendation."
        )
    )
    parser.add_argument("--full-csv", default=str(DEFAULT_FULL_CSV))
    parser.add_argument("--hard-csv", default=str(DEFAULT_HARD_CSV))
    parser.add_argument("--core-csv", default=str(DEFAULT_CORE_CSV))
    parser.add_argument("--cpu-cold-csv", default=str(DEFAULT_CPU_COLD_CSV))
    parser.add_argument("--coreml-cold-csv", default=str(DEFAULT_COREML_COLD_CSV))
    parser.add_argument("--cpu-persistent-csv", default=str(DEFAULT_CPU_PERSISTENT_CSV))
    parser.add_argument("--coreml-persistent-csv", default=str(DEFAULT_COREML_PERSISTENT_CSV))
    parser.add_argument("--dashboard-json", default=str(DEFAULT_DASHBOARD_JSON))
    parser.add_argument("--dashboard-csv", default=str(DEFAULT_DASHBOARD_CSV))
    parser.add_argument("--dashboard-markdown", default=str(DEFAULT_DASHBOARD_MD))
    parser.add_argument("--recommendation-markdown", default=str(DEFAULT_RECOMMENDATION_MD))
    return parser.parse_args()


def read_csv_rows(path: pathlib.Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = {row["candidate"]: row for row in reader}
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def percent_text(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def ms_text(value: Any) -> str:
    return f"{float(value):.3f} ms"


def mb_text_from_bytes(value: Any) -> str:
    return f"{int(value) / (1024.0 * 1024.0):.2f} MB"


def candidate_sort_key(candidate: str) -> tuple[int, str]:
    try:
        return (PREFERRED_ORDER.index(candidate), candidate)
    except ValueError:
        return (len(PREFERRED_ORDER), candidate)


def enrich_rows(
    full_rows: dict[str, dict[str, str]],
    hard_rows: dict[str, dict[str, str]],
    core_rows: dict[str, dict[str, str]],
    cpu_cold_rows: dict[str, dict[str, str]],
    coreml_cold_rows: dict[str, dict[str, str]],
    cpu_persistent_rows: dict[str, dict[str, str]],
    coreml_persistent_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    candidates = sorted(
        set(full_rows)
        | set(hard_rows)
        | set(core_rows)
        | set(cpu_cold_rows)
        | set(coreml_cold_rows)
        | set(cpu_persistent_rows)
        | set(coreml_persistent_rows),
        key=candidate_sort_key,
    )
    if "float" not in candidates:
        raise RuntimeError("Dashboard requires a float reference row")

    float_full = full_rows["float"]
    float_cpu_persistent = cpu_persistent_rows["float"]
    float_coreml_persistent = coreml_persistent_rows["float"]

    merged: list[dict[str, Any]] = []
    for candidate in candidates:
        full = full_rows[candidate]
        hard = hard_rows[candidate]
        core = core_rows[candidate]
        cpu_cold = cpu_cold_rows[candidate]
        coreml_cold = coreml_cold_rows[candidate]
        cpu_persistent = cpu_persistent_rows[candidate]
        coreml_persistent = coreml_persistent_rows[candidate]

        row: dict[str, Any] = {
            "candidate": candidate,
            "model_path": CANDIDATE_PATHS.get(candidate, ""),
            "size_bytes": int(cpu_cold["file_size_bytes"]),
            "size_mb": int(cpu_cold["file_size_bytes"]) / (1024.0 * 1024.0),
            "full_accuracy": float(full["accuracy"]),
            "full_accuracy_hits": int(full["accuracy_hits"]),
            "full_examples": int(full["examples"]),
            "full_hf_agreement": float(full["hf_agreement"]),
            "full_hf_agreements": int(full["hf_agreements"]),
            "full_net_delta_vs_float": int(full["net_accuracy_delta_vs_float"]),
            "full_fixed_float_errors": int(full["fixed_float_errors"]),
            "full_new_errors_vs_float": int(full["introduced_new_errors_vs_float"]),
            "full_xnli_zh_accuracy": float(full["xnli_zh_accuracy"]),
            "full_mean_max_abs_logit_delta_vs_hf": float(full["mean_max_abs_logit_delta_vs_hf"]),
            "hard_accuracy": float(hard["accuracy"]),
            "hard_accuracy_hits": int(hard["accuracy_hits"]),
            "hard_examples": int(hard["examples"]),
            "hard_hf_agreement": float(hard["hf_agreement"]),
            "hard_xnli_zh_accuracy": float(hard["xnli_zh_accuracy"]),
            "core_accuracy": float(core["accuracy"]),
            "core_accuracy_hits": int(core["accuracy_hits"]),
            "core_examples": int(core["examples"]),
            "core_hf_agreement": float(core["hf_agreement"]),
            "core_xnli_zh_accuracy": float(core["xnli_zh_accuracy"]),
            "cpu_cold_load_median_ms": float(cpu_cold["load_median_ms"]),
            "cpu_cold_warm_median_ms": float(cpu_cold["warm_median_ms"]),
            "cpu_cold_warm_p95_ms": float(cpu_cold["warm_p95_ms"]),
            "coreml_cold_load_median_ms": float(coreml_cold["load_median_ms"]),
            "coreml_cold_warm_median_ms": float(coreml_cold["warm_median_ms"]),
            "coreml_cold_warm_p95_ms": float(coreml_cold["warm_p95_ms"]),
            "cpu_persistent_load_median_ms": float(cpu_persistent["load_median_ms"]),
            "cpu_persistent_warm_median_ms": float(cpu_persistent["warm_median_ms"]),
            "cpu_persistent_warm_p95_ms": float(cpu_persistent["warm_p95_ms"]),
            "coreml_persistent_load_median_ms": float(coreml_persistent["load_median_ms"]),
            "coreml_persistent_warm_median_ms": float(coreml_persistent["warm_median_ms"]),
            "coreml_persistent_warm_p95_ms": float(coreml_persistent["warm_p95_ms"]),
        }
        row["full_hf_gap_vs_float"] = row["full_hf_agreement"] - float(float_full["hf_agreement"])
        row["cpu_persistent_warm_delta_vs_float_ms"] = (
            row["cpu_persistent_warm_median_ms"] - float(float_cpu_persistent["warm_median_ms"])
        )
        row["coreml_persistent_warm_delta_vs_float_ms"] = (
            row["coreml_persistent_warm_median_ms"]
            - float(float_coreml_persistent["warm_median_ms"])
        )
        merged.append(row)
    return merged


def build_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    float_row = next(row for row in rows if row["candidate"] == "float")
    quantized_rows = [row for row in rows if row["candidate"] != "float"]
    accuracy_candidate = max(
        quantized_rows,
        key=lambda row: (
            row["full_accuracy"],
            row["hard_accuracy"],
            row["core_accuracy"],
            row["full_hf_agreement"],
        ),
    )
    fidelity_candidate = max(
        quantized_rows,
        key=lambda row: (
            row["full_hf_agreement"],
            -row["full_mean_max_abs_logit_delta_vs_hf"],
            row["cpu_persistent_warm_median_ms"] * -1.0,
        ),
    )
    fastest_persistent_candidate = min(
        quantized_rows,
        key=lambda row: (
            row["cpu_persistent_warm_median_ms"] + row["coreml_persistent_warm_median_ms"],
            -row["full_accuracy"],
        ),
    )

    return {
        "default_candidate": "float",
        "default_model_path": CANDIDATE_PATHS["float"],
        "optional_experimental_candidate": accuracy_candidate["candidate"],
        "optional_experimental_model_path": accuracy_candidate["model_path"],
        "quantized_fidelity_candidate": fidelity_candidate["candidate"],
        "quantized_fidelity_model_path": fidelity_candidate["model_path"],
        "fastest_quantized_persistent_candidate": fastest_persistent_candidate["candidate"],
        "rationale": [
            (
                "Float remains the best default because it is the exact HF-equivalent "
                "reference, keeps 100.00% HF label agreement on the full suite, and still "
                "loads fastest on both CPU and CoreML."
            ),
            (
                f"{accuracy_candidate['candidate']} is the best optional quantized model for "
                "accuracy-oriented experiments because it has the strongest full-suite "
                f"accuracy ({percent_text(accuracy_candidate['full_accuracy'])}) and the "
                f"largest net gain over float (+{accuracy_candidate['full_net_delta_vs_float']} "
                "correct examples on the full suite)."
            ),
            (
                f"{fidelity_candidate['candidate']} remains the better quantized fidelity "
                "baseline, but its full-suite HF agreement edge over "
                f"{accuracy_candidate['candidate']} is tiny "
                f"({percent_text(fidelity_candidate['full_hf_agreement'])} vs "
                f"{percent_text(accuracy_candidate['full_hf_agreement'])}) and does not come "
                "with a decisive runtime or size advantage."
            ),
        ],
        "serving_mode_caveat": (
            f"In persistent-session serving, the quantized finalists are modestly faster on warm "
            f"median latency than float: {fastest_persistent_candidate['candidate']} saves about "
            f"{abs(fastest_persistent_candidate['cpu_persistent_warm_delta_vs_float_ms']):.2f} ms "
            "on CPU and "
            f"{abs(fastest_persistent_candidate['coreml_persistent_warm_delta_vs_float_ms']):.2f} ms "
            "on CoreML. That is real, but only about a 2-3% steady-state edge."
        ),
    }


def write_dashboard_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate",
        "model_path",
        "size_mb",
        "full_accuracy",
        "full_hf_agreement",
        "full_net_delta_vs_float",
        "full_xnli_zh_accuracy",
        "hard_accuracy",
        "hard_hf_agreement",
        "core_accuracy",
        "core_hf_agreement",
        "cpu_cold_load_median_ms",
        "cpu_cold_warm_median_ms",
        "coreml_cold_load_median_ms",
        "coreml_cold_warm_median_ms",
        "cpu_persistent_load_median_ms",
        "cpu_persistent_warm_median_ms",
        "cpu_persistent_warm_delta_vs_float_ms",
        "coreml_persistent_load_median_ms",
        "coreml_persistent_warm_median_ms",
        "coreml_persistent_warm_delta_vs_float_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_dashboard_json(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
    inputs: dict[str, pathlib.Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_from": {
            key: str(value.relative_to(REPO_ROOT)) if value.is_relative_to(REPO_ROOT) else str(value)
            for key, value in inputs.items()
        },
        "recommendation": recommendation,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_dashboard_markdown(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> None:
    lines = [
        "# Quantization Dashboard",
        "",
        "## Recommendation",
        "",
        f"- Default model: `{recommendation['default_candidate']}` "
        f"(`{recommendation['default_model_path']}`)",
        f"- Optional experimental quantized model: `{recommendation['optional_experimental_candidate']}` "
        f"(`{recommendation['optional_experimental_model_path']}`)",
        f"- Quantized fidelity baseline: `{recommendation['quantized_fidelity_candidate']}` "
        f"(`{recommendation['quantized_fidelity_model_path']}`)",
        f"- Serving-mode caveat: {recommendation['serving_mode_caveat']}",
        "",
        "Why:",
        "",
    ]
    for item in recommendation["rationale"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Consolidated Table",
            "",
            "| Candidate | Size | Full Acc | Full HF | Hard Acc | Core Acc | `xnli-zh` Full | CPU Cold Load | CPU Persistent Warm | CoreML Cold Load | CoreML Persistent Warm |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row['candidate']}` | "
            f"{row['size_mb']:.2f} MB | "
            f"{percent_text(row['full_accuracy'])} | "
            f"{percent_text(row['full_hf_agreement'])} | "
            f"{percent_text(row['hard_accuracy'])} | "
            f"{percent_text(row['core_accuracy'])} | "
            f"{percent_text(row['full_xnli_zh_accuracy'])} | "
            f"{ms_text(row['cpu_cold_load_median_ms'])} | "
            f"{ms_text(row['cpu_persistent_warm_median_ms'])} | "
            f"{ms_text(row['coreml_cold_load_median_ms'])} | "
            f"{ms_text(row['coreml_persistent_warm_median_ms'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The full-suite benchmark remains the final quality ranking.",
            "- The hard probe and core probe are stress gates, not balanced evaluation sets.",
            "- Cold-start runtime is from the CLI path that loads a model per invocation.",
            "- Persistent runtime is the better proxy for long-lived serving, but the observed advantage remains modest.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendation_markdown(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> None:
    row_map = {row["candidate"]: row for row in rows}
    default_row = row_map[recommendation["default_candidate"]]
    experimental_row = row_map[recommendation["optional_experimental_candidate"]]
    fidelity_row = row_map[recommendation["quantized_fidelity_candidate"]]

    lines = [
        "# Quantization Recommendation 1",
        "",
        "## Recommendation",
        "",
        f"- Default model: `{recommendation['default_model_path']}`",
        f"- Optional experimental quantized model: `{recommendation['optional_experimental_model_path']}`",
        f"- Secondary quantized fidelity baseline: `{recommendation['quantized_fidelity_model_path']}`",
        "",
        "## Rationale",
        "",
        (
            f"- Float remains the best default because it keeps "
            f"{percent_text(default_row['full_hf_agreement'])} HF agreement on the full suite, "
            f"has the strongest `xnli-zh` result ({percent_text(default_row['full_xnli_zh_accuracy'])}), "
            "and initializes fastest on both CPU and CoreML."
        ),
        (
            f"- `{experimental_row['candidate']}` is the recommended experimental quantized path because "
            f"it has the best full-suite accuracy among quantized models "
            f"({percent_text(experimental_row['full_accuracy'])}) and the best net gain over float "
            f"({experimental_row['full_net_delta_vs_float']:+d} correct examples on the full suite)."
        ),
        (
            f"- `{fidelity_row['candidate']}` is still useful as the research fidelity baseline because "
            f"its full-suite HF agreement is slightly higher "
            f"({percent_text(fidelity_row['full_hf_agreement'])}), but that edge is too small to justify "
            "preferring it over the accuracy-oriented quantized candidate for a repo-level experimental default."
        ),
        (
            "- Persistent-session runtime does favor quantization, but only modestly. The recommended "
            f"experimental path saves about {abs(experimental_row['cpu_persistent_warm_delta_vs_float_ms']):.2f} ms "
            "on CPU warm median and about "
            f"{abs(experimental_row['coreml_persistent_warm_delta_vs_float_ms']):.2f} ms on CoreML warm median."
        ),
        "",
        "## Caveat",
        "",
        (
            "- If deployment is clearly persistent-session oriented and exact HF fidelity matters less than "
            "warm steady-state latency, a quantized model is defensible as an opt-in path. That still does not "
            "justify changing the repo default away from float."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(rows: list[dict[str, Any]], recommendation: dict[str, Any]) -> None:
    print("candidate               full_acc   full_hf   hard_acc   core_acc   cpu_persist   coreml_persist")
    print("-----------------------------------------------------------------------------------------------")
    for row in rows:
        print(
            f"{row['candidate']:<22}"
            f"{percent_text(row['full_accuracy']):>10} "
            f"{percent_text(row['full_hf_agreement']):>9} "
            f"{percent_text(row['hard_accuracy']):>9} "
            f"{percent_text(row['core_accuracy']):>9} "
            f"{ms_text(row['cpu_persistent_warm_median_ms']):>13} "
            f"{ms_text(row['coreml_persistent_warm_median_ms']):>16}"
        )
    print()
    print(f"default: {recommendation['default_candidate']} -> {recommendation['default_model_path']}")
    print(
        "optional_experimental: "
        f"{recommendation['optional_experimental_candidate']} -> "
        f"{recommendation['optional_experimental_model_path']}"
    )
    print(
        "quantized_fidelity: "
        f"{recommendation['quantized_fidelity_candidate']} -> "
        f"{recommendation['quantized_fidelity_model_path']}"
    )


def main() -> int:
    args = parse_args()
    inputs = {
        "full_csv": pathlib.Path(args.full_csv),
        "hard_csv": pathlib.Path(args.hard_csv),
        "core_csv": pathlib.Path(args.core_csv),
        "cpu_cold_csv": pathlib.Path(args.cpu_cold_csv),
        "coreml_cold_csv": pathlib.Path(args.coreml_cold_csv),
        "cpu_persistent_csv": pathlib.Path(args.cpu_persistent_csv),
        "coreml_persistent_csv": pathlib.Path(args.coreml_persistent_csv),
    }
    full_rows = read_csv_rows(inputs["full_csv"])
    hard_rows = read_csv_rows(inputs["hard_csv"])
    core_rows = read_csv_rows(inputs["core_csv"])
    cpu_cold_rows = read_csv_rows(inputs["cpu_cold_csv"])
    coreml_cold_rows = read_csv_rows(inputs["coreml_cold_csv"])
    cpu_persistent_rows = read_csv_rows(inputs["cpu_persistent_csv"])
    coreml_persistent_rows = read_csv_rows(inputs["coreml_persistent_csv"])

    rows = enrich_rows(
        full_rows,
        hard_rows,
        core_rows,
        cpu_cold_rows,
        coreml_cold_rows,
        cpu_persistent_rows,
        coreml_persistent_rows,
    )
    recommendation = build_recommendation(rows)

    write_dashboard_json(pathlib.Path(args.dashboard_json), rows, recommendation, inputs)
    write_dashboard_csv(pathlib.Path(args.dashboard_csv), rows)
    write_dashboard_markdown(pathlib.Path(args.dashboard_markdown), rows, recommendation)
    write_recommendation_markdown(pathlib.Path(args.recommendation_markdown), rows, recommendation)
    print_summary(rows, recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
