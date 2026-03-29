#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models/mdeberta/onnx/candidates/attempt1"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "benchmarks/nli/attempt1-search-summary.json"
DEFAULT_SUMMARY_CSV = REPO_ROOT / "benchmarks/nli/attempt1-search-summary.csv"
DEFAULT_CORE_PROBE = REPO_ROOT / "benchmarks/nli/hf-core-probe.tsv"
DEFAULT_HARD_PROBE = REPO_ROOT / "benchmarks/nli/hf-probe-set.tsv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    track: str
    path: pathlib.Path
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the attempt1 candidate generation flow across NNCF accuracy/fidelity PTQ and "
            "ORT static QDQ tuning, screening candidates in core -> hard -> full order."
        )
    )
    parser.add_argument(
        "--calibration-tsv",
        dest="calibration_tsvs",
        action="append",
        default=[],
        help="Calibration TSV for NNCF/static generation. Repeat to add more.",
    )
    parser.add_argument(
        "--validation-tsv",
        dest="validation_tsvs",
        action="append",
        default=[],
        help="Held-out validation TSV for NNCF accuracy-control mode. Repeat to add more.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for generated candidates (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--candidate",
        dest="candidate_names",
        action="append",
        default=[],
        help="Candidate name to include. Repeat to filter the sweep.",
    )
    parser.add_argument(
        "--static-max-examples",
        dest="static_subset_sizes",
        action="append",
        type=int,
        default=[],
        help="Static quantization max-total-examples value. Repeat to add more. Defaults to 32 and 128.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_SUMMARY_JSON),
        help=f"Summary JSON output (default: {DEFAULT_SUMMARY_JSON}).",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(DEFAULT_SUMMARY_CSV),
        help=f"Summary CSV output (default: {DEFAULT_SUMMARY_CSV}).",
    )
    parser.add_argument(
        "--run-runtime-top-k",
        type=int,
        default=0,
        help="Run persistent runtime + RSS only for the union of the top K full-accuracy and top K HF-agreement candidates (default: 0).",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_summary_csv(path: pathlib.Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    candidate_rows = [row for row in rows if row["candidate"] != "float"]
    if not candidate_rows:
        raise RuntimeError(f"No non-float candidate row found in {path}")
    return candidate_rows[0]


def write_summary_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "candidate",
        "track",
        "path",
        "generate_status",
        "core_accuracy",
        "core_hf_agreement",
        "hard_accuracy",
        "hard_hf_agreement",
        "full_accuracy",
        "full_hf_agreement",
        "full_xnli_zh_accuracy",
        "runtime_cpu_warm_ms",
        "runtime_coreml_warm_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_candidate_specs(args: argparse.Namespace) -> list[CandidateSpec]:
    if not args.calibration_tsvs:
        raise RuntimeError("At least one --calibration-tsv is required")
    if not args.validation_tsvs:
        raise RuntimeError("At least one --validation-tsv is required")

    output_dir = pathlib.Path(args.output_dir)
    families = ["none", "attention_only", "attention_proj_only"]
    static_subset_sizes = args.static_subset_sizes or [32, 128]

    specs: list[CandidateSpec] = []
    for family in families:
        for track_name, metric in (
            ("accuracy", "gold_accuracy"),
            ("fidelity", "hf_agreement"),
        ):
            name = f"nncf_{track_name}_{family}"
            path = output_dir / f"{name}.onnx"
            command = [
                sys.executable,
                str(REPO_ROOT / "tools/quantize-onnx-nncf.py"),
                "--mode",
                "accuracy-control",
                "--metric",
                metric,
                "--preprocess",
                "--preset",
                "mixed",
                "--ignored-scope-family",
                family,
                "--subset-size",
                "300",
                "--output",
                str(path),
            ]
            for tsv in args.calibration_tsvs:
                command.extend(["--calibration-tsv", tsv])
            for tsv in args.validation_tsvs:
                command.extend(["--validation-tsv", tsv])
            if args.force:
                command.append("--force")
            specs.append(CandidateSpec(name=name, track=f"nncf_{track_name}", path=path, command=command))

    static_configs = [
        ("s8s8", "qint8", "qint8", False),
        ("u8u8", "quint8", "quint8", False),
        ("u8s8_rr", "quint8", "qint8", True),
    ]
    for family in families:
        for method in ("minmax", "percentile"):
            for subset_size in static_subset_sizes:
                for dtype_name, activation_type, weight_type, reduce_range in static_configs:
                    name = f"static_{family}_{dtype_name}_{method}_n{subset_size}"
                    path = output_dir / f"{name}.onnx"
                    command = [
                        sys.executable,
                        str(REPO_ROOT / "tools/quantize-onnx-static.py"),
                        "--output",
                        str(path),
                        "--ignore-family",
                        family,
                        "--quant-format",
                        "qdq",
                        "--activation-type",
                        activation_type,
                        "--weight-type",
                        weight_type,
                        "--calibrate-method",
                        method,
                        "--max-total-examples",
                        str(subset_size),
                    ]
                    if reduce_range:
                        command.append("--reduce-range")
                    for tsv in args.calibration_tsvs:
                        command.extend(["--calibration-tsv", tsv])
                    if args.force:
                        command.append("--force")
                    specs.append(CandidateSpec(name=name, track="static", path=path, command=command))

    if args.candidate_names:
        allowed = set(args.candidate_names)
        specs = [spec for spec in specs if spec.name in allowed]
    if not specs:
        raise RuntimeError("No attempt1 candidates selected")
    return specs


def run_generation(spec: CandidateSpec) -> str:
    subprocess.run(spec.command, check=True)
    return "ok"


def run_hf_stage(
    spec: CandidateSpec,
    stage_name: str,
    tsv_paths: list[pathlib.Path] | None,
) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix=f"attempt1-{stage_name}-") as tmp_dir:
        summary_csv = pathlib.Path(tmp_dir) / f"{stage_name}.csv"
        command = [
            sys.executable,
            str(REPO_ROOT / "tools/benchmark-hf-onnx-models.py"),
            "--sample-mode",
            "first",
            "--max-examples-per-source",
            "0",
            "--compare-model",
            f"{spec.name}={spec.path}",
            "--summary-csv",
            str(summary_csv),
        ]
        if tsv_paths:
            for tsv_path in tsv_paths:
                command.extend(["--tsv", str(tsv_path)])
        subprocess.run(command, check=True)
        return parse_summary_csv(summary_csv)


def run_runtime_stage(specs: list[CandidateSpec]) -> dict[str, dict[str, float]]:
    if not specs:
        return {}
    with tempfile.TemporaryDirectory(prefix="attempt1-runtime-") as tmp_dir:
        summary_json = pathlib.Path(tmp_dir) / "runtime.json"
        command = [
            sys.executable,
            str(REPO_ROOT / "tools/benchmark-nli-runtime.py"),
            "--mode",
            "persistent",
            "--max-examples",
            "0",
            "--repeat",
            "3",
            "--warmup",
            "1",
            "--measure-rss",
            "--summary-json",
            str(summary_json),
        ]
        for spec in specs:
            command.extend(["--model", f"{spec.name}={spec.path}"])
        subprocess.run(command, check=True)
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        per_candidate: dict[str, dict[str, float]] = {}
        for row in payload["rows"]:
            per_candidate.setdefault(row["candidate"], {})
            per_candidate[row["candidate"]][f"{row['backend']}_warm_ms"] = row["warm_latency_ms"]["median"]
        return per_candidate


def select_runtime_candidates(rows: list[dict[str, object]], top_k: int) -> list[str]:
    full_rows = [row for row in rows if row.get("full_accuracy") not in ("", None)]
    if not full_rows or top_k <= 0:
        return []

    top_accuracy = sorted(
        full_rows,
        key=lambda row: float(row["full_accuracy"]),
        reverse=True,
    )[:top_k]
    top_fidelity = sorted(
        full_rows,
        key=lambda row: float(row["full_hf_agreement"]),
        reverse=True,
    )[:top_k]
    names = {row["candidate"] for row in top_accuracy + top_fidelity}
    return sorted(names)


def main() -> int:
    args = parse_args()
    specs = build_candidate_specs(args)

    rows: list[dict[str, object]] = []
    for spec in specs:
        row: dict[str, object] = {
            "candidate": spec.name,
            "track": spec.track,
            "path": str(spec.path),
            "generate_status": "pending",
        }
        try:
            row["generate_status"] = run_generation(spec)
            core = run_hf_stage(spec, "core", [DEFAULT_CORE_PROBE])
            row["core_accuracy"] = core["accuracy"]
            row["core_hf_agreement"] = core["hf_agreement"]

            hard = run_hf_stage(spec, "hard", [DEFAULT_HARD_PROBE])
            row["hard_accuracy"] = hard["accuracy"]
            row["hard_hf_agreement"] = hard["hf_agreement"]

            full = run_hf_stage(spec, "full", None)
            row["full_accuracy"] = full["accuracy"]
            row["full_hf_agreement"] = full["hf_agreement"]
            row["full_xnli_zh_accuracy"] = full["xnli_zh_accuracy"]
        except subprocess.CalledProcessError as exc:
            row["generate_status"] = f"failed({exc.returncode})"
        rows.append(row)

    runtime_names = set(select_runtime_candidates(rows, args.run_runtime_top_k))
    runtime_specs = [spec for spec in specs if spec.name in runtime_names]
    runtime_rows = run_runtime_stage(runtime_specs) if runtime_specs else {}
    for row in rows:
        runtime_payload = runtime_rows.get(str(row["candidate"]))
        if not runtime_payload:
            continue
        row["runtime_cpu_warm_ms"] = runtime_payload.get("cpu_warm_ms", "")
        row["runtime_coreml_warm_ms"] = runtime_payload.get("coreml_warm_ms", "")

    summary_json_path = pathlib.Path(args.summary_json)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(pathlib.Path(args.summary_csv), rows)

    for row in rows:
        print(
            f"{row['candidate']}: status={row['generate_status']} "
            f"core_acc={row.get('core_accuracy', '-')}"
            f" hard_acc={row.get('hard_accuracy', '-')}"
            f" full_acc={row.get('full_accuracy', '-')}"
            f" full_hf={row.get('full_hf_agreement', '-')}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
