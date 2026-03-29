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
DEFAULT_INPUT_MODEL = REPO_ROOT / "models/mdeberta/onnx/model.onnx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models/mdeberta/onnx/candidates/attempt1"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "benchmarks/nli/attempt1-search-summary.json"
DEFAULT_SUMMARY_CSV = REPO_ROOT / "benchmarks/nli/attempt1-search-summary.csv"
DEFAULT_CORE_PROBE = REPO_ROOT / "benchmarks/nli/hf-core-probe.tsv"
DEFAULT_HARD_PROBE = REPO_ROOT / "benchmarks/nli/hf-probe-set.tsv"
SUPPORTED_FAMILIES = ("none", "attention_only", "attention_proj_only")
SUPPORTED_TRACKS = ("nncf_accuracy", "nncf_fidelity", "static")
SUPPORTED_STATIC_METHODS = ("minmax", "percentile")
SUPPORTED_STATIC_DTYPES = ("s8s8", "u8u8", "u8s8_rr")
SUPPORTED_NNCF_BIAS_CORRECTIONS = ("fast", "accurate")
FLOAT_CORE_ACCURACY = 0.44
FLOAT_HARD_ACCURACY = 28 / 61
MIN_PROBE_HF_AGREEMENT = 0.85
MIN_HARD_XNLI_ZH_HITS = 3


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
        "--input-model",
        default=str(DEFAULT_INPUT_MODEL),
        help=f"Input ONNX model used for candidate generation (default: {DEFAULT_INPUT_MODEL}).",
    )
    parser.add_argument(
        "--candidate",
        dest="candidate_names",
        action="append",
        default=[],
        help="Candidate name to include. Repeat to filter the sweep.",
    )
    parser.add_argument(
        "--family",
        dest="family_names",
        action="append",
        choices=SUPPORTED_FAMILIES,
        default=[],
        help="Structured ignore family to include. Repeat to filter the sweep.",
    )
    parser.add_argument(
        "--track",
        dest="track_names",
        action="append",
        choices=SUPPORTED_TRACKS,
        default=[],
        help="Candidate track to include. Repeat to filter the sweep.",
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
        "--static-calibrate-method",
        dest="static_calibrate_methods",
        action="append",
        choices=SUPPORTED_STATIC_METHODS,
        default=[],
        help="Static calibration method to include. Repeat to add more. Defaults to minmax only.",
    )
    parser.add_argument(
        "--static-dtype",
        dest="static_dtype_names",
        action="append",
        choices=SUPPORTED_STATIC_DTYPES,
        default=[],
        help="Static activation/weight scheme to include. Repeat to add more.",
    )
    parser.add_argument(
        "--disable-static-preprocess",
        action="store_true",
        help="Skip ORT quant_pre_process for static candidates.",
    )
    parser.add_argument(
        "--static-skip-preprocess-optimization",
        action="store_true",
        help="Disable graph optimization during static quant_pre_process.",
    )
    parser.add_argument(
        "--nncf-subset-size",
        dest="nncf_subset_sizes",
        action="append",
        type=int,
        default=[],
        help="NNCF subset size to include. Repeat to add more. Defaults to 300.",
    )
    parser.add_argument(
        "--nncf-max-drop",
        dest="nncf_max_drops",
        action="append",
        type=float,
        default=[],
        help="NNCF max-drop value to include. Repeat to add more. Defaults to 0.01.",
    )
    parser.add_argument(
        "--nncf-bias-correction",
        dest="nncf_bias_corrections",
        action="append",
        choices=SUPPORTED_NNCF_BIAS_CORRECTIONS,
        default=[],
        help="NNCF bias-correction mode to include. Repeat to add more. Defaults to fast only.",
    )
    parser.add_argument(
        "--disable-nncf-preprocess",
        action="store_true",
        help="Skip ORT quant_pre_process for NNCF candidates.",
    )
    parser.add_argument(
        "--nncf-skip-preprocess-optimization",
        action="store_true",
        help="Disable graph optimization during NNCF quant_pre_process.",
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
    parser.add_argument(
        "--disable-probe-gating",
        action="store_true",
        help="Always run the full suite, even for candidates that fail the probe promotion gates.",
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
        "screening_status",
        "screening_reason",
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


def write_summary_files(summary_json: pathlib.Path, summary_csv: pathlib.Path, rows: list[dict[str, object]]) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(summary_csv, rows)


def format_name_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_candidate_specs(args: argparse.Namespace) -> list[CandidateSpec]:
    if not args.calibration_tsvs:
        raise RuntimeError("At least one --calibration-tsv is required")

    output_dir = pathlib.Path(args.output_dir)
    input_model = pathlib.Path(args.input_model)
    if not input_model.is_file():
        raise RuntimeError(f"Input model not found: {input_model}")

    families = args.family_names or list(SUPPORTED_FAMILIES)
    allowed_tracks = set(args.track_names or SUPPORTED_TRACKS)
    if ("nncf_accuracy" in allowed_tracks or "nncf_fidelity" in allowed_tracks) and not args.validation_tsvs:
        raise RuntimeError("At least one --validation-tsv is required for NNCF accuracy-control tracks")
    static_subset_sizes = args.static_subset_sizes or [32, 128]
    static_methods = args.static_calibrate_methods or ["minmax"]
    static_dtypes = set(args.static_dtype_names or SUPPORTED_STATIC_DTYPES)
    nncf_subset_sizes = args.nncf_subset_sizes or [300]
    nncf_max_drops = args.nncf_max_drops or [0.01]
    nncf_bias_corrections = args.nncf_bias_corrections or ["fast"]

    specs: list[CandidateSpec] = []
    if "nncf_accuracy" in allowed_tracks or "nncf_fidelity" in allowed_tracks:
        for family in families:
            for track_name, metric in (
                ("accuracy", "gold_accuracy"),
                ("fidelity", "hf_agreement"),
            ):
                spec_track = f"nncf_{track_name}"
                if spec_track not in allowed_tracks:
                    continue
                for subset_size in nncf_subset_sizes:
                    for max_drop in nncf_max_drops:
                        for bias_correction in nncf_bias_corrections:
                            suffix_parts = []
                            if subset_size != 300:
                                suffix_parts.append(f"n{subset_size}")
                            if max_drop != 0.01:
                                suffix_parts.append(f"drop{format_name_float(max_drop)}")
                            if bias_correction != "fast":
                                suffix_parts.append(f"{bias_correction}bc")
                            name = f"nncf_{track_name}_{family}"
                            if suffix_parts:
                                name = f"{name}_{'_'.join(suffix_parts)}"
                            path = output_dir / f"{name}.onnx"
                            command = [
                                sys.executable,
                                str(REPO_ROOT / "tools/quantize-onnx-nncf.py"),
                                "--input",
                                str(input_model),
                                "--mode",
                                "accuracy-control",
                                "--metric",
                                metric,
                                "--preset",
                                "mixed",
                                "--ignored-scope-family",
                                family,
                                "--subset-size",
                                str(subset_size),
                                "--max-drop",
                                str(max_drop),
                                "--output",
                                str(path),
                            ]
                            if not args.disable_nncf_preprocess:
                                command.append("--preprocess")
                                if args.nncf_skip_preprocess_optimization:
                                    command.append("--skip-preprocess-optimization")
                            if bias_correction == "accurate":
                                command.append("--accurate-bias-correction")
                            else:
                                command.append("--fast-bias-correction")
                            for tsv in args.calibration_tsvs:
                                command.extend(["--calibration-tsv", tsv])
                            for tsv in args.validation_tsvs:
                                command.extend(["--validation-tsv", tsv])
                            if args.force:
                                command.append("--force")
                            specs.append(
                                CandidateSpec(name=name, track=spec_track, path=path, command=command)
                            )

    static_configs = [
        ("s8s8", "qint8", "qint8", False),
        ("u8u8", "quint8", "quint8", False),
        ("u8s8_rr", "quint8", "qint8", True),
    ]
    if "static" in allowed_tracks:
        for family in families:
            for method in static_methods:
                for subset_size in static_subset_sizes:
                    for dtype_name, activation_type, weight_type, reduce_range in static_configs:
                        if dtype_name not in static_dtypes:
                            continue
                        name = f"static_{family}_{dtype_name}_{method}_n{subset_size}"
                        path = output_dir / f"{name}.onnx"
                        command = [
                            sys.executable,
                            str(REPO_ROOT / "tools/quantize-onnx-static.py"),
                            "--input",
                            str(input_model),
                            "--output",
                            str(path),
                            "--ignore-family",
                            family,
                            "--quant-format",
                            "qdq",
                            "--max-examples-per-source",
                            "0",
                            "--activation-type",
                            activation_type,
                            "--weight-type",
                            weight_type,
                            "--calibrate-method",
                            method,
                            "--max-total-examples",
                            str(subset_size),
                        ]
                        if not args.disable_static_preprocess:
                            command.append("--preprocess")
                            if args.static_skip_preprocess_optimization:
                                command.append("--skip-preprocess-optimization")
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


def parse_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in ("", None):
        raise RuntimeError(f"Missing numeric field {field}")
    return float(value)


def parse_int(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if value in ("", None):
        raise RuntimeError(f"Missing integer field {field}")
    return int(float(value))


def accuracy_probe_gate(core: dict[str, str], hard: dict[str, str]) -> bool:
    return (
        parse_float(core, "accuracy") >= FLOAT_CORE_ACCURACY
        and parse_float(hard, "accuracy") >= FLOAT_HARD_ACCURACY
        and parse_int(hard, "xnli_zh_accuracy_hits") >= MIN_HARD_XNLI_ZH_HITS
    )


def fidelity_probe_gate(core: dict[str, str], hard: dict[str, str]) -> bool:
    return (
        parse_float(core, "hf_agreement") >= MIN_PROBE_HF_AGREEMENT
        and parse_float(hard, "hf_agreement") >= MIN_PROBE_HF_AGREEMENT
        and parse_int(hard, "xnli_zh_accuracy_hits") >= MIN_HARD_XNLI_ZH_HITS
    )


def probe_gate_outcome(
    spec: CandidateSpec,
    core: dict[str, str],
    hard: dict[str, str],
) -> tuple[bool, str, str]:
    accuracy_ok = accuracy_probe_gate(core, hard)
    fidelity_ok = fidelity_probe_gate(core, hard)

    if spec.track == "nncf_accuracy":
        if accuracy_ok:
            return True, "promoted_full", "accuracy_probe_gate"
        return False, "screened_out", "failed_accuracy_probe_gate"
    if spec.track == "nncf_fidelity":
        if fidelity_ok:
            return True, "promoted_full", "fidelity_probe_gate"
        return False, "screened_out", "failed_fidelity_probe_gate"

    if accuracy_ok:
        return True, "promoted_full", "static_accuracy_probe_gate"
    if fidelity_ok:
        return True, "promoted_full", "static_fidelity_probe_gate"
    return False, "screened_out", "failed_static_probe_gates"


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
    summary_json_path = pathlib.Path(args.summary_json)
    summary_csv_path = pathlib.Path(args.summary_csv)

    rows: list[dict[str, object]] = []
    for spec in specs:
        row: dict[str, object] = {
            "candidate": spec.name,
            "track": spec.track,
            "path": str(spec.path),
            "generate_status": "pending",
            "screening_status": "pending",
            "screening_reason": "",
        }
        try:
            row["generate_status"] = run_generation(spec)
            core = run_hf_stage(spec, "core", [DEFAULT_CORE_PROBE])
            row["core_accuracy"] = core["accuracy"]
            row["core_hf_agreement"] = core["hf_agreement"]

            hard = run_hf_stage(spec, "hard", [DEFAULT_HARD_PROBE])
            row["hard_accuracy"] = hard["accuracy"]
            row["hard_hf_agreement"] = hard["hf_agreement"]

            should_run_full = True
            screening_status = "promoted_full"
            screening_reason = "probe_gating_disabled"
            if not args.disable_probe_gating:
                should_run_full, screening_status, screening_reason = probe_gate_outcome(
                    spec,
                    core,
                    hard,
                )

            row["screening_status"] = screening_status
            row["screening_reason"] = screening_reason

            if should_run_full:
                full = run_hf_stage(spec, "full", None)
                row["full_accuracy"] = full["accuracy"]
                row["full_hf_agreement"] = full["hf_agreement"]
                row["full_xnli_zh_accuracy"] = full["xnli_zh_accuracy"]
        except subprocess.CalledProcessError as exc:
            row["generate_status"] = f"failed({exc.returncode})"
        rows.append(row)
        write_summary_files(summary_json_path, summary_csv_path, rows)

    runtime_names = set(select_runtime_candidates(rows, args.run_runtime_top_k))
    runtime_specs = [spec for spec in specs if spec.name in runtime_names]
    runtime_rows = run_runtime_stage(runtime_specs) if runtime_specs else {}
    for row in rows:
        runtime_payload = runtime_rows.get(str(row["candidate"]))
        if not runtime_payload:
            continue
        row["runtime_cpu_warm_ms"] = runtime_payload.get("cpu_warm_ms", "")
        row["runtime_coreml_warm_ms"] = runtime_payload.get("coreml_warm_ms", "")

    write_summary_files(summary_json_path, summary_csv_path, rows)

    for row in rows:
        print(
            f"{row['candidate']}: status={row['generate_status']} "
            f"screen={row.get('screening_status', '-')}/{row.get('screening_reason', '-')}"
            f" "
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
