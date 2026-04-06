#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import shutil
import sqlite3
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCRATCHPAD_ROOT = REPO_ROOT / "scratchpad" / "attempt4_cpu_focus"
DEFAULT_CATALOG_PATH = REPO_ROOT / "research" / "attempt4_cpu-focus" / "study_quantization_catalog.json"
DEFAULT_STUDY_BINARY = REPO_ROOT / "builddir" / "nli-study"
DEFAULT_EXECUTABLE = REPO_ROOT / "builddir" / "nli"
DEFAULT_PERSISTENT_EXECUTABLE = REPO_ROOT / "builddir" / "nli-runtime-bench"
SMOKE_DATASETS = (
    "hf-probe-set.tsv",
    "hf-core-probe.tsv",
)
QUANTIZATIONS = (
    "reference",
    "model_quantized",
    "dynamic_qint8_default",
    "dynamic_qint8_per_channel",
    "attention_only",
    "nncf_accuracy_attention_only",
    "nncf_fidelity_attention_proj_only",
    "nncf_fidelity_attention_only_n128_drop0p005",
    "static_attention_only_u8u8_minmax_n128",
    "static_attention_proj_only_s8s8_minmax_n300",
    "static_attention_proj_only_u8s8_rr_minmax_n128",
)
TOKENIZER_ASSET_FILENAMES = (
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded attempt4 CPU deployment study end-to-end: dataset prep, "
            "study evaluation, CPU runtime/RSS benchmarking, and final report generation."
        )
    )
    parser.add_argument(
        "--scratchpad-root",
        default=str(DEFAULT_SCRATCHPAD_ROOT),
        help=f"Scratchpad root directory (default: {DEFAULT_SCRATCHPAD_ROOT})",
    )
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_CATALOG_PATH),
        help=f"Study catalog path (default: {DEFAULT_CATALOG_PATH})",
    )
    parser.add_argument(
        "--study-binary",
        default=str(DEFAULT_STUDY_BINARY),
        help=f"Path to nli-study (default: {DEFAULT_STUDY_BINARY})",
    )
    parser.add_argument(
        "--executable",
        default=str(DEFAULT_EXECUTABLE),
        help=f"Path to nli CLI executable (default: {DEFAULT_EXECUTABLE})",
    )
    parser.add_argument(
        "--persistent-executable",
        default=str(DEFAULT_PERSISTENT_EXECUTABLE),
        help=f"Path to persistent benchmark executable (default: {DEFAULT_PERSISTENT_EXECUTABLE})",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-datasets",
        action="store_true",
        help="Regenerate the attempt4 dataset pack instead of reusing existing local files.",
    )
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        check=False,
    )
    if completed.returncode != 0 and not allow_failure:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed


def prepare_datasets(scratchpad_root: pathlib.Path, force: bool) -> pathlib.Path:
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "tools" / "prepare-attempt4-cpu-datasets.py"),
        "--scratchpad-root",
        str(scratchpad_root),
    ]
    if force:
        command.append("--force")
    run_command(command)
    return scratchpad_root / "reports" / "attempt4-datasets-manifest.json"


def stage_runtime_assets(scratchpad_root: pathlib.Path) -> None:
    source_root = REPO_ROOT / "models" / "mdeberta"
    dest_root = scratchpad_root / "models" / "mdeberta"
    dest_root.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_ASSET_FILENAMES:
        source_path = source_root / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Tokenizer asset not found: {source_path}")
        shutil.copyfile(source_path, dest_root / filename)


def initialize_study_db(
    study_binary: pathlib.Path,
    scratchpad_root: pathlib.Path,
    catalog_path: pathlib.Path,
    force: bool,
) -> None:
    command = [
        str(study_binary),
        "init",
        "--scratchpad-root",
        str(scratchpad_root),
        "--catalog",
        str(catalog_path),
    ]
    if force:
        command.append("--force")
    run_command(command)


def run_study(
    study_binary: pathlib.Path,
    scratchpad_root: pathlib.Path,
    quantization: str,
    dataset_name: str,
) -> None:
    run_command(
        [
            str(study_binary),
            "run",
            "--scratchpad-root",
            str(scratchpad_root),
            "--backend",
            "cpu",
            "--quantization",
            quantization,
            "--dataset",
            dataset_name,
        ]
    )


def summarize_role(
    scratchpad_root: pathlib.Path,
    role: str,
    output_prefix: pathlib.Path,
) -> pathlib.Path:
    run_command(
        [
            sys.executable,
            "-u",
            str(REPO_ROOT / "tools" / "summarize-study-db.py"),
            "--scratchpad-root",
            str(scratchpad_root),
            "--role",
            role,
            "--backend",
            "cpu",
            "--output-prefix",
            str(output_prefix),
        ]
    )
    return output_prefix.with_suffix(".json")


def artifact_paths_for_quantizations(
    scratchpad_root: pathlib.Path,
    quantizations: list[str],
) -> dict[str, str]:
    db_path = scratchpad_root / "db.sqlite3"
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            """
            SELECT q.name, a.path
            FROM quantization q
            JOIN artifact a ON a.quantization_id = q.id
            ORDER BY q.name
            """
        ).fetchall()
    finally:
        connection.close()
    path_map = {row[0]: row[1] for row in rows}
    return {name: path_map[name] for name in quantizations if name in path_map}


def complete_quantizations(summary_json_path: pathlib.Path, required_datasets: list[str]) -> list[str]:
    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    required_dataset_set = set(required_datasets)
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["quantization"]), set()).add(str(row["dataset"]))
    return sorted(
        quantization
        for quantization, datasets in grouped.items()
        if datasets == required_dataset_set
    )


def benchmark_cpu(
    scratchpad_root: pathlib.Path,
    executable: pathlib.Path,
    persistent_executable: pathlib.Path,
    *,
    mode: str,
    quantizations: list[str],
    output_prefix: pathlib.Path,
) -> pathlib.Path:
    model_paths = artifact_paths_for_quantizations(scratchpad_root, quantizations)
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "tools" / "benchmark-nli-runtime.py"),
        "--executable",
        str(executable),
        "--persistent-executable",
        str(persistent_executable),
        "--mode",
        mode,
        "--backend",
        "cpu",
        "--tsv",
        str(scratchpad_root / "datasets" / "hf-core-probe.tsv"),
        "--sample-mode",
        "first",
        "--max-examples",
        "0",
        "--measure-rss",
        "--summary-json",
        str(output_prefix.with_suffix(".json")),
        "--summary-csv",
        str(output_prefix.with_suffix(".csv")),
    ]
    for quantization in quantizations:
        model_path = model_paths.get(quantization)
        if not model_path:
            raise RuntimeError(f"Missing artifact path for benchmark candidate {quantization}")
        command.extend(["--model", f"{quantization}={model_path}"])
    run_command(command)
    return output_prefix.with_suffix(".csv")


def verify_role_assignments(
    scratchpad_root: pathlib.Path,
    expected: dict[str, list[str]],
) -> None:
    connection = sqlite3.connect(str(scratchpad_root / "db.sqlite3"))
    try:
        rows = connection.execute("SELECT name, role FROM dataset ORDER BY name").fetchall()
    finally:
        connection.close()
    actual_by_role: dict[str, list[str]] = {}
    for name, role in rows:
        actual_by_role.setdefault(role, []).append(name)
    for role, expected_names in expected.items():
        actual_names = sorted(actual_by_role.get(role, []))
        if sorted(expected_names) != actual_names:
            raise RuntimeError(
                f"Dataset-role mismatch for {role}: expected {sorted(expected_names)}, got {actual_names}"
            )


def build_report(
    datasets_manifest_path: pathlib.Path,
    validation_summary_path: pathlib.Path,
    validation_runtime_csv_path: pathlib.Path,
    output_prefix: pathlib.Path,
    *,
    test_summary_path: pathlib.Path | None = None,
    stress_summary_path: pathlib.Path | None = None,
    cold_benchmark_csv_path: pathlib.Path | None = None,
) -> pathlib.Path:
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "tools" / "build-attempt4-cpu-report.py"),
        "--datasets-manifest",
        str(datasets_manifest_path),
        "--validation-summary-json",
        str(validation_summary_path),
        "--validation-runtime-csv",
        str(validation_runtime_csv_path),
        "--output-prefix",
        str(output_prefix),
    ]
    if test_summary_path:
        command.extend(["--test-summary-json", str(test_summary_path)])
    if stress_summary_path:
        command.extend(["--stress-summary-json", str(stress_summary_path)])
    if cold_benchmark_csv_path:
        command.extend(["--cold-benchmark-csv", str(cold_benchmark_csv_path)])
    run_command(command)
    return output_prefix.with_suffix(".json")


def report_artifact_paths(output_prefix: pathlib.Path) -> dict[str, str]:
    return {
        "candidate_csv": str(output_prefix.with_suffix(".csv")),
        "candidate_json": str(output_prefix.with_suffix(".json")),
        "per_dataset_csv": str(output_prefix.parent / f"{output_prefix.name}-per-dataset.csv"),
        "per_dataset_json": str(output_prefix.parent / f"{output_prefix.name}-per-dataset.json"),
        "per_language_csv": str(output_prefix.parent / f"{output_prefix.name}-per-language.csv"),
        "per_language_json": str(output_prefix.parent / f"{output_prefix.name}-per-language.json"),
        "report_markdown": str(output_prefix.with_suffix(".md")),
    }


def main() -> int:
    args = parse_args()

    scratchpad_root = pathlib.Path(args.scratchpad_root).resolve()
    catalog_path = pathlib.Path(args.catalog).resolve()
    study_binary = pathlib.Path(args.study_binary).resolve()
    executable = pathlib.Path(args.executable).resolve()
    persistent_executable = pathlib.Path(args.persistent_executable).resolve()

    if not study_binary.is_file():
        raise FileNotFoundError(f"Study binary not found: {study_binary}")
    if not executable.is_file():
        raise FileNotFoundError(f"CLI executable not found: {executable}")
    if not persistent_executable.is_file():
        raise FileNotFoundError(f"Persistent benchmark executable not found: {persistent_executable}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Study catalog not found: {catalog_path}")

    scratchpad_root.mkdir(parents=True, exist_ok=True)
    reports_root = scratchpad_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    datasets_manifest_path = prepare_datasets(scratchpad_root, args.force_datasets)
    stage_runtime_assets(scratchpad_root)
    dataset_manifest = json.loads(datasets_manifest_path.read_text(encoding="utf-8"))

    initialize_study_db(study_binary, scratchpad_root, catalog_path, args.force)
    verify_role_assignments(
        scratchpad_root,
        {
            "calibration": list(dataset_manifest["calibration_datasets"]),
            "smoke": list(dataset_manifest["smoke_datasets"]),
            "fidelity_validation": list(dataset_manifest["validation_datasets"]),
            "fidelity_test": list(dataset_manifest["test_datasets"]),
            "stress_test": list(dataset_manifest["stress_datasets"]),
        },
    )

    print("Running smoke and development evaluation for the bounded CPU catalog.")
    for quantization in QUANTIZATIONS:
        for dataset_name in [*SMOKE_DATASETS, *dataset_manifest["validation_datasets"]]:
            run_study(study_binary, scratchpad_root, quantization, dataset_name)

    validation_summary_path = summarize_role(
        scratchpad_root,
        "fidelity_validation",
        reports_root / "attempt4-validation-summary",
    )
    validation_complete = complete_quantizations(
        validation_summary_path,
        list(dataset_manifest["validation_datasets"]),
    )

    print("Benchmarking complete development candidates on persistent CPU runtime and RSS.")
    validation_runtime_csv_path = benchmark_cpu(
        scratchpad_root,
        executable,
        persistent_executable,
        mode="persistent",
        quantizations=validation_complete,
        output_prefix=reports_root / "attempt4-validation-cpu-persistent",
    )

    report_output_prefix = reports_root / "attempt4-cpu-summary"
    intermediate_report_path = build_report(
        datasets_manifest_path,
        validation_summary_path,
        validation_runtime_csv_path,
        report_output_prefix,
    )
    intermediate_report = json.loads(intermediate_report_path.read_text(encoding="utf-8"))
    locked_quantizations = list(intermediate_report["locked_quantizations"])

    manifest: dict[str, object] = {
        "scratchpad_root": str(scratchpad_root),
        "catalog": str(catalog_path),
        "validation_complete_quantizations": validation_complete,
        "locked_quantizations": locked_quantizations,
        "datasets_manifest": str(datasets_manifest_path),
        "validation_summary": str(validation_summary_path),
        "validation_runtime_csv": str(validation_runtime_csv_path),
        "report_artifacts": report_artifact_paths(report_output_prefix),
    }

    if not args.skip_test:
        print("Running locked final-test and stress-test evaluation.")
        for dataset_name in dataset_manifest["test_datasets"]:
            for quantization in locked_quantizations:
                run_study(study_binary, scratchpad_root, quantization, dataset_name)
        for dataset_name in dataset_manifest["stress_datasets"]:
            for quantization in locked_quantizations:
                run_study(study_binary, scratchpad_root, quantization, dataset_name)

        test_summary_path = summarize_role(
            scratchpad_root,
            "fidelity_test",
            reports_root / "attempt4-test-summary",
        )
        stress_summary_path = summarize_role(
            scratchpad_root,
            "stress_test",
            reports_root / "attempt4-stress-summary",
        )
        cold_benchmark_csv_path = benchmark_cpu(
            scratchpad_root,
            executable,
            persistent_executable,
            mode="coldstart",
            quantizations=locked_quantizations,
            output_prefix=reports_root / "attempt4-test-cpu-cold",
        )

        final_report_path = build_report(
            datasets_manifest_path,
            validation_summary_path,
            validation_runtime_csv_path,
            report_output_prefix,
            test_summary_path=test_summary_path,
            stress_summary_path=stress_summary_path,
            cold_benchmark_csv_path=cold_benchmark_csv_path,
        )
        manifest["test_summary"] = str(test_summary_path)
        manifest["stress_summary"] = str(stress_summary_path)
        manifest["cold_benchmark_csv"] = str(cold_benchmark_csv_path)
        manifest["final_report"] = str(final_report_path)
    else:
        print("Skipping final test and stress evaluation because --skip-test was requested.")
        manifest["final_report"] = str(intermediate_report_path)

    manifest_path = reports_root / "attempt4-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
