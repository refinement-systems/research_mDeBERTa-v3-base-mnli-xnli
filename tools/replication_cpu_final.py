#!/usr/bin/env python3

"""Minimal, standalone replication script for the final CPU winner.

This script intentionally focuses on a *single* objective:

1. Download only the assets needed to reproduce the final CPU-winning quantization recipe.
2. Recompute that winner candidate (`nncf_accuracy_attention_only`) from the float model.
3. Evaluate the recomputed candidate versus the float reference model.

The script is written to be explicit and heavily commented so each step is easy to audit.
"""

from __future__ import annotations

import argparse
import pathlib
import shlex
import subprocess
import sys


# Resolve repository root from this file location so the script can be launched from any cwd.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# The final CPU study winner name used across the research docs/catalogs.
WINNER_NAME = "nncf_accuracy_attention_only"


def parse_args() -> argparse.Namespace:
    """Parse CLI flags.

    We keep the interface intentionally small:
    - a workspace directory for generated artifacts/logs,
    - and a force flag for deterministic re-runs.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Recompute the final CPU-study winner quantization and compare it against "
            "the float reference model."
        )
    )
    parser.add_argument(
        "--workspace",
        default=str(REPO_ROOT / "scratchpad" / "replication_cpu_final"),
        help="Directory for generated quantized model and evaluation assets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite/rebuild existing outputs where supported.",
    )
    return parser.parse_args()


def run(command: list[str], *, cwd: pathlib.Path = REPO_ROOT) -> None:
    """Run a subprocess, streaming output, and fail fast on non-zero exits.

    A tiny wrapper like this keeps the script readable while still surfacing the
    exact command lines for reproducibility.
    """

    print(f"\n+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def ensure_binaries_built() -> None:
    """Configure and build required C++ binaries.

    Quantization itself runs in Python, but the model-vs-model evaluation uses the
    local C++ evaluator (`nli-eval`) to mirror the repository's runtime path.
    """

    # Configure CMake/Ninja build tree (idempotent if already configured).
    run([str(REPO_ROOT / "tools" / "setup.sh")])

    # Build only the evaluator target we need for this script.
    run([str(REPO_ROOT / "tools" / "build.sh"), "--target", "nli-eval"])


def download_model_assets(force: bool) -> None:
    """Download the base model + tokenizer assets to the repo-standard location.

    We intentionally use the default location (`models/mdeberta`) because the C++
    evaluator auto-discovers tokenizer assets from that layout.
    """

    command = [
        str(REPO_ROOT / "tools" / "download-mdeberta-v3-base.sh"),
        "--tokenizer-assets",
    ]
    if force:
        command.append("--force")
    run(command)


def download_quantization_datasets(dataset_dir: pathlib.Path, force: bool) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Download calibration/validation slices that match the winning NNCF recipe.

    The winner's recipe uses:
    - calibration data: MNLI train + XNLI(en) validation
    - validation data (for accuracy-control): disjoint slices with skip offsets

    We recreate those two dataset groups with the same per-label sizing used in
    the cataloged winner recipe.
    """

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1) Calibration slices (64 MNLI per label, 32 XNLI-en per label).
    calibration_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "download-nli-eval-slices.py"),
        "--output-dir",
        str(dataset_dir),
        "--mnli-split",
        "train",
        "--xnli-split",
        "validation",
        "--xnli-language",
        "en",
        "--mnli-per-label",
        "64",
        "--xnli-per-label",
        "32",
        "--name-tag",
        "calibration",
    ]
    if force:
        calibration_cmd.append("--force")
    run(calibration_cmd)

    # 2) Search-validation slices for NNCF accuracy-control (disjoint via skip).
    validation_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "download-nli-eval-slices.py"),
        "--output-dir",
        str(dataset_dir),
        "--mnli-split",
        "train",
        "--xnli-split",
        "validation",
        "--xnli-language",
        "en",
        "--mnli-per-label",
        "64",
        "--xnli-per-label",
        "32",
        "--mnli-skip-per-label",
        "64",
        "--xnli-skip-per-label",
        "32",
        "--name-tag",
        "search-validation",
    ]
    if force:
        validation_cmd.append("--force")
    run(validation_cmd)

    # 3) A compact labeled evaluation slice for the model-vs-model comparison step.
    # We use MNLI matched validation with 200 examples per label.
    eval_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "download-nli-eval-slices.py"),
        "--output-dir",
        str(dataset_dir),
        "--skip-xnli",
        "--mnli-per-label",
        "200",
        "--mnli-split",
        "validation_matched",
    ]
    if force:
        eval_cmd.append("--force")
    run(eval_cmd)

    calibration_paths = [
        dataset_dir / "mnli-train-calibration-64-per-label.tsv",
        dataset_dir / "xnli-en-validation-calibration-32-per-label.tsv",
    ]
    validation_paths = [
        dataset_dir / "mnli-train-search-validation-skip64-64-per-label.tsv",
        dataset_dir / "xnli-en-validation-search-validation-skip32-32-per-label.tsv",
    ]

    return calibration_paths, validation_paths


def quantize_winner_model(
    output_model_path: pathlib.Path,
    calibration_paths: list[pathlib.Path],
    validation_paths: list[pathlib.Path],
    force: bool,
) -> None:
    """Run the cataloged NNCF accuracy-control recipe for the final CPU winner."""

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "quantize-onnx-nncf.py"),
        "--input",
        str(REPO_ROOT / "models" / "mdeberta" / "onnx" / "model.onnx"),
        "--output",
        str(output_model_path),
        "--mode",
        "accuracy-control",
        "--metric",
        "gold_accuracy",
        "--preset",
        "mixed",
        "--subset-size",
        "300",
        "--ignored-scope-family",
        "attention_only",
        "--max-drop",
        "0.01",
        "--preprocess",
        "--fast-bias-correction",
        # Make the script standalone by allowing it to bootstrap missing Python deps.
        "--install-deps",
    ]

    for tsv in calibration_paths:
        command.extend(["--calibration-tsv", str(tsv)])
    for tsv in validation_paths:
        command.extend(["--validation-tsv", str(tsv)])
    if force:
        command.append("--force")

    run(command)


def evaluate_vs_reference(quantized_model_path: pathlib.Path, eval_tsv: pathlib.Path) -> None:
    """Compare the recomputed winner against the float reference model.

    `nli-eval` prints:
    - primary/compare accuracy when labels are present,
    - model agreement,
    - sample disagreements.

    We pass the recomputed winner as the *primary* model and float as compare model
    so the output is directly interpreted as "winner relative to reference".
    """

    run(
        [
            str(REPO_ROOT / "builddir" / "nli-eval"),
            "-b",
            "cpu",
            "--model",
            str(quantized_model_path),
            "--compare-model",
            str(REPO_ROOT / "models" / "mdeberta" / "onnx" / "model.onnx"),
            str(eval_tsv),
        ]
    )


def main() -> int:
    args = parse_args()
    workspace = pathlib.Path(args.workspace).resolve()
    datasets_dir = workspace / "datasets"
    generated_dir = workspace / "generated"
    quantized_model = generated_dir / f"{WINNER_NAME}.onnx"

    # Keep the flow linear and explicit so it is easy to reproduce manually.
    ensure_binaries_built()
    download_model_assets(force=args.force)
    calibration_tsvs, validation_tsvs = download_quantization_datasets(datasets_dir, force=args.force)
    quantize_winner_model(quantized_model, calibration_tsvs, validation_tsvs, force=args.force)

    # Evaluate on a labeled held-out slice.
    eval_tsv = datasets_dir / "mnli-validation_matched-200-per-label.tsv"
    evaluate_vs_reference(quantized_model, eval_tsv)

    print("\nDone.")
    print(f"workspace: {workspace}")
    print(f"winner_model: {quantized_model}")
    print(f"eval_tsv: {eval_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
