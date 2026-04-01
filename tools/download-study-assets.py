#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCRATCHPAD_ROOT = REPO_ROOT / "scratchpad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate a scratchpad workspace with downloaded model assets and reproducible "
            "calibration/validation/test datasets for the study workflow."
        )
    )
    parser.add_argument(
        "--scratchpad-root",
        default=str(DEFAULT_SCRATCHPAD_ROOT),
        help=f"Scratchpad root directory (default: {DEFAULT_SCRATCHPAD_ROOT})",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download/regeneration.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads.")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset downloads.")
    parser.add_argument(
        "--with-hf-reference-weights",
        action="store_true",
        help="Also download Hugging Face reference weights into scratchpad/models/mdeberta.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def download_models(scratchpad_root: pathlib.Path, force: bool, with_hf_reference_weights: bool) -> None:
    model_root = scratchpad_root / "models" / "mdeberta"
    command = [
        str(REPO_ROOT / "tools/download-mdeberta-v3-base.sh"),
        "--dir",
        str(model_root),
        "--tokenizer-assets",
    ]
    if with_hf_reference_weights:
        command.append("--reference-weights")
    if force:
        command.append("--force")
    run_command(command)


def download_datasets(scratchpad_root: pathlib.Path, force: bool) -> None:
    dataset_root = scratchpad_root / "datasets"

    prepare_command = [
        sys.executable,
        str(REPO_ROOT / "tools/prepare-attempt1-quantization-data.py"),
        "--output-dir",
        str(dataset_root),
        "--skip-fine-tune",
    ]
    if force:
        prepare_command.append("--force")
    run_command(prepare_command)

    final_suite_command = [
        sys.executable,
        str(REPO_ROOT / "tools/download-nli-eval-slices.py"),
        "--output-dir",
        str(dataset_root),
        "--mnli-per-label",
        "200",
        "--xnli-per-label",
        "50",
    ]
    if force:
        final_suite_command.append("--force")
    run_command(final_suite_command)


def main() -> int:
    args = parse_args()
    scratchpad_root = pathlib.Path(args.scratchpad_root).resolve()
    scratchpad_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_models:
        download_models(scratchpad_root, args.force, args.with_hf_reference_weights)
    if not args.skip_datasets:
        download_datasets(scratchpad_root, args.force)

    print(f"scratchpad_root: {scratchpad_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

