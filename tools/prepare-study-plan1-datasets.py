#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCRATCHPAD_ROOT = REPO_ROOT / "scratchpad" / "plan1"
DEFAULT_LANGUAGES = ("en", "de", "es", "fr", "zh")

CALIBRATION_DATASET_FILENAMES = (
    "mnli-train-calibration-64-per-label.tsv",
    "xnli-de-validation-calibration-32-per-label.tsv",
    "xnli-en-validation-calibration-32-per-label.tsv",
    "xnli-es-validation-calibration-32-per-label.tsv",
    "xnli-fr-validation-calibration-32-per-label.tsv",
    "xnli-zh-validation-calibration-32-per-label.tsv",
)
SMOKE_DATASET_FILENAMES = (
    "hf-probe-set.tsv",
    "hf-core-probe.tsv",
)
PLAN0_GUARD_FILENAMES = (
    *CALIBRATION_DATASET_FILENAMES,
    "mnli-train-search-validation-skip64-64-per-label.tsv",
    "xnli-de-validation-search-validation-skip32-32-per-label.tsv",
    "xnli-en-validation-search-validation-skip32-32-per-label.tsv",
    "xnli-es-validation-search-validation-skip32-32-per-label.tsv",
    "xnli-fr-validation-search-validation-skip32-32-per-label.tsv",
    "xnli-zh-validation-search-validation-skip32-32-per-label.tsv",
    "mnli-validation_matched-200-per-label.tsv",
    "mnli-validation_mismatched-200-per-label.tsv",
    "xnli-de-test-50-per-label.tsv",
    "xnli-en-test-50-per-label.tsv",
    "xnli-es-test-50-per-label.tsv",
    "xnli-fr-test-50-per-label.tsv",
    "xnli-zh-test-50-per-label.tsv",
    *SMOKE_DATASET_FILENAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a fresh plan1 dataset pack under a dedicated scratchpad root. "
            "Copies frozen calibration/smoke assets and generates new disjoint "
            "fidelity-validation and fidelity-test windows."
        )
    )
    parser.add_argument(
        "--scratchpad-root",
        default=str(DEFAULT_SCRATCHPAD_ROOT),
        help=f"Scratchpad root directory (default: {DEFAULT_SCRATCHPAD_ROOT})",
    )
    parser.add_argument(
        "--xnli-language",
        action="append",
        dest="xnli_languages",
        help="XNLI language to include. Repeat to override the defaults.",
    )
    parser.add_argument("--mnli-validation-per-label", type=int, default=128)
    parser.add_argument("--xnli-validation-per-label", type=int, default=32)
    parser.add_argument("--mnli-validation-skip-per-label", type=int, default=128)
    parser.add_argument("--xnli-validation-skip-per-label", type=int, default=64)
    parser.add_argument("--mnli-test-per-label", type=int, default=100)
    parser.add_argument("--xnli-test-per-label", type=int, default=50)
    parser.add_argument("--mnli-test-skip-per-label", type=int, default=200)
    parser.add_argument("--xnli-test-skip-per-label", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--api-base-url", default="https://datasets-server.huggingface.co")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def tagged_output_name(
    prefix: str,
    split: str,
    per_label: int,
    tag: str,
    skip_per_label: int,
) -> str:
    parts = [prefix, split]
    if tag:
        parts.append(tag)
    if skip_per_label > 0:
        parts.append(f"skip{skip_per_label}")
    parts.append(f"{per_label}-per-label.tsv")
    return "-".join(parts)


def copy_frozen_datasets(
    dataset_root: pathlib.Path,
    force: bool,
    filenames: tuple[str, ...],
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in filenames:
        source_path = REPO_ROOT / "benchmarks" / "nli" / dataset_name
        dest_path = dataset_root / dataset_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Frozen dataset not found: {source_path}")
        if dest_path.exists() and not force:
            continue
        shutil.copyfile(source_path, dest_path)


def validation_paths(
    dataset_root: pathlib.Path,
    languages: list[str],
    mnli_per_label: int,
    xnli_per_label: int,
    mnli_skip_per_label: int,
    xnli_skip_per_label: int,
) -> list[pathlib.Path]:
    paths = [
        dataset_root
        / tagged_output_name(
            "mnli",
            "train",
            mnli_per_label,
            "plan1-search-validation",
            mnli_skip_per_label,
        )
    ]
    for language in languages:
        paths.append(
            dataset_root
            / tagged_output_name(
                f"xnli-{language}",
                "validation",
                xnli_per_label,
                "plan1-search-validation",
                xnli_skip_per_label,
            )
        )
    return paths


def test_paths(
    dataset_root: pathlib.Path,
    languages: list[str],
    mnli_per_label: int,
    xnli_per_label: int,
    mnli_skip_per_label: int,
    xnli_skip_per_label: int,
) -> list[pathlib.Path]:
    paths = []
    for split in ("validation_matched", "validation_mismatched"):
        paths.append(
            dataset_root
            / tagged_output_name(
                "mnli",
                split,
                mnli_per_label,
                "plan1-test",
                mnli_skip_per_label,
            )
        )
    for language in languages:
        paths.append(
            dataset_root
            / tagged_output_name(
                f"xnli-{language}",
                "test",
                xnli_per_label,
                "plan1-test",
                xnli_skip_per_label,
            )
        )
    return paths


def generate_validation_datasets(
    dataset_root: pathlib.Path,
    languages: list[str],
    args: argparse.Namespace,
) -> list[pathlib.Path]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "download-nli-eval-slices.py"),
        "--output-dir",
        str(dataset_root),
        "--mnli-split",
        "train",
        "--xnli-split",
        "validation",
        "--mnli-per-label",
        str(args.mnli_validation_per_label),
        "--xnli-per-label",
        str(args.xnli_validation_per_label),
        "--mnli-skip-per-label",
        str(args.mnli_validation_skip_per_label),
        "--xnli-skip-per-label",
        str(args.xnli_validation_skip_per_label),
        "--name-tag",
        "plan1-search-validation",
        "--seed",
        str(args.seed),
        "--page-size",
        str(args.page_size),
        "--api-base-url",
        args.api_base_url,
    ]
    for language in languages:
        command.extend(["--xnli-language", language])
    if args.force:
        command.append("--force")
    run_command(command)
    return validation_paths(
        dataset_root,
        languages,
        args.mnli_validation_per_label,
        args.xnli_validation_per_label,
        args.mnli_validation_skip_per_label,
        args.xnli_validation_skip_per_label,
    )


def generate_test_datasets(
    dataset_root: pathlib.Path,
    languages: list[str],
    args: argparse.Namespace,
) -> list[pathlib.Path]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "download-nli-eval-slices.py"),
        "--output-dir",
        str(dataset_root),
        "--mnli-split",
        "validation_matched",
        "--mnli-split",
        "validation_mismatched",
        "--xnli-split",
        "test",
        "--mnli-per-label",
        str(args.mnli_test_per_label),
        "--xnli-per-label",
        str(args.xnli_test_per_label),
        "--mnli-skip-per-label",
        str(args.mnli_test_skip_per_label),
        "--xnli-skip-per-label",
        str(args.xnli_test_skip_per_label),
        "--name-tag",
        "plan1-test",
        "--seed",
        str(args.seed),
        "--page-size",
        str(args.page_size),
        "--api-base-url",
        args.api_base_url,
    ]
    for language in languages:
        command.extend(["--xnli-language", language])
    if args.force:
        command.append("--force")
    run_command(command)
    return test_paths(
        dataset_root,
        languages,
        args.mnli_test_per_label,
        args.xnli_test_per_label,
        args.mnli_test_skip_per_label,
        args.xnli_test_skip_per_label,
    )


def verify_disjointness(generated_paths: list[pathlib.Path]) -> None:
    verify_tool = str(REPO_ROOT / "tools" / "verify-nli-slice-disjointness.py")
    run_command(
        [
            sys.executable,
            verify_tool,
            *[item for path in generated_paths for item in ("--tsv", str(path))],
        ]
    )

    legacy_paths = [REPO_ROOT / "benchmarks" / "nli" / name for name in PLAN0_GUARD_FILENAMES]
    for generated_path in generated_paths:
        for legacy_path in legacy_paths:
            run_command(
                [
                    sys.executable,
                    verify_tool,
                    "--tsv",
                    str(generated_path),
                    "--tsv",
                    str(legacy_path),
                ]
            )


def main() -> int:
    args = parse_args()
    scratchpad_root = pathlib.Path(args.scratchpad_root).resolve()
    dataset_root = scratchpad_root / "datasets"
    languages = args.xnli_languages or list(DEFAULT_LANGUAGES)

    copy_frozen_datasets(dataset_root, args.force, CALIBRATION_DATASET_FILENAMES)
    copy_frozen_datasets(dataset_root, args.force, SMOKE_DATASET_FILENAMES)
    generated_paths = generate_validation_datasets(dataset_root, languages, args)
    generated_paths += generate_test_datasets(dataset_root, languages, args)
    verify_disjointness(generated_paths)

    print(f"scratchpad_root: {scratchpad_root}")
    print("generated plan1 datasets:")
    for path in generated_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
