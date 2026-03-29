#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks/nli"
DEFAULT_LANGUAGES = ("en", "de", "es", "fr", "zh")
DEFAULT_FINAL_GATES = [
    DEFAULT_OUTPUT_DIR / "mnli-validation_matched-200-per-label.tsv",
    DEFAULT_OUTPUT_DIR / "mnli-validation_mismatched-200-per-label.tsv",
    DEFAULT_OUTPUT_DIR / "xnli-de-test-50-per-label.tsv",
    DEFAULT_OUTPUT_DIR / "xnli-en-test-50-per-label.tsv",
    DEFAULT_OUTPUT_DIR / "xnli-es-test-50-per-label.tsv",
    DEFAULT_OUTPUT_DIR / "xnli-fr-test-50-per-label.tsv",
    DEFAULT_OUTPUT_DIR / "xnli-zh-test-50-per-label.tsv",
    DEFAULT_OUTPUT_DIR / "hf-probe-set.tsv",
    DEFAULT_OUTPUT_DIR / "hf-core-probe.tsv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare disjoint attempt1 quantization datasets for calibration, search validation, "
            "and optional fine-tuning, then verify they do not overlap with the final eval gates."
        )
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--xnli-language",
        action="append",
        dest="xnli_languages",
        help="XNLI language to include. Repeat to override the defaults.",
    )
    parser.add_argument("--calibration-mnli-per-label", type=int, default=64)
    parser.add_argument("--calibration-xnli-per-label", type=int, default=32)
    parser.add_argument("--search-mnli-per-label", type=int, default=64)
    parser.add_argument("--search-xnli-per-label", type=int, default=32)
    parser.add_argument("--fine-tune-mnli-per-label", type=int, default=128)
    parser.add_argument("--fine-tune-xnli-per-label", type=int, default=32)
    parser.add_argument(
        "--skip-fine-tune",
        action="store_true",
        help="Prepare only calibration and search-validation slices.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--api-base-url", default="https://datasets-server.huggingface.co")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def tagged_output_name(prefix: str, split: str, per_label: int, tag: str, skip_per_label: int) -> str:
    parts = [prefix, split]
    if tag:
        parts.append(tag)
    if skip_per_label > 0:
        parts.append(f"skip{skip_per_label}")
    parts.append(f"{per_label}-per-label.tsv")
    return "-".join(parts)


def expected_stage_paths(
    output_dir: pathlib.Path,
    tag: str,
    mnli_per_label: int,
    xnli_per_label: int,
    mnli_skip_per_label: int,
    xnli_skip_per_label: int,
    languages: list[str],
) -> list[pathlib.Path]:
    paths = [
        output_dir / tagged_output_name("mnli", "train", mnli_per_label, tag, mnli_skip_per_label),
    ]
    for language in languages:
        paths.append(
            output_dir
            / tagged_output_name(
                f"xnli-{language}",
                "validation",
                xnli_per_label,
                tag,
                xnli_skip_per_label,
            )
        )
    return paths


def run_download_stage(
    output_dir: pathlib.Path,
    tag: str,
    mnli_per_label: int,
    xnli_per_label: int,
    mnli_skip_per_label: int,
    xnli_skip_per_label: int,
    languages: list[str],
    args: argparse.Namespace,
) -> list[pathlib.Path]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/download-nli-eval-slices.py"),
        "--output-dir",
        str(output_dir),
        "--mnli-split",
        "train",
        "--xnli-split",
        "validation",
        "--mnli-per-label",
        str(mnli_per_label),
        "--xnli-per-label",
        str(xnli_per_label),
        "--mnli-skip-per-label",
        str(mnli_skip_per_label),
        "--xnli-skip-per-label",
        str(xnli_skip_per_label),
        "--name-tag",
        tag,
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

    subprocess.run(command, check=True)
    return expected_stage_paths(
        output_dir,
        tag,
        mnli_per_label,
        xnli_per_label,
        mnli_skip_per_label,
        xnli_skip_per_label,
        languages,
    )


def run_disjointness_check(paths: list[pathlib.Path]) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/verify-nli-slice-disjointness.py"),
    ]
    for path in paths:
        command.extend(["--tsv", str(path)])
    subprocess.run(command, check=True)


def verify_generated_against_final_gates(
    generated_paths: list[pathlib.Path],
    final_gate_paths: list[pathlib.Path],
) -> None:
    run_disjointness_check(generated_paths)
    for generated_path in generated_paths:
        for final_gate_path in final_gate_paths:
            run_disjointness_check([generated_path, final_gate_path])


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    languages = args.xnli_languages or list(DEFAULT_LANGUAGES)

    calibration_paths = run_download_stage(
        output_dir=output_dir,
        tag="calibration",
        mnli_per_label=args.calibration_mnli_per_label,
        xnli_per_label=args.calibration_xnli_per_label,
        mnli_skip_per_label=0,
        xnli_skip_per_label=0,
        languages=languages,
        args=args,
    )
    search_paths = run_download_stage(
        output_dir=output_dir,
        tag="search-validation",
        mnli_per_label=args.search_mnli_per_label,
        xnli_per_label=args.search_xnli_per_label,
        mnli_skip_per_label=args.calibration_mnli_per_label,
        xnli_skip_per_label=args.calibration_xnli_per_label,
        languages=languages,
        args=args,
    )

    fine_tune_paths: list[pathlib.Path] = []
    if not args.skip_fine_tune:
        fine_tune_paths = run_download_stage(
            output_dir=output_dir,
            tag="fine-tune",
            mnli_per_label=args.fine_tune_mnli_per_label,
            xnli_per_label=args.fine_tune_xnli_per_label,
            mnli_skip_per_label=args.calibration_mnli_per_label + args.search_mnli_per_label,
            xnli_skip_per_label=args.calibration_xnli_per_label + args.search_xnli_per_label,
            languages=languages,
            args=args,
        )

    generated_paths = calibration_paths + search_paths + fine_tune_paths
    missing_gates = [path for path in DEFAULT_FINAL_GATES if not path.exists()]
    if missing_gates:
        raise RuntimeError(
            "Final eval gate files are missing: " + ", ".join(str(path) for path in missing_gates)
        )

    verify_generated_against_final_gates(generated_paths, DEFAULT_FINAL_GATES)

    print("generated attempt1 data:")
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
