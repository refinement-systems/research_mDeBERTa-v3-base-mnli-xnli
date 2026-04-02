#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sqlite3
import subprocess
import sys
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCRATCHPAD_ROOT = REPO_ROOT / "scratchpad" / "attempt3_coreml"
DEFAULT_CATALOG_PATH = REPO_ROOT / "research" / "attempt3_coreml" / "study_quantization_catalog.json"
DEFAULT_STUDY_BINARY = REPO_ROOT / "builddir" / "nli-study"
DEFAULT_LANGUAGES = ("en", "de", "es", "fr", "zh")

SMOKE_DATASETS = (
    "hf-probe-set.tsv",
    "hf-core-probe.tsv",
)


@dataclass(frozen=True)
class SummaryAggregate:
    quantization: str
    size_bytes: int
    example_count: int
    disagreement_count: int
    max_abs_logit_delta: float
    float_label_agreement: float
    mean_abs_logit_delta: float
    pareto_frontier: bool


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


def validation_dataset_names(languages: tuple[str, ...]) -> list[str]:
    names = [
        tagged_output_name(
            "mnli",
            "train",
            64,
            "attempt3-coreml-validation",
            256,
        )
    ]
    for language in languages:
        names.append(
            tagged_output_name(
                f"xnli-{language}",
                "validation",
                32,
                "attempt3-coreml-validation",
                96,
            )
        )
    return names


def test_dataset_names(languages: tuple[str, ...]) -> list[str]:
    names = []
    for split in ("validation_matched", "validation_mismatched"):
        names.append(
            tagged_output_name(
                "mnli",
                split,
                50,
                "attempt3-coreml-test",
                300,
            )
        )
    for language in languages:
        names.append(
            tagged_output_name(
                f"xnli-{language}",
                "test",
                50,
                "attempt3-coreml-test",
                100,
            )
        )
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and run the first CoreML-specific study attempt outside the sandbox. "
            "This script keeps the run narrow: float reference first, fp16 second, "
            "integer controls only when requested."
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
        "--xnli-language",
        action="append",
        dest="xnli_languages",
        help="XNLI language to include. Repeat to override the defaults.",
    )
    parser.add_argument(
        "--include-int8-controls",
        action="store_true",
        help="Also run the secondary integer controls on smoke and validation.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Stop after validation and do not run the untouched test split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force dataset regeneration and study DB recreation.",
    )
    parser.add_argument(
        "--api-base-url",
        default="https://datasets-server.huggingface.co",
        help="Datasets-server API base URL used for fresh attempt3 slices.",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_command(command: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(command))
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        check=False,
    )
    if completed.returncode != 0 and not allow_failure:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed


def prepare_scratchpad_assets(
    scratchpad_root: pathlib.Path,
    languages: tuple[str, ...],
    api_base_url: str,
    page_size: int,
    seed: int,
    force: bool,
) -> None:
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "download-study-assets.py"),
            "--scratchpad-root",
            str(scratchpad_root),
            *(["--force"] if force else []),
        ]
    )

    dataset_root = scratchpad_root / "datasets"

    validation_command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "download-nli-eval-slices.py"),
        "--output-dir",
        str(dataset_root),
        "--mnli-split",
        "train",
        "--xnli-split",
        "validation",
        "--mnli-per-label",
        "64",
        "--xnli-per-label",
        "32",
        "--mnli-skip-per-label",
        "256",
        "--xnli-skip-per-label",
        "96",
        "--name-tag",
        "attempt3-coreml-validation",
        "--seed",
        str(seed),
        "--page-size",
        str(page_size),
        "--api-base-url",
        api_base_url,
    ]
    for language in languages:
        validation_command.extend(["--xnli-language", language])
    if force:
        validation_command.append("--force")
    run_command(validation_command)

    test_command = [
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
        "50",
        "--xnli-per-label",
        "50",
        "--mnli-skip-per-label",
        "300",
        "--xnli-skip-per-label",
        "100",
        "--name-tag",
        "attempt3-coreml-test",
        "--seed",
        str(seed),
        "--page-size",
        str(page_size),
        "--api-base-url",
        api_base_url,
    ]
    for language in languages:
        test_command.extend(["--xnli-language", language])
    if force:
        test_command.append("--force")
    run_command(test_command)


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
    *,
    allow_failure: bool = False,
) -> bool:
    completed = run_command(
        [
            str(study_binary),
            "run",
            "--scratchpad-root",
            str(scratchpad_root),
            "--quantization",
            quantization,
            "--backend",
            "coreml",
            "--dataset",
            dataset_name,
        ],
        allow_failure=allow_failure,
    )
    return completed.returncode == 0


def summarize_role(
    scratchpad_root: pathlib.Path,
    role: str,
    output_prefix: pathlib.Path,
) -> pathlib.Path:
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "summarize-study-db.py"),
            "--scratchpad-root",
            str(scratchpad_root),
            "--role",
            role,
            "--backend",
            "coreml",
            "--output-prefix",
            str(output_prefix),
        ]
    )
    return output_prefix.with_suffix(".json")


def dominates(left: SummaryAggregate, right: SummaryAggregate) -> bool:
    if left.quantization == right.quantization:
        return False
    if left.size_bytes > right.size_bytes:
        return False
    if left.float_label_agreement < right.float_label_agreement:
        return False
    if left.size_bytes < right.size_bytes or left.float_label_agreement > right.float_label_agreement:
        return True
    return left.mean_abs_logit_delta < right.mean_abs_logit_delta


def aggregate_summary(
    summary_json_path: pathlib.Path,
    required_datasets: list[str],
    output_json_path: pathlib.Path,
) -> dict[str, object]:
    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    required_dataset_set = set(required_datasets)
    grouped: dict[str, dict[str, object]] = {}

    for row in rows:
        quantization = row["quantization"]
        item = grouped.setdefault(
            quantization,
            {
                "quantization": quantization,
                "size_bytes": row["size_bytes"],
                "example_count": 0,
                "disagreement_count": 0,
                "max_abs_logit_delta": 0.0,
                "_weighted_mean_sum": 0.0,
                "_datasets": set(),
            },
        )
        if item["size_bytes"] != row["size_bytes"]:
            raise RuntimeError(f"Size mismatch across rows for quantization {quantization}")
        item["example_count"] += row["example_count"]
        item["disagreement_count"] += row["disagreement_count"]
        item["max_abs_logit_delta"] = max(item["max_abs_logit_delta"], row["max_abs_logit_delta"])
        item["_weighted_mean_sum"] += row["mean_abs_logit_delta"] * row["example_count"]
        item["_datasets"].add(row["dataset"])

    aggregated: list[SummaryAggregate] = []
    for quantization, item in grouped.items():
        if item["_datasets"] != required_dataset_set:
            continue
        example_count = int(item["example_count"])
        disagreement_count = int(item["disagreement_count"])
        aggregated.append(
            SummaryAggregate(
                quantization=quantization,
                size_bytes=int(item["size_bytes"]),
                example_count=example_count,
                disagreement_count=disagreement_count,
                max_abs_logit_delta=float(item["max_abs_logit_delta"]),
                float_label_agreement=(example_count - disagreement_count) / example_count,
                mean_abs_logit_delta=float(item["_weighted_mean_sum"]) / example_count,
                pareto_frontier=False,
            )
        )

    aggregated.sort(
        key=lambda item: (
            item.size_bytes,
            -item.float_label_agreement,
            item.mean_abs_logit_delta,
            item.quantization,
        )
    )

    frontier: list[SummaryAggregate] = []
    with_frontier_flags: list[SummaryAggregate] = []
    for item in aggregated:
        is_frontier = not any(dominates(other, item) for other in aggregated)
        updated = SummaryAggregate(
            quantization=item.quantization,
            size_bytes=item.size_bytes,
            example_count=item.example_count,
            disagreement_count=item.disagreement_count,
            max_abs_logit_delta=item.max_abs_logit_delta,
            float_label_agreement=item.float_label_agreement,
            mean_abs_logit_delta=item.mean_abs_logit_delta,
            pareto_frontier=is_frontier,
        )
        with_frontier_flags.append(updated)
        if is_frontier:
            frontier.append(updated)

    output_payload = {
        "aggregated": [aggregate.__dict__ for aggregate in with_frontier_flags],
        "frontier": [aggregate.__dict__ for aggregate in frontier],
    }
    output_json_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    return output_payload


def completed_dataset_names(
    scratchpad_root: pathlib.Path,
    role: str,
) -> list[str]:
    db_path = scratchpad_root / "db.sqlite3"
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT name FROM dataset WHERE role = ? ORDER BY name",
            (role,),
        ).fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]


def main() -> int:
    args = parse_args()

    scratchpad_root = pathlib.Path(args.scratchpad_root).resolve()
    catalog_path = pathlib.Path(args.catalog).resolve()
    study_binary = pathlib.Path(args.study_binary).resolve()
    languages = tuple(args.xnli_languages or DEFAULT_LANGUAGES)

    if not study_binary.is_file():
        raise FileNotFoundError(f"Study binary not found: {study_binary}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Study catalog not found: {catalog_path}")

    validation_datasets = validation_dataset_names(languages)
    test_datasets = test_dataset_names(languages)
    scratchpad_root.mkdir(parents=True, exist_ok=True)

    prepare_scratchpad_assets(
        scratchpad_root,
        languages,
        args.api_base_url,
        args.page_size,
        args.seed,
        args.force,
    )
    initialize_study_db(study_binary, scratchpad_root, catalog_path, args.force)

    print("Running CoreML fp32 reference gates.")
    for dataset_name in [*SMOKE_DATASETS, *validation_datasets]:
        run_study(study_binary, scratchpad_root, "reference", dataset_name)

    print("Running CoreML fp16 primary candidate.")
    for dataset_name in [*SMOKE_DATASETS, *validation_datasets]:
        run_study(study_binary, scratchpad_root, "reference_fp16", dataset_name)

    int8_failures: dict[str, str] = {}
    if args.include_int8_controls:
        print("Running secondary integer controls.")
        for quantization in ("model_quantized", "dynamic_qint8_default"):
            failed = False
            for dataset_name in [*SMOKE_DATASETS, *validation_datasets]:
                if not run_study(
                    study_binary,
                    scratchpad_root,
                    quantization,
                    dataset_name,
                    allow_failure=True,
                ):
                    int8_failures[quantization] = dataset_name
                    failed = True
                    break
            if failed:
                print(
                    f"Control {quantization} failed on {int8_failures[quantization]}; "
                    "continuing because integer controls are secondary."
                )

    reports_root = scratchpad_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    validation_summary_path = summarize_role(
        scratchpad_root,
        "fidelity_validation",
        reports_root / "validation-summary",
    )
    validation_aggregate_payload = aggregate_summary(
        validation_summary_path,
        validation_datasets,
        reports_root / "validation-frontier-aggregated.json",
    )

    locked_quantizations = {
        item["quantization"] for item in validation_aggregate_payload["frontier"]
    }
    locked_quantizations.add("reference")
    locked_quantizations = set(sorted(locked_quantizations))

    manifest = {
        "scratchpad_root": str(scratchpad_root),
        "catalog": str(catalog_path),
        "languages": list(languages),
        "smoke_datasets": list(SMOKE_DATASETS),
        "validation_datasets": validation_datasets,
        "test_datasets": test_datasets,
        "int8_controls_requested": args.include_int8_controls,
        "int8_failures": int8_failures,
        "locked_quantizations": sorted(locked_quantizations),
    }

    if not args.skip_test:
        print("Running untouched CoreML test split on locked quantizations.")
        for dataset_name in test_datasets:
            run_study(study_binary, scratchpad_root, "reference", dataset_name)
        for quantization in sorted(locked_quantizations):
            if quantization == "reference":
                continue
            for dataset_name in test_datasets:
                run_study(study_binary, scratchpad_root, quantization, dataset_name)

        test_summary_path = summarize_role(
            scratchpad_root,
            "fidelity_test",
            reports_root / "test-summary",
        )
        test_aggregate_payload = aggregate_summary(
            test_summary_path,
            test_datasets,
            reports_root / "test-frontier-aggregated.json",
        )
        manifest["test_frontier"] = [item["quantization"] for item in test_aggregate_payload["frontier"]]
    else:
        print("Skipping untouched test split because --skip-test was requested.")

    manifest["available_validation_role_datasets"] = completed_dataset_names(
        scratchpad_root, "fidelity_validation"
    )
    manifest_path = reports_root / "attempt3-coreml-plan0-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"manifest: {manifest_path}")
    print("locked_quantizations:")
    for quantization in sorted(locked_quantizations):
        print(f"  - {quantization}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
