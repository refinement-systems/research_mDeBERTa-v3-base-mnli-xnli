#!/usr/bin/env python3

import argparse
import csv
import json
import pathlib
import random
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


API_BASE_URL = "https://datasets-server.huggingface.co"
DEFAULT_OUTPUT_DIR = "benchmarks/nli"
DEFAULT_MNLI_PER_LABEL = 100
DEFAULT_XNLI_PER_LABEL = 50
DEFAULT_XNLI_LANGUAGES = ("en", "de", "es", "fr", "zh")
DEFAULT_PAGE_SIZE = 100
LABELS = ("entailment", "neutral", "contradiction")


@dataclass(frozen=True)
class ExportTarget:
    dataset: str
    config: str
    split: str
    per_label: int
    output_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download compact balanced MNLI/XNLI slices from the Hugging Face "
            "datasets-server API and write them as TSV files that nli-eval "
            "can read directly."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated TSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--mnli-per-label",
        type=int,
        default=DEFAULT_MNLI_PER_LABEL,
        help=(
            "Examples per label for each MNLI validation split "
            f"(default: {DEFAULT_MNLI_PER_LABEL})"
        ),
    )
    parser.add_argument(
        "--xnli-per-label",
        type=int,
        default=DEFAULT_XNLI_PER_LABEL,
        help=(
            "Examples per label for each requested XNLI language "
            f"(default: {DEFAULT_XNLI_PER_LABEL})"
        ),
    )
    parser.add_argument(
        "--xnli-language",
        action="append",
        dest="xnli_languages",
        help=(
            "XNLI language config to fetch from the test split. "
            "Repeat the flag to request multiple languages."
        ),
    )
    parser.add_argument(
        "--skip-mnli",
        action="store_true",
        help="Do not download MNLI validation slices.",
    )
    parser.add_argument(
        "--skip-xnli",
        action="store_true",
        help="Do not download XNLI test slices.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Datasets-server page size, must be 1-100 (default: {DEFAULT_PAGE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for the final balanced slice order (default: 0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    parser.add_argument(
        "--api-base-url",
        default=API_BASE_URL,
        help=f"Datasets-server API base URL (default: {API_BASE_URL})",
    )

    args = parser.parse_args()

    if args.page_size < 1 or args.page_size > 100:
        parser.error("--page-size must be between 1 and 100")
    if args.mnli_per_label < 0:
        parser.error("--mnli-per-label must be non-negative")
    if args.xnli_per_label < 0:
        parser.error("--xnli-per-label must be non-negative")

    return args


def fetch_json(base_url: str, path: str, params: dict[str, object]) -> dict[str, object]:
    query = urllib.parse.urlencode(params, safe="/")
    url = f"{base_url}{path}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "nli-eval-slice-downloader/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"HTTP {exc.code} while fetching {url}{suffix}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc.reason}") from exc


def label_names_from_features(features: list[dict[str, object]]) -> list[str]:
    for feature in features:
        if feature.get("name") != "label":
            continue
        feature_type = feature.get("type")
        if isinstance(feature_type, dict) and feature_type.get("_type") == "ClassLabel":
            names = feature_type.get("names")
            if isinstance(names, list) and names:
                return [str(name) for name in names]
    return list(LABELS)


def normalize_label(value: object, label_names: list[str]) -> str:
    if isinstance(value, str):
        if value in LABELS:
            return value
        raise RuntimeError(f"Unsupported string label value: {value!r}")

    if isinstance(value, int):
        if 0 <= value < len(label_names):
            label = label_names[value]
            if label in LABELS:
                return label
            raise RuntimeError(f"Unsupported label name {label!r} in class label metadata")
        raise RuntimeError(f"Label index {value} is out of range for {label_names!r}")

    raise RuntimeError(f"Unsupported label value type: {value!r}")


def sanitize_text(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def output_path_for(base_dir: pathlib.Path, target: ExportTarget) -> pathlib.Path:
    return base_dir / target.output_name


def build_targets(args: argparse.Namespace) -> list[ExportTarget]:
    targets: list[ExportTarget] = []

    if not args.skip_mnli and args.mnli_per_label > 0:
        for split in ("validation_matched", "validation_mismatched"):
            targets.append(
                ExportTarget(
                    dataset="nyu-mll/multi_nli",
                    config="default",
                    split=split,
                    per_label=args.mnli_per_label,
                    output_name=f"mnli-{split}-{args.mnli_per_label}-per-label.tsv",
                )
            )

    xnli_languages = args.xnli_languages or list(DEFAULT_XNLI_LANGUAGES)
    if not args.skip_xnli and args.xnli_per_label > 0:
        for language in xnli_languages:
            targets.append(
                ExportTarget(
                    dataset="facebook/xnli",
                    config=language,
                    split="test",
                    per_label=args.xnli_per_label,
                    output_name=f"xnli-{language}-test-{args.xnli_per_label}-per-label.tsv",
                )
            )

    if not targets:
        raise RuntimeError("Nothing to do; all requested benchmark counts are zero or skipped")

    return targets


def collect_balanced_examples(
    api_base_url: str,
    target: ExportTarget,
    page_size: int,
    seed: int,
) -> list[dict[str, object]]:
    kept_by_label = {label: [] for label in LABELS}
    offset = 0
    total_rows = None
    label_names: list[str] | None = None
    dataset_slug = target.dataset.replace("/", "-")

    while any(len(examples) < target.per_label for examples in kept_by_label.values()):
        payload = fetch_json(
            api_base_url,
            "/rows",
            {
                "dataset": target.dataset,
                "config": target.config,
                "split": target.split,
                "offset": offset,
                "length": page_size,
            },
        )

        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            break

        if total_rows is None:
            total_rows = int(payload.get("num_rows_total", 0))
        if label_names is None:
            features = payload.get("features")
            if not isinstance(features, list):
                raise RuntimeError(
                    f"datasets-server response for {target.dataset}/{target.config}/{target.split} "
                    "did not include feature metadata"
                )
            label_names = label_names_from_features(features)

        for item in rows:
            if not isinstance(item, dict):
                continue

            row = item.get("row")
            if not isinstance(row, dict):
                continue

            label = normalize_label(row.get("label"), label_names)
            if len(kept_by_label[label]) >= target.per_label:
                continue

            premise = sanitize_text(row.get("premise"))
            hypothesis = sanitize_text(row.get("hypothesis"))
            if not premise or not hypothesis:
                continue

            row_idx = int(item.get("row_idx", offset + len(kept_by_label[label])))
            kept_by_label[label].append(
                {
                    "id": f"{dataset_slug}-{target.config}-{target.split}-{row_idx:06d}",
                    "label": label,
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "dataset": target.dataset,
                    "config": target.config,
                    "split": target.split,
                    "row_idx": row_idx,
                }
            )

        offset += len(rows)
        if total_rows is not None and offset >= total_rows:
            break

    missing = [label for label, rows in kept_by_label.items() if len(rows) < target.per_label]
    if missing:
        counts = ", ".join(f"{label}={len(kept_by_label[label])}" for label in LABELS)
        raise RuntimeError(
            f"Could not collect {target.per_label} examples per label for "
            f"{target.dataset}/{target.config}/{target.split}; got {counts}"
        )

    examples: list[dict[str, object]] = []
    for label in LABELS:
        examples.extend(kept_by_label[label])

    random.Random(seed).shuffle(examples)
    return examples


def write_tsv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "label",
                "premise",
                "hypothesis",
                "dataset",
                "config",
                "split",
                "row_idx",
            ],
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


def counts_summary(rows: list[dict[str, object]]) -> str:
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[str(row["label"])] += 1
    return " ".join(f"{label}={counts[label]}" for label in LABELS)


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    targets = build_targets(args)

    for target in targets:
        output_path = output_path_for(output_dir, target)
        if output_path.exists() and not args.force:
            print(f"Skipping {output_path}; already exists")
            continue

        print(
            f"Fetching {target.dataset}/{target.config}/{target.split} "
            f"({target.per_label} per label)"
        )
        rows = collect_balanced_examples(
            args.api_base_url,
            target,
            page_size=args.page_size,
            seed=args.seed,
        )
        write_tsv(output_path, rows)
        print(f"Wrote {output_path} ({len(rows)} rows; {counts_summary(rows)})")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
