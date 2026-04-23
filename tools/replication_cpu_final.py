#!/usr/bin/env python3

"""Minimal, standalone replication script for the final CPU winner.

This script intentionally focuses on a *single* objective:

1. Prepare the attempt4 dataset workspace needed by the current winner-only flow.
2. Recompute the final CPU winner candidate (`nncf_accuracy_attention_only`) from
   the float model.
3. Evaluate the recomputed candidate versus the float reference model.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import email.utils
import json
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WINNER_NAME = "nncf_accuracy_attention_only"

API_BASE_URL = "https://datasets-server.huggingface.co"
DEFAULT_PAGE_SIZE = 100
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 2.0
DEFAULT_MAX_RETRIES = 10
DEFAULT_INITIAL_BACKOFF_SECONDS = 10.0
DEFAULT_MAX_BACKOFF_SECONDS = 300.0
HANS_EVALUATION_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
DEFAULT_XNLI_LANGUAGES = (
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
)
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
COMPATIBILITY_DATASET_FILENAMES = (
    "mnli-train-search-validation-skip64-64-per-label.tsv",
    "xnli-en-validation-search-validation-skip32-32-per-label.tsv",
    "mnli-validation_matched-200-per-label.tsv",
)
WINNER_CALIBRATION_DATASET_FILENAMES = (
    "mnli-train-calibration-64-per-label.tsv",
    "xnli-en-validation-calibration-32-per-label.tsv",
)
WINNER_VALIDATION_DATASET_FILENAMES = (
    "mnli-train-search-validation-skip64-64-per-label.tsv",
    "xnli-en-validation-search-validation-skip32-32-per-label.tsv",
)
WINNER_EVAL_DATASET_FILENAME = "mnli-validation_matched-200-per-label.tsv"
TERNARY_LABELS = ("entailment", "neutral", "contradiction")
HANS_LABELS = ("entailment", "non-entailment")

DEFAULT_ANLI_DATASET = "facebook/anli"
DEFAULT_WANLI_DATASET = "alisawuffles/WANLI"
DEFAULT_HANS_DATASET = "jhu-cogsci/hans"


@dataclass(frozen=True)
class ExportTarget:
    dataset: str
    config: str
    split: str
    output_name: str
    benchmark: str
    label_kind: str
    source_kind: str = "datasets_server"


class DatasetsServerClient:
    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_seconds: float,
        min_request_interval_seconds: float,
        max_retries: int,
        initial_backoff_seconds: float,
        max_backoff_seconds: float,
    ) -> None:
        self.base_url = base_url
        self.request_timeout_seconds = request_timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._last_request_started_at: float | None = None

    def fetch_json(self, path: str, params: dict[str, object]) -> dict[str, object]:
        query = urllib.parse.urlencode(params, safe="/")
        url = f"{self.base_url}{path}?{query}"
        raw = self.fetch_bytes_url(url)
        return json.loads(raw.decode("utf-8", errors="replace"))

    def fetch_bytes_url(self, url: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "attempt4-cpu-dataset-prep/1.0",
            },
        )

        for attempt in range(self.max_retries + 1):
            self._wait_for_request_slot()
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace").strip()
                if self._should_retry_http_error(exc.code) and attempt < self.max_retries:
                    delay = self._compute_retry_delay_seconds(exc, attempt)
                    print(
                        f"Retrying HTTP {exc.code} for {url} in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(f"HTTP {exc.code} while fetching {url}{suffix}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    delay = self._compute_retry_delay_seconds(None, attempt)
                    print(
                        f"Retrying network error for {url} in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries}): {exc.reason}",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Failed to fetch {url}: {exc.reason}") from exc

        raise RuntimeError(f"Unreachable retry state while fetching {url}")

    def fetch_text_url(self, url: str) -> str:
        return self.fetch_bytes_url(url).decode("utf-8")

    def _wait_for_request_slot(self) -> None:
        if self._last_request_started_at is not None:
            elapsed = time.monotonic() - self._last_request_started_at
            remaining = self.min_request_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started_at = time.monotonic()

    @staticmethod
    def _should_retry_http_error(status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code < 600

    def _compute_retry_delay_seconds(
        self,
        http_error: urllib.error.HTTPError | None,
        attempt: int,
    ) -> float:
        retry_after_seconds = parse_retry_after_seconds(http_error) if http_error else None
        if retry_after_seconds is not None:
            return min(max(retry_after_seconds, self.initial_backoff_seconds), self.max_backoff_seconds)
        exponential = self.initial_backoff_seconds * (2 ** attempt)
        return min(exponential, self.max_backoff_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the final CPU replication datasets, recompute the winning "
            "quantization recipe, and compare it against the float reference model."
        )
    )
    parser.add_argument(
        "--workspace",
        default=str(REPO_ROOT / "scratchpad" / "replication_cpu_final"),
        help="Directory for generated quantized model, datasets, manifests, and logs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite/rebuild existing outputs where supported.",
    )
    return parser.parse_args()


def run(command: list[str], *, cwd: pathlib.Path = REPO_ROOT) -> None:
    print(f"\n+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def ensure_binaries_built() -> None:
    run([str(REPO_ROOT / "tools" / "setup.sh")])
    run([str(REPO_ROOT / "tools" / "build.sh"), "--target", "nli-eval"])


def download_model_assets(force: bool) -> None:
    command = [
        str(REPO_ROOT / "tools" / "download-mdeberta-v3-base.sh"),
        "--tokenizer-assets",
    ]
    if force:
        command.append("--force")
    run(command)


def parse_retry_after_seconds(http_error: urllib.error.HTTPError | None) -> float | None:
    if http_error is None or http_error.headers is None:
        return None
    header_value = http_error.headers.get("Retry-After")
    if not header_value:
        return None

    try:
        return max(float(header_value), 0.0)
    except ValueError:
        pass

    try:
        retry_at = email.utils.parsedate_to_datetime(header_value)
    except (TypeError, ValueError, IndexError):
        return None
    if retry_at is None:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    return max((retry_at - now).total_seconds(), 0.0)


def sanitize_text(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def copy_frozen_datasets(
    dataset_root: pathlib.Path,
    force: bool,
    filenames: tuple[str, ...],
) -> list[str]:
    copied: list[str] = []
    dataset_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in filenames:
        source_path = REPO_ROOT / "benchmarks" / "nli" / dataset_name
        dest_path = dataset_root / dataset_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Frozen dataset not found: {source_path}")
        if dest_path.exists() and not force:
            copied.append(dest_path.name)
            continue
        shutil.copyfile(source_path, dest_path)
        copied.append(dest_path.name)
    return copied


def has_required_tsv_columns(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    fieldnames = set(reader.fieldnames or [])
    return {"premise", "hypothesis"}.issubset(fieldnames)


def label_names_from_features(
    features: list[dict[str, object]],
    label_field_name: str,
) -> list[str]:
    for feature in features:
        if feature.get("name") != label_field_name:
            continue
        feature_type = feature.get("type")
        if isinstance(feature_type, dict) and feature_type.get("_type") == "ClassLabel":
            names = feature_type.get("names")
            if isinstance(names, list) and names:
                return [str(name) for name in names]
    return []


def select_field_name(row: dict[str, object], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in row:
            return name
    raise RuntimeError(f"Could not find any of {candidates!r} in row keys {sorted(row.keys())!r}")


def normalize_string_label(value: str, label_kind: str) -> str:
    normalized = value.strip().lower()
    if label_kind == "ternary":
        mapping = {
            "entailment": "entailment",
            "entails": "entailment",
            "e": "entailment",
            "neutral": "neutral",
            "n": "neutral",
            "contradiction": "contradiction",
            "contradicts": "contradiction",
            "c": "contradiction",
        }
    else:
        mapping = {
            "entailment": "entailment",
            "entails": "entailment",
            "e": "entailment",
            "non-entailment": "non-entailment",
            "non_entailment": "non-entailment",
            "nonentailment": "non-entailment",
            "not_entailment": "non-entailment",
            "not-entailment": "non-entailment",
        }

    if normalized in mapping:
        return mapping[normalized]
    raise RuntimeError(f"Unsupported {label_kind} label value: {value!r}")


def normalize_label(value: object, label_names: list[str], label_kind: str) -> str:
    if isinstance(value, int):
        if not label_names:
            raise RuntimeError(f"Integer label {value} needs class-label metadata")
        if 0 <= value < len(label_names):
            return normalize_string_label(label_names[value], label_kind)
        raise RuntimeError(f"Label index {value} is out of range for {label_names!r}")

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() and label_names:
            return normalize_label(int(stripped), label_names, label_kind)
        return normalize_string_label(stripped, label_kind)

    raise RuntimeError(f"Unsupported label value type: {value!r}")


def language_text_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a language map, got: {value!r}")

    languages = value.get("language")
    translations = value.get("translation")
    if isinstance(languages, list) and isinstance(translations, list) and len(languages) == len(translations):
        mapped = {
            sanitize_text(language): sanitize_text(translation)
            for language, translation in zip(languages, translations)
            if sanitize_text(language) and sanitize_text(translation)
        }
        if mapped:
            return mapped

    mapped: dict[str, str] = {}
    for key, item in value.items():
        language = sanitize_text(key)
        if not language:
            continue
        if isinstance(item, dict) and "translation" in item:
            text = sanitize_text(item.get("translation"))
        else:
            text = sanitize_text(item)
        if text:
            mapped[language] = text

    if mapped:
        return mapped
    raise RuntimeError(f"Could not decode language map: {value!r}")


def export_target(
    client: DatasetsServerClient,
    output_dir: pathlib.Path,
    target: ExportTarget,
    *,
    page_size: int,
    force: bool,
) -> dict[str, object]:
    if target.source_kind == "hans_raw":
        return export_hans_raw_target(
            client,
            output_dir,
            target,
            force=force,
        )

    output_path = output_dir / target.output_name
    if not force and has_required_tsv_columns(output_path):
        return {
            "name": target.output_name,
            "path": str(output_path),
            "dataset": target.dataset,
            "config": target.config,
            "split": target.split,
            "benchmark": target.benchmark,
            "label_kind": target.label_kind,
            "skipped": True,
        }

    print(f"Fetching {target.dataset}/{target.config}/{target.split} -> {target.output_name}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    total_rows = None
    row_count = 0
    label_counts: dict[str, int] = {}
    label_field_name: str | None = None
    premise_field_name: str | None = None
    hypothesis_field_name: str | None = None
    id_field_name: str | None = None
    label_names: list[str] = []
    dataset_slug = target.dataset.replace("/", "-")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark",
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

        while True:
            payload = client.fetch_json(
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

            for item_index, item in enumerate(rows):
                if not isinstance(item, dict):
                    continue
                row = item.get("row")
                if not isinstance(row, dict):
                    continue

                if premise_field_name is None:
                    premise_field_name = select_field_name(row, ("premise", "sentence1"))
                if hypothesis_field_name is None:
                    hypothesis_field_name = select_field_name(row, ("hypothesis", "sentence2"))
                if label_field_name is None:
                    label_field_name = select_field_name(row, ("label", "gold_label", "gold", "answer"))
                    features = payload.get("features")
                    if isinstance(features, list):
                        label_names = label_names_from_features(features, label_field_name)
                if id_field_name is None:
                    try:
                        id_field_name = select_field_name(row, ("uid", "id", "pairID"))
                    except RuntimeError:
                        id_field_name = ""

                premise = sanitize_text(row.get(premise_field_name))
                hypothesis = sanitize_text(row.get(hypothesis_field_name))
                if not premise or not hypothesis:
                    continue

                label = normalize_label(row.get(label_field_name), label_names, target.label_kind)
                row_idx = int(item.get("row_idx", offset + row_count))
                example_id = sanitize_text(row.get(id_field_name)) if id_field_name else ""
                if not example_id:
                    example_id = f"{dataset_slug}-{target.config}-{target.split}-{row_idx:06d}"

                writer.writerow(
                    {
                        "benchmark": target.benchmark,
                        "id": example_id,
                        "label": label,
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "dataset": target.dataset,
                        "config": target.config,
                        "split": target.split,
                        "row_idx": row_idx,
                    }
                )
                label_counts[label] = label_counts.get(label, 0) + 1
                row_count += 1

            offset += len(rows)
            if total_rows is not None and offset >= total_rows:
                break

    return {
        "name": target.output_name,
        "path": str(output_path),
        "dataset": target.dataset,
        "config": target.config,
        "split": target.split,
        "benchmark": target.benchmark,
        "label_kind": target.label_kind,
        "skipped": False,
        "row_count": row_count,
        "label_counts": label_counts,
    }


def export_hans_raw_target(
    client: DatasetsServerClient,
    output_dir: pathlib.Path,
    target: ExportTarget,
    *,
    force: bool,
) -> dict[str, object]:
    output_path = output_dir / target.output_name
    if not force and has_required_tsv_columns(output_path):
        return {
            "name": target.output_name,
            "path": str(output_path),
            "dataset": target.dataset,
            "config": target.config,
            "split": target.split,
            "benchmark": target.benchmark,
            "label_kind": target.label_kind,
            "skipped": True,
        }

    print(f"Fetching raw HANS evaluation set -> {target.output_name}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = client.fetch_text_url(HANS_EVALUATION_URL)
    label_counts: dict[str, int] = {label: 0 for label in HANS_LABELS}
    row_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark",
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

        for line_number, line in enumerate(text.splitlines()):
            if line_number == 0:
                continue
            fields = line.strip().split("\t")
            if len(fields) < 11:
                continue
            if fields[0] == "-":
                continue

            label = normalize_string_label(fields[0], "binary")
            premise = sanitize_text(fields[5])
            hypothesis = sanitize_text(fields[6])
            if not premise or not hypothesis:
                continue

            writer.writerow(
                {
                    "benchmark": target.benchmark,
                    "id": f"hans-validation-{line_number:06d}",
                    "label": label,
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "dataset": target.dataset,
                    "config": target.config,
                    "split": target.split,
                    "row_idx": line_number,
                }
            )
            label_counts[label] += 1
            row_count += 1

    return {
        "name": target.output_name,
        "path": str(output_path),
        "dataset": target.dataset,
        "config": target.config,
        "split": target.split,
        "benchmark": target.benchmark,
        "label_kind": target.label_kind,
        "skipped": False,
        "row_count": row_count,
        "label_counts": label_counts,
    }


def export_xnli_all_languages_test(
    client: DatasetsServerClient,
    output_dir: pathlib.Path,
    languages: tuple[str, ...],
    *,
    page_size: int,
    force: bool,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        language: output_dir / f"xnli-{language}-test-attempt4-test.tsv"
        for language in languages
    }
    if not force and all(has_required_tsv_columns(path) for path in output_paths.values()):
        return [
            {
                "name": output_paths[language].name,
                "path": str(output_paths[language]),
                "dataset": "facebook/xnli",
                "config": language,
                "split": "test",
                "benchmark": f"xnli-{language}-test",
                "label_kind": "ternary",
                "skipped": True,
            }
            for language in languages
        ]

    print(
        "Fetching facebook/xnli/all_languages/test -> "
        + ", ".join(path.name for path in output_paths.values()),
        flush=True,
    )
    label_names: list[str] = []
    row_counts = {language: 0 for language in languages}
    label_counts = {language: {} for language in languages}
    handles: dict[str, object] = {}
    writers: dict[str, csv.DictWriter] = {}

    try:
        for language, output_path in output_paths.items():
            handle = output_path.open("w", encoding="utf-8", newline="")
            handles[language] = handle
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "benchmark",
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
            writers[language] = writer

        offset = 0
        total_rows = None
        while True:
            payload = client.fetch_json(
                "/rows",
                {
                    "dataset": "facebook/xnli",
                    "config": "all_languages",
                    "split": "test",
                    "offset": offset,
                    "length": page_size,
                },
            )
            rows = payload.get("rows")
            if not isinstance(rows, list) or not rows:
                break

            if total_rows is None:
                total_rows = int(payload.get("num_rows_total", 0))
                features = payload.get("features")
                if isinstance(features, list):
                    label_names = label_names_from_features(features, "label")

            for item_index, item in enumerate(rows):
                if not isinstance(item, dict):
                    continue
                row = item.get("row")
                if not isinstance(row, dict):
                    continue

                premise_map = language_text_map(row.get("premise"))
                hypothesis_map = language_text_map(row.get("hypothesis"))
                label = normalize_label(row.get("label"), label_names, "ternary")
                row_idx = int(item.get("row_idx", offset + item_index))

                for language in languages:
                    premise = premise_map.get(language, "")
                    hypothesis = hypothesis_map.get(language, "")
                    if not premise or not hypothesis:
                        raise RuntimeError(
                            f"Missing XNLI translation for language {language} at row {row_idx}"
                        )
                    writers[language].writerow(
                        {
                            "benchmark": f"xnli-{language}-test",
                            "id": f"xnli-{language}-test-{row_idx:06d}",
                            "label": label,
                            "premise": premise,
                            "hypothesis": hypothesis,
                            "dataset": "facebook/xnli",
                            "config": language,
                            "split": "test",
                            "row_idx": row_idx,
                        }
                    )
                    label_counts[language][label] = label_counts[language].get(label, 0) + 1
                    row_counts[language] += 1

            offset += len(rows)
            if total_rows is not None and offset >= total_rows:
                break
    finally:
        for handle in handles.values():
            handle.close()

    return [
        {
            "name": output_paths[language].name,
            "path": str(output_paths[language]),
            "dataset": "facebook/xnli",
            "config": language,
            "split": "test",
            "benchmark": f"xnli-{language}-test",
            "label_kind": "ternary",
            "skipped": False,
            "row_count": row_counts[language],
            "label_counts": label_counts[language],
        }
        for language in languages
    ]


def build_targets(
    *,
    anli_dataset: str = DEFAULT_ANLI_DATASET,
    wanli_dataset: str = DEFAULT_WANLI_DATASET,
    hans_dataset: str = DEFAULT_HANS_DATASET,
) -> dict[str, list[ExportTarget]]:
    validation_targets = [
        ExportTarget(
            dataset="nyu-mll/multi_nli",
            config="default",
            split="validation_matched",
            output_name="mnli-validation_matched-attempt4-dev.tsv",
            benchmark="mnli-validation_matched",
            label_kind="ternary",
        ),
        ExportTarget(
            dataset="nyu-mll/multi_nli",
            config="default",
            split="validation_mismatched",
            output_name="mnli-validation_mismatched-attempt4-dev.tsv",
            benchmark="mnli-validation_mismatched",
            label_kind="ternary",
        ),
    ]

    for round_name in ("r1", "r2", "r3"):
        validation_targets.append(
            ExportTarget(
                dataset=anli_dataset,
                config="plain_text",
                split=f"dev_{round_name}",
                output_name=f"anli-{round_name}-dev-attempt4-dev.tsv",
                benchmark=f"anli-{round_name}-dev",
                label_kind="ternary",
            )
        )

    test_targets: list[ExportTarget] = []
    for round_name in ("r1", "r2", "r3"):
        test_targets.append(
            ExportTarget(
                dataset=anli_dataset,
                config="plain_text",
                split=f"test_{round_name}",
                output_name=f"anli-{round_name}-test-attempt4-test.tsv",
                benchmark=f"anli-{round_name}-test",
                label_kind="ternary",
            )
        )

    stress_targets = [
        ExportTarget(
            dataset=hans_dataset,
            config="plain_text",
            split="validation",
            output_name="hans-evaluation-attempt4-stress-test.tsv",
            benchmark="hans-evaluation",
            label_kind="binary",
            source_kind="hans_raw",
        ),
        ExportTarget(
            dataset=wanli_dataset,
            config="default",
            split="test",
            output_name="wanli-test-attempt4-stress-test.tsv",
            benchmark="wanli-test",
            label_kind="ternary",
        ),
    ]

    return {
        "validation": validation_targets,
        "test": test_targets,
        "stress": stress_targets,
    }


def verify_disjointness(dataset_root: pathlib.Path, dataset_names: list[str]) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "verify-nli-slice-disjointness.py"),
    ]
    for dataset_name in dataset_names:
        command.extend(["--tsv", str(dataset_root / dataset_name)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="", file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, command)
    print(f"verified disjointness across {len(dataset_names)} attempt4 dataset slices", flush=True)


def prepare_attempt4_datasets(
    workspace: pathlib.Path,
    force: bool,
    *,
    client: DatasetsServerClient | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    xnli_languages: tuple[str, ...] = DEFAULT_XNLI_LANGUAGES,
) -> pathlib.Path:
    scratchpad_root = workspace.resolve()
    dataset_root = scratchpad_root / "datasets"
    reports_root = scratchpad_root / "reports"
    dataset_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    copied_calibration = copy_frozen_datasets(dataset_root, force, CALIBRATION_DATASET_FILENAMES)
    copied_smoke = copy_frozen_datasets(dataset_root, force, SMOKE_DATASET_FILENAMES)
    copy_frozen_datasets(dataset_root, force, COMPATIBILITY_DATASET_FILENAMES)

    dataset_client = client or DatasetsServerClient(
        API_BASE_URL,
        request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        min_request_interval_seconds=DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        initial_backoff_seconds=DEFAULT_INITIAL_BACKOFF_SECONDS,
        max_backoff_seconds=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    targets = build_targets()

    exported: dict[str, list[dict[str, object]]] = {"validation": [], "test": [], "stress": []}
    exported["test"].extend(
        export_xnli_all_languages_test(
            dataset_client,
            dataset_root,
            xnli_languages,
            page_size=page_size,
            force=force,
        )
    )
    for role, role_targets in targets.items():
        for target in role_targets:
            exported[role].append(
                export_target(
                    dataset_client,
                    dataset_root,
                    target,
                    page_size=page_size,
                    force=force,
                )
            )

    verify_disjointness(
        dataset_root,
        [
            *copied_calibration,
            *[item["name"] for item in exported["validation"]],
            *[item["name"] for item in exported["test"]],
            *[item["name"] for item in exported["stress"]],
        ],
    )

    manifest = {
        "scratchpad_root": str(scratchpad_root),
        "calibration_datasets": copied_calibration,
        "smoke_datasets": copied_smoke,
        "validation_datasets": [item["name"] for item in exported["validation"]],
        "test_datasets": [item["name"] for item in exported["test"]],
        "stress_datasets": [item["name"] for item in exported["stress"]],
        "exports": exported,
    }
    manifest_path = reports_root / "attempt4-datasets-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)
    return manifest_path


def winner_dataset_paths(workspace: pathlib.Path) -> tuple[list[pathlib.Path], list[pathlib.Path], pathlib.Path]:
    dataset_dir = workspace / "datasets"
    calibration_paths = [dataset_dir / name for name in WINNER_CALIBRATION_DATASET_FILENAMES]
    validation_paths = [dataset_dir / name for name in WINNER_VALIDATION_DATASET_FILENAMES]
    eval_path = dataset_dir / WINNER_EVAL_DATASET_FILENAME
    return calibration_paths, validation_paths, eval_path


def quantize_winner_model(
    output_model_path: pathlib.Path,
    calibration_paths: list[pathlib.Path],
    validation_paths: list[pathlib.Path],
    force: bool,
) -> None:
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
    generated_dir = workspace / "generated"
    quantized_model = generated_dir / f"{WINNER_NAME}.onnx"

    ensure_binaries_built()
    download_model_assets(force=args.force)
    manifest_path = prepare_attempt4_datasets(workspace, args.force)
    calibration_tsvs, validation_tsvs, eval_tsv = winner_dataset_paths(workspace)
    quantize_winner_model(quantized_model, calibration_tsvs, validation_tsvs, force=args.force)
    evaluate_vs_reference(quantized_model, eval_tsv)

    print("\nDone.")
    print(f"workspace: {workspace}")
    print(f"manifest: {manifest_path}")
    print(f"winner_model: {quantized_model}")
    print(f"eval_tsv: {eval_tsv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
