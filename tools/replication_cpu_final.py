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
import hashlib
import json
import pathlib
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Sequence


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
DEFAULT_STUDY_CATALOG_PATH = (
    REPO_ROOT / "research" / "attempt4_cpu-focus" / "study_quantization_catalog.json"
)
STUDY_REQUIRED_KEYS = {
    "name",
    "generator_program",
    "generator_args_json",
    "source_artifact_name",
    "output_relpath",
    "calibration_role",
    "validation_role",
    "allowed_backends",
    "notes",
}
STUDY_RUNTIME_ASSETS = (
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
)
STUDY_STATUS_MATERIALIZED = "materialized"
STUDY_STATUS_MISSING = "missing"
STUDY_STATUS_INVALID = "invalid"
STUDY_STATUS_FAILED = "failed"
STUDY_RUN_STATUS_PENDING = "pending"
STUDY_RUN_STATUS_RUNNING = "running"
STUDY_RUN_STATUS_COMPLETED = "completed"
STUDY_RUN_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class ExportTarget:
    dataset: str
    config: str
    split: str
    output_name: str
    benchmark: str
    label_kind: str
    source_kind: str = "datasets_server"


@dataclass(frozen=True)
class StudyCatalogEntry:
    name: str
    generator_program: str
    generator_args: tuple[str, ...]
    source_artifact_name: str | None
    output_relpath: str
    calibration_role: str | None
    validation_role: str | None
    allowed_backends: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class MaterializationInfo:
    status: str
    sha256: str
    size_bytes: int
    exists: bool


@dataclass(frozen=True)
class DatasetRowRecord:
    id: int
    premise: str
    hypothesis: str


@dataclass(frozen=True)
class DatasetExample:
    source_row_id: str
    label: str | None
    premise: str
    hypothesis: str


@dataclass(frozen=True)
class QuantizationRecord:
    quantization_id: int
    artifact_id: int
    name: str
    generator_program: str
    generator_args: tuple[str, ...]
    source_artifact_name: str | None
    calibration_role: str | None
    validation_role: str | None
    allowed_backends: tuple[str, ...]
    artifact_path: pathlib.Path
    artifact_sha256: str | None
    artifact_status: str
    stdout_log_path: pathlib.Path
    stderr_log_path: pathlib.Path


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


class CpuOrtPredictor:
    def __init__(self, model_path: pathlib.Path, tokenizer_root: pathlib.Path, backend: str) -> None:
        backend_name = normalize_backend_name(backend)
        if backend_name != "cpu":
            raise RuntimeError(f"Only CPU execution is implemented in this slice, got: {backend}")

        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CPU evaluation requires numpy, onnxruntime, transformers, and sentencepiece in the active environment"
            ) from exc

        self._np = np
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_root),
            local_files_only=True,
            use_fast=True,
        )
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self._session.get_inputs()}

    def predict_logits(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        encoded = self._tokenizer(
            [premise],
            [hypothesis],
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        feed = {
            name: value
            for name, value in encoded.items()
            if name in self._input_names
        }
        if not feed:
            raise RuntimeError("ONNX session inputs did not match tokenizer outputs")

        outputs = self._session.run(None, feed)
        if not outputs:
            raise RuntimeError("ONNX session returned no outputs")

        logits = self._np.asarray(outputs[0], dtype=self._np.float32).reshape(-1)
        if logits.size < 3:
            raise RuntimeError(f"Expected three logits, got {logits.size}")
        return (float(logits[0]), float(logits[1]), float(logits[2]))


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
    handles: dict[str, Any] = {}
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


def current_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_absolute_path(value: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(value).expanduser().resolve()


def ensure_study_directory_layout(scratchpad_root: pathlib.Path) -> pathlib.Path:
    for path in (
        scratchpad_root,
        scratchpad_root / "datasets",
        scratchpad_root / "models" / "mdeberta",
        scratchpad_root / "models" / "mdeberta" / "onnx",
        scratchpad_root / "candidates",
        scratchpad_root / "logs",
        scratchpad_root / "logs" / "generation",
        scratchpad_root / "logs" / "evaluation",
        scratchpad_root / "reports",
    ):
        path.mkdir(parents=True, exist_ok=True)
    db_path = scratchpad_root / "db.sqlite3"
    db_path.touch(exist_ok=True)
    return db_path


def remove_database_files(db_path: pathlib.Path) -> None:
    for candidate in (db_path, pathlib.Path(str(db_path) + "-wal"), pathlib.Path(str(db_path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def stage_runtime_assets(scratchpad_root: pathlib.Path) -> None:
    source_root = REPO_ROOT / "models" / "mdeberta"
    dest_root = scratchpad_root / "models" / "mdeberta"
    dest_root.mkdir(parents=True, exist_ok=True)

    for asset_name in STUDY_RUNTIME_ASSETS:
        source_path = source_root / asset_name
        dest_path = dest_root / asset_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing runtime asset: {source_path}")
        if dest_path.is_file():
            source_hash = compute_file_sha256_hex(source_path)
            dest_hash = compute_file_sha256_hex(dest_path)
            if source_hash == dest_hash:
                continue
        shutil.copyfile(source_path, dest_path)


def open_study_connection(db_path: pathlib.Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def create_study_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS dataset (
          id INTEGER PRIMARY KEY,
          name TEXT NOT NULL UNIQUE,
          role TEXT NOT NULL,
          source_path TEXT NOT NULL,
          source_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dataset_row (
          id INTEGER PRIMARY KEY,
          dataset_id INTEGER NOT NULL,
          row_idx INTEGER NOT NULL,
          source_row_id TEXT,
          label TEXT,
          premise TEXT NOT NULL,
          hypothesis TEXT NOT NULL,
          FOREIGN KEY(dataset_id) REFERENCES dataset(id) ON DELETE CASCADE,
          UNIQUE(dataset_id, row_idx)
        );
        CREATE TABLE IF NOT EXISTS quantization (
          id INTEGER PRIMARY KEY,
          name TEXT NOT NULL UNIQUE,
          generator_program TEXT NOT NULL,
          generator_args_json TEXT NOT NULL,
          source_artifact_name TEXT,
          output_relpath TEXT NOT NULL,
          calibration_role TEXT,
          validation_role TEXT,
          allowed_backends_json TEXT NOT NULL,
          notes TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS artifact (
          id INTEGER PRIMARY KEY,
          quantization_id INTEGER NOT NULL UNIQUE,
          path TEXT NOT NULL,
          artifact_sha256 TEXT,
          size_bytes INTEGER NOT NULL,
          status TEXT NOT NULL,
          stdout_log_path TEXT NOT NULL,
          stderr_log_path TEXT NOT NULL,
          materialized_at TEXT,
          FOREIGN KEY(quantization_id) REFERENCES quantization(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS backend (
          id INTEGER PRIMARY KEY,
          name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS evaluation_run (
          id INTEGER PRIMARY KEY,
          artifact_id INTEGER NOT NULL,
          backend_id INTEGER NOT NULL,
          dataset_id INTEGER NOT NULL,
          command_json TEXT NOT NULL,
          status TEXT NOT NULL,
          started_at TEXT,
          finished_at TEXT,
          FOREIGN KEY(artifact_id) REFERENCES artifact(id) ON DELETE CASCADE,
          FOREIGN KEY(backend_id) REFERENCES backend(id),
          FOREIGN KEY(dataset_id) REFERENCES dataset(id) ON DELETE CASCADE,
          UNIQUE(artifact_id, backend_id, dataset_id)
        );
        CREATE TABLE IF NOT EXISTS evaluation (
          id INTEGER PRIMARY KEY,
          evaluation_run_id INTEGER NOT NULL,
          dataset_row_id INTEGER NOT NULL,
          entailment_logit REAL NOT NULL,
          neutral_logit REAL NOT NULL,
          contradiction_logit REAL NOT NULL,
          predicted_label TEXT NOT NULL,
          FOREIGN KEY(evaluation_run_id) REFERENCES evaluation_run(id) ON DELETE CASCADE,
          FOREIGN KEY(dataset_row_id) REFERENCES dataset_row(id) ON DELETE CASCADE,
          UNIQUE(evaluation_run_id, dataset_row_id)
        );
        """
    )
    connection.commit()


def seed_backends(connection: sqlite3.Connection) -> None:
    connection.executemany(
        "INSERT INTO backend (name) VALUES (?) ON CONFLICT(name) DO NOTHING",
        [("CPU",), ("CoreML",)],
    )
    connection.commit()


def load_study_catalog(path: pathlib.Path | None = None) -> list[StudyCatalogEntry]:
    catalog_path = pathlib.Path(path) if path else DEFAULT_STUDY_CATALOG_PATH
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Catalog must be a JSON array: {catalog_path}")

    seen_names: set[str] = set()
    validated: list[StudyCatalogEntry] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(f"Catalog entry {index} must be a JSON object")

        missing = STUDY_REQUIRED_KEYS - set(item.keys())
        if missing:
            raise RuntimeError(
                f"Catalog entry {index} is missing keys: {', '.join(sorted(missing))}"
            )

        name = item["name"]
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"Catalog entry {index} must define a non-empty string name")
        if name in seen_names:
            raise RuntimeError(f"Catalog entry names must be unique: {name}")
        seen_names.add(name)

        args_json = item["generator_args_json"]
        if not isinstance(args_json, list) or not all(isinstance(entry, str) for entry in args_json):
            raise RuntimeError(f"Catalog entry {name} must use a string array for generator_args_json")

        allowed_backends = item["allowed_backends"]
        if not isinstance(allowed_backends, list) or not all(
            isinstance(entry, str) for entry in allowed_backends
        ):
            raise RuntimeError(f"Catalog entry {name} must use a string array for allowed_backends")

        validated.append(
            StudyCatalogEntry(
                name=name,
                generator_program=str(item["generator_program"]),
                generator_args=tuple(args_json),
                source_artifact_name=None
                if item["source_artifact_name"] is None
                else str(item["source_artifact_name"]),
                output_relpath=str(item["output_relpath"]),
                calibration_role=None
                if item["calibration_role"] is None
                else str(item["calibration_role"]),
                validation_role=None
                if item["validation_role"] is None
                else str(item["validation_role"]),
                allowed_backends=tuple(str(entry) for entry in allowed_backends),
                notes=str(item["notes"]),
            )
        )

    return validated


def compute_file_sha256_hex(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_dataset_files(dataset_root: pathlib.Path) -> list[pathlib.Path]:
    if not dataset_root.exists():
        return []
    paths = [path for path in dataset_root.iterdir() if path.is_file() and path.suffix == ".tsv"]
    return sorted(paths)


def classify_dataset_role(dataset_name: str) -> str:
    if dataset_name in {"hf-probe-set.tsv", "hf-core-probe.tsv"}:
        return "smoke"
    if "stress-test" in dataset_name:
        return "stress_test"
    if "calibration" in dataset_name:
        return "calibration"
    if "attempt4-dev" in dataset_name:
        return "fidelity_validation"
    if (
        "validation_matched" in dataset_name
        or "validation_mismatched" in dataset_name
        or "attempt4-test" in dataset_name
        or "-test-" in dataset_name
    ):
        return "fidelity_test"
    if "search-validation" in dataset_name or "-validation-" in dataset_name:
        return "fidelity_validation"
    raise RuntimeError(f"Could not infer dataset role from filename: {dataset_name}")


def read_nli_eval_examples(path: pathlib.Path) -> list[DatasetExample]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"Dataset is missing a TSV header: {path}")
        fieldnames = set(reader.fieldnames)
        if "premise" not in fieldnames or "hypothesis" not in fieldnames:
            raise RuntimeError(f"Dataset must contain premise and hypothesis columns: {path}")

        examples: list[DatasetExample] = []
        for row_index, row in enumerate(reader):
            premise = sanitize_text(row.get("premise"))
            hypothesis = sanitize_text(row.get("hypothesis"))
            if not premise or not hypothesis:
                continue
            source_row_id = sanitize_text(row.get("id") or row.get("source_row_id") or str(row_index))
            label_value = sanitize_text(row.get("label"))
            examples.append(
                DatasetExample(
                    source_row_id=source_row_id,
                    label=label_value or None,
                    premise=premise,
                    hypothesis=hypothesis,
                )
            )
        return examples


def import_datasets(connection: sqlite3.Connection, dataset_root: pathlib.Path) -> None:
    for path in discover_dataset_files(dataset_root):
        dataset_name = path.name
        role = classify_dataset_role(dataset_name)
        source_sha256 = compute_file_sha256_hex(path)
        examples = read_nli_eval_examples(path)

        row = connection.execute(
            "SELECT id, source_sha256 FROM dataset WHERE name = ?",
            (dataset_name,),
        ).fetchone()
        if row is None:
            cursor = connection.execute(
                "INSERT INTO dataset (name, role, source_path, source_sha256) VALUES (?, ?, ?, ?)",
                (dataset_name, role, str(path), source_sha256),
            )
            dataset_id = int(cursor.lastrowid)
        else:
            dataset_id = int(row[0])
            existing_sha256 = str(row[1])
            if existing_sha256 != source_sha256:
                raise RuntimeError(f"Dataset name collision with different content: {dataset_name}")
            existing_row_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM dataset_row WHERE dataset_id = ?",
                    (dataset_id,),
                ).fetchone()[0]
            )
            if existing_row_count == len(examples):
                continue
            connection.execute("DELETE FROM dataset_row WHERE dataset_id = ?", (dataset_id,))

        for row_index, example in enumerate(examples):
            connection.execute(
                """
                INSERT INTO dataset_row
                    (dataset_id, row_idx, source_row_id, label, premise, hypothesis)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    row_index,
                    example.source_row_id,
                    example.label,
                    example.premise,
                    example.hypothesis,
                ),
            )
    connection.commit()


def inspect_artifact(
    artifact_path: pathlib.Path,
    expected_sha256: str | None,
) -> MaterializationInfo:
    if not artifact_path.exists():
        return MaterializationInfo(STUDY_STATUS_MISSING, "", 0, False)

    size_bytes = artifact_path.stat().st_size
    if size_bytes == 0:
        return MaterializationInfo(STUDY_STATUS_INVALID, "", 0, False)

    sha256 = compute_file_sha256_hex(artifact_path)
    if expected_sha256 and expected_sha256 != sha256:
        return MaterializationInfo(STUDY_STATUS_INVALID, sha256, size_bytes, True)
    return MaterializationInfo(STUDY_STATUS_MATERIALIZED, sha256, size_bytes, True)


def upsert_catalog_entries(
    connection: sqlite3.Connection,
    entries: Sequence[StudyCatalogEntry],
    scratchpad_root: pathlib.Path,
) -> None:
    for entry in entries:
        connection.execute(
            """
            INSERT INTO quantization
                (name, generator_program, generator_args_json, source_artifact_name, output_relpath,
                 calibration_role, validation_role, allowed_backends_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                generator_program=excluded.generator_program,
                generator_args_json=excluded.generator_args_json,
                source_artifact_name=excluded.source_artifact_name,
                output_relpath=excluded.output_relpath,
                calibration_role=excluded.calibration_role,
                validation_role=excluded.validation_role,
                allowed_backends_json=excluded.allowed_backends_json,
                notes=excluded.notes
            """,
            (
                entry.name,
                entry.generator_program,
                json.dumps(list(entry.generator_args)),
                entry.source_artifact_name,
                entry.output_relpath,
                entry.calibration_role,
                entry.validation_role,
                json.dumps(list(entry.allowed_backends)),
                entry.notes,
            ),
        )
        quantization_id = int(
            connection.execute(
                "SELECT id FROM quantization WHERE name = ?",
                (entry.name,),
            ).fetchone()[0]
        )

        artifact_path = (scratchpad_root / entry.output_relpath).resolve()
        stdout_log_path = scratchpad_root / "logs" / "generation" / f"{entry.name}.stdout.log"
        stderr_log_path = scratchpad_root / "logs" / "generation" / f"{entry.name}.stderr.log"
        artifact_state = inspect_artifact(artifact_path, None)
        materialized_at = (
            current_timestamp() if artifact_state.status == STUDY_STATUS_MATERIALIZED else None
        )

        connection.execute(
            """
            INSERT INTO artifact
                (quantization_id, path, artifact_sha256, size_bytes, status, stdout_log_path, stderr_log_path, materialized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(quantization_id) DO UPDATE SET
                path=excluded.path,
                artifact_sha256=excluded.artifact_sha256,
                size_bytes=excluded.size_bytes,
                status=excluded.status,
                stdout_log_path=excluded.stdout_log_path,
                stderr_log_path=excluded.stderr_log_path,
                materialized_at=excluded.materialized_at
            """,
            (
                quantization_id,
                str(artifact_path),
                artifact_state.sha256 or None,
                artifact_state.size_bytes,
                artifact_state.status,
                str(stdout_log_path),
                str(stderr_log_path),
                materialized_at,
            ),
        )
    connection.commit()


def initialize_study_workspace(
    scratchpad_root: pathlib.Path | str,
    *,
    catalog_path: pathlib.Path | str = DEFAULT_STUDY_CATALOG_PATH,
    force: bool = False,
) -> pathlib.Path:
    scratchpad = resolve_absolute_path(scratchpad_root)
    catalog = resolve_absolute_path(catalog_path)
    db_path = scratchpad / "db.sqlite3"

    ensure_study_directory_layout(scratchpad)
    stage_runtime_assets(scratchpad)
    if force:
        remove_database_files(db_path)
        ensure_study_directory_layout(scratchpad)

    connection = open_study_connection(db_path)
    try:
        create_study_schema(connection)
        seed_backends(connection)
        upsert_catalog_entries(connection, load_study_catalog(catalog), scratchpad)
        import_datasets(connection, scratchpad / "datasets")
    finally:
        connection.close()

    return db_path


def resolve_path_argument(argument: str) -> str:
    if not argument or argument.startswith("-"):
        return argument
    path = pathlib.Path(argument)
    if path.is_absolute():
        return str(path)
    candidate = (REPO_ROOT / path).resolve()
    if candidate.exists():
        return str(candidate)
    return argument


def resolve_program_argument(argument: str) -> str:
    if argument == "python3":
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python.resolve())
    return resolve_path_argument(argument)


def dataset_paths_for_role(connection: sqlite3.Connection, role: str) -> list[str]:
    rows = connection.execute(
        "SELECT source_path FROM dataset WHERE role = ? ORDER BY name",
        (role,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def expand_argument_token(
    token: str,
    scratchpad_root: pathlib.Path,
    source_path: str,
    dest_path: str,
    calibration_paths: Sequence[str],
    validation_paths: Sequence[str],
) -> list[str]:
    expanded = (
        token.replace("${SRC}", source_path)
        .replace("${DEST}", dest_path)
        .replace("${SCRATCHPAD}", str(scratchpad_root))
    )
    if "${CALIBRATION_TSVS}" in token:
        if not calibration_paths:
            raise RuntimeError(f"Generator placeholder expands to an empty dataset list: {token}")
        return [expanded.replace("${CALIBRATION_TSVS}", value) for value in calibration_paths]
    if "${VALIDATION_TSVS}" in token:
        if not validation_paths:
            raise RuntimeError(f"Generator placeholder expands to an empty dataset list: {token}")
        return [expanded.replace("${VALIDATION_TSVS}", value) for value in validation_paths]
    return [expanded]


def parse_generation_metadata(stdout_log_path: pathlib.Path) -> tuple[bool | None, str]:
    if not stdout_log_path.is_file():
        return None, ""

    text = stdout_log_path.read_text(encoding="utf-8", errors="replace")
    smooth_quant_disabled: bool | None = None
    retry_reason = ""

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if "smooth_quant_disabled" in payload:
                value = payload["smooth_quant_disabled"]
                smooth_quant_disabled = None if value is None else bool(value)
            if "retry_reason" in payload and payload["retry_reason"] is not None:
                retry_reason = str(payload["retry_reason"])
            return smooth_quant_disabled, retry_reason

    for line in text.splitlines():
        stripped = line.strip()
        if "smooth_quant_disabled=" in stripped:
            smooth_quant_disabled = stripped.split("=", 1)[1] == "True"
        if line.startswith("  retry_reason: "):
            retry_reason = line.split(": ", 1)[1].strip()
    return smooth_quant_disabled, retry_reason


def is_inline_stage_study_artifact(
    generator_program: str,
    generator_args: Sequence[str],
) -> bool:
    if normalize_path_string(resolve_program_argument(generator_program)).endswith("python"):
        return bool(generator_args) and generator_args[0].endswith("tools/stage-study-artifact.py")
    return generator_program == "python3" and bool(generator_args) and generator_args[0].endswith(
        "tools/stage-study-artifact.py"
    )


def normalize_path_string(value: str) -> str:
    return pathlib.Path(value).as_posix()


def parse_option_argument(arguments: Sequence[str], option_name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option_name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(option_name + "="):
            return argument.split("=", 1)[1]
    return None


def materialize_stage_study_artifact(
    arguments: Sequence[str],
    stdout_log_path: pathlib.Path,
    stderr_log_path: pathlib.Path,
) -> None:
    source_value = parse_option_argument(arguments, "--src")
    dest_value = parse_option_argument(arguments, "--dest")
    source_stdout_log_value = parse_option_argument(arguments, "--source-stdout-log")
    if not source_value or not dest_value:
        raise RuntimeError("Inline stage-study-artifact requires --src and --dest")

    source_path = pathlib.Path(resolve_path_argument(source_value)).resolve()
    dest_path = pathlib.Path(dest_value).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source artifact not found: {source_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dest_path)

    smooth_quant_disabled = None
    retry_reason = ""
    if source_stdout_log_value:
        source_stdout_log = pathlib.Path(resolve_path_argument(source_stdout_log_value)).resolve()
        smooth_quant_disabled, retry_reason = parse_generation_metadata(source_stdout_log)

    payload = {
        "source": str(source_path),
        "dest": str(dest_path),
        "size_bytes": dest_path.stat().st_size,
        "smooth_quant_disabled": smooth_quant_disabled,
        "retry_reason": retry_reason,
    }
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_log_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    stderr_log_path.write_text("", encoding="utf-8")


def run_generator_process(
    program: str,
    arguments: Sequence[str],
    stdout_log_path: pathlib.Path,
    stderr_log_path: pathlib.Path,
) -> int:
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [program, *arguments],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    stdout_log_path.write_text(completed.stdout, encoding="utf-8")
    stderr_log_path.write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def load_quantization_record(connection: sqlite3.Connection, quantization_name: str) -> QuantizationRecord:
    row = connection.execute(
        """
        SELECT
          q.id, q.name, q.generator_program, q.generator_args_json, q.source_artifact_name,
          q.calibration_role, q.validation_role, q.allowed_backends_json,
          a.id, a.path, a.artifact_sha256, a.status, a.stdout_log_path, a.stderr_log_path
        FROM quantization q
        JOIN artifact a ON a.quantization_id = q.id
        WHERE q.name = ?
        """,
        (quantization_name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Unknown quantization: {quantization_name}")
    return QuantizationRecord(
        quantization_id=int(row[0]),
        artifact_id=int(row[8]),
        name=str(row[1]),
        generator_program=str(row[2]),
        generator_args=tuple(json.loads(str(row[3]))),
        source_artifact_name=None if row[4] is None else str(row[4]),
        calibration_role=None if row[5] is None else str(row[5]),
        validation_role=None if row[6] is None else str(row[6]),
        allowed_backends=tuple(json.loads(str(row[7]))),
        artifact_path=pathlib.Path(str(row[9])),
        artifact_sha256=None if row[10] is None else str(row[10]),
        artifact_status=str(row[11]),
        stdout_log_path=pathlib.Path(str(row[12])),
        stderr_log_path=pathlib.Path(str(row[13])),
    )


def update_artifact_state(
    connection: sqlite3.Connection,
    artifact_id: int,
    info: MaterializationInfo,
    stdout_log_path: pathlib.Path,
    stderr_log_path: pathlib.Path,
) -> None:
    materialized_at = current_timestamp() if info.status == STUDY_STATUS_MATERIALIZED else None
    connection.execute(
        """
        UPDATE artifact
        SET artifact_sha256 = ?, size_bytes = ?, status = ?, stdout_log_path = ?, stderr_log_path = ?, materialized_at = ?
        WHERE id = ?
        """,
        (
            info.sha256 or None,
            info.size_bytes,
            info.status,
            str(stdout_log_path),
            str(stderr_log_path),
            materialized_at,
            artifact_id,
        ),
    )
    connection.commit()


def normalize_backend_name(backend: str) -> str:
    value = backend.strip().lower()
    if value not in {"cpu", "coreml"}:
        raise RuntimeError(f"Unknown backend: {backend}")
    return value


def ensure_backend_allowed(quantization: QuantizationRecord, backend: str) -> None:
    backend_name = normalize_backend_name(backend)
    if backend_name not in quantization.allowed_backends:
        raise RuntimeError(
            f"Quantization '{quantization.name}' is not allowed on backend '{backend_name}'"
        )


def ensure_artifact_materialized(
    connection: sqlite3.Connection,
    scratchpad_root: pathlib.Path,
    quantization: QuantizationRecord,
    *,
    force_regenerate: bool,
    in_progress: set[str] | None = None,
) -> str:
    if in_progress is None:
        in_progress = set()
    if quantization.name in in_progress:
        raise RuntimeError(f"Detected quantization generation cycle at: {quantization.name}")
    in_progress.add(quantization.name)

    try:
        current_state = inspect_artifact(quantization.artifact_path, quantization.artifact_sha256)
        if not force_regenerate and current_state.status == STUDY_STATUS_MATERIALIZED:
            update_artifact_state(
                connection,
                quantization.artifact_id,
                current_state,
                quantization.stdout_log_path,
                quantization.stderr_log_path,
            )
            return str(quantization.artifact_path)

        source_path = ""
        if quantization.source_artifact_name:
            source_quantization = load_quantization_record(connection, quantization.source_artifact_name)
            source_path = ensure_artifact_materialized(
                connection,
                scratchpad_root,
                source_quantization,
                force_regenerate=False,
                in_progress=in_progress,
            )

        if not quantization.generator_program:
            failure_status = (
                STUDY_STATUS_INVALID
                if current_state.status == STUDY_STATUS_INVALID
                else STUDY_STATUS_MISSING
            )
            update_artifact_state(
                connection,
                quantization.artifact_id,
                MaterializationInfo(failure_status, current_state.sha256, current_state.size_bytes, current_state.exists),
                quantization.stdout_log_path,
                quantization.stderr_log_path,
            )
            raise RuntimeError(
                f"Artifact is unavailable and has no generator command: {quantization.name}"
            )

        calibration_paths = (
            dataset_paths_for_role(connection, quantization.calibration_role)
            if quantization.calibration_role
            else []
        )
        validation_paths = (
            dataset_paths_for_role(connection, quantization.validation_role)
            if quantization.validation_role
            else []
        )

        expanded_args: list[str] = []
        for token in quantization.generator_args:
            expanded_args.extend(
                expand_argument_token(
                    token,
                    scratchpad_root,
                    source_path,
                    str(quantization.artifact_path),
                    calibration_paths,
                    validation_paths,
                )
            )

        quantization.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if is_inline_stage_study_artifact(quantization.generator_program, quantization.generator_args):
            materialize_stage_study_artifact(
                [resolve_path_argument(arg) for arg in expanded_args],
                quantization.stdout_log_path,
                quantization.stderr_log_path,
            )
            exit_code = 0
        else:
            program = resolve_program_argument(quantization.generator_program)
            exit_code = run_generator_process(
                program,
                [resolve_path_argument(arg) for arg in expanded_args],
                quantization.stdout_log_path,
                quantization.stderr_log_path,
            )

        if exit_code != 0:
            update_artifact_state(
                connection,
                quantization.artifact_id,
                MaterializationInfo(STUDY_STATUS_FAILED, "", 0, False),
                quantization.stdout_log_path,
                quantization.stderr_log_path,
            )
            raise RuntimeError(
                f"Generator command failed for {quantization.name} with exit code {exit_code}"
            )

        updated_state = inspect_artifact(quantization.artifact_path, None)
        update_artifact_state(
            connection,
            quantization.artifact_id,
            updated_state,
            quantization.stdout_log_path,
            quantization.stderr_log_path,
        )
        if updated_state.status != STUDY_STATUS_MATERIALIZED:
            raise RuntimeError(f"Generator did not produce a valid artifact for {quantization.name}")

        return str(quantization.artifact_path)
    finally:
        in_progress.remove(quantization.name)


def require_backend_id(connection: sqlite3.Connection, backend: str) -> int:
    backend_name = "CPU" if normalize_backend_name(backend) == "cpu" else "CoreML"
    row = connection.execute(
        "SELECT id FROM backend WHERE name = ?",
        (backend_name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Unknown backend: {backend_name}")
    return int(row[0])


def require_dataset_id(connection: sqlite3.Connection, dataset_name: str) -> int:
    row = connection.execute(
        "SELECT id FROM dataset WHERE name = ?",
        (dataset_name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Unknown dataset: {dataset_name}")
    return int(row[0])


def ensure_evaluation_run(
    connection: sqlite3.Connection,
    artifact_id: int,
    backend_id: int,
    dataset_id: int,
    command_json: dict[str, object],
    *,
    force_rerun: bool,
) -> int:
    row = connection.execute(
        """
        SELECT id FROM evaluation_run
        WHERE artifact_id = ? AND backend_id = ? AND dataset_id = ?
        """,
        (artifact_id, backend_id, dataset_id),
    ).fetchone()
    if row is None:
        cursor = connection.execute(
            """
            INSERT INTO evaluation_run
                (artifact_id, backend_id, dataset_id, command_json, status, started_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                backend_id,
                dataset_id,
                json.dumps(command_json),
                STUDY_RUN_STATUS_PENDING,
                None,
                None,
            ),
        )
        run_id = int(cursor.lastrowid)
    else:
        run_id = int(row[0])

    if force_rerun:
        connection.execute("DELETE FROM evaluation WHERE evaluation_run_id = ?", (run_id,))

    connection.execute(
        """
        UPDATE evaluation_run
        SET command_json = ?, status = ?, started_at = ?, finished_at = NULL
        WHERE id = ?
        """,
        (json.dumps(command_json), STUDY_RUN_STATUS_RUNNING, current_timestamp(), run_id),
    )
    connection.commit()
    return run_id


def mark_evaluation_run_status(connection: sqlite3.Connection, run_id: int, status: str) -> None:
    connection.execute(
        "UPDATE evaluation_run SET status = ?, finished_at = ? WHERE id = ?",
        (status, current_timestamp(), run_id),
    )
    connection.commit()


def missing_dataset_rows(
    connection: sqlite3.Connection,
    dataset_id: int,
    evaluation_run_id: int,
) -> list[DatasetRowRecord]:
    rows = connection.execute(
        """
        SELECT dr.id, dr.premise, dr.hypothesis
        FROM dataset_row dr
        LEFT JOIN evaluation e
            ON e.dataset_row_id = dr.id AND e.evaluation_run_id = ?
        WHERE dr.dataset_id = ? AND e.id IS NULL
        ORDER BY dr.row_idx
        """,
        (evaluation_run_id, dataset_id),
    ).fetchall()
    return [
        DatasetRowRecord(id=int(row[0]), premise=str(row[1]), hypothesis=str(row[2]))
        for row in rows
    ]


def predicted_label_from_logits(logits: Sequence[float]) -> str:
    if len(logits) < 3:
        raise RuntimeError(f"Expected three logits, got {len(logits)}")
    labels = list(TERNARY_LABELS)
    winner_index = max(range(3), key=lambda index: float(logits[index]))
    return labels[winner_index]


def insert_evaluation_row(
    connection: sqlite3.Connection,
    evaluation_run_id: int,
    dataset_row_id: int,
    logits: Sequence[float],
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO evaluation
            (evaluation_run_id, dataset_row_id, entailment_logit, neutral_logit, contradiction_logit, predicted_label)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            evaluation_run_id,
            dataset_row_id,
            float(logits[0]),
            float(logits[1]),
            float(logits[2]),
            predicted_label_from_logits(logits),
        ),
    )


def study_tokenizer_root(scratchpad_root: pathlib.Path) -> pathlib.Path:
    return scratchpad_root / "models" / "mdeberta"


def _run_study_evaluation_internal(
    scratchpad_root: pathlib.Path,
    quantization_name: str,
    backend: str,
    dataset_name: str,
    *,
    force_regenerate: bool,
    force_rerun: bool,
    predictor_factory: Callable[[pathlib.Path, pathlib.Path, str], Any] | None,
    allow_reference_recursion: bool,
) -> int:
    db_path = scratchpad_root / "db.sqlite3"
    if not db_path.exists():
        raise RuntimeError(f"Study database not found: {db_path}")

    connection = open_study_connection(db_path)
    run_id = 0
    try:
        quantization = load_quantization_record(connection, quantization_name)
        ensure_backend_allowed(quantization, backend)
        if normalize_backend_name(backend) != "cpu":
            raise RuntimeError(
                f"Only CPU execution is implemented in this slice, got backend: {backend}"
            )

        if allow_reference_recursion and quantization.name != "reference":
            _run_study_evaluation_internal(
                scratchpad_root,
                "reference",
                backend,
                dataset_name,
                force_regenerate=False,
                force_rerun=False,
                predictor_factory=predictor_factory,
                allow_reference_recursion=False,
            )

        artifact_path = ensure_artifact_materialized(
            connection,
            scratchpad_root,
            quantization,
            force_regenerate=force_regenerate,
        )
        dataset_id = require_dataset_id(connection, dataset_name)
        backend_id = require_backend_id(connection, backend)
        command_json = {
            "quantization": quantization_name,
            "backend": normalize_backend_name(backend),
            "dataset": dataset_name,
            "artifact_path": artifact_path,
        }
        run_id = ensure_evaluation_run(
            connection,
            quantization.artifact_id,
            backend_id,
            dataset_id,
            command_json,
            force_rerun=force_rerun,
        )

        rows = missing_dataset_rows(connection, dataset_id, run_id)
        if rows:
            factory = predictor_factory or (lambda model_path, tokenizer_root, requested_backend: CpuOrtPredictor(model_path, tokenizer_root, requested_backend))
            predictor = factory(
                pathlib.Path(artifact_path),
                study_tokenizer_root(scratchpad_root),
                backend,
            )
            for row in rows:
                logits = predictor.predict_logits(row.premise, row.hypothesis)
                insert_evaluation_row(connection, run_id, row.id, logits)
            connection.commit()

        mark_evaluation_run_status(connection, run_id, STUDY_RUN_STATUS_COMPLETED)
        return run_id
    except Exception:
        if run_id:
            try:
                mark_evaluation_run_status(connection, run_id, STUDY_RUN_STATUS_FAILED)
            except Exception:
                pass
        raise
    finally:
        connection.close()


def run_study_evaluation(
    scratchpad_root: pathlib.Path | str,
    quantization_name: str,
    dataset_name: str,
    *,
    backend: str = "cpu",
    force_regenerate: bool = False,
    force_rerun: bool = False,
    predictor_factory: Callable[[pathlib.Path, pathlib.Path, str], Any] | None = None,
) -> int:
    scratchpad = resolve_absolute_path(scratchpad_root)
    return _run_study_evaluation_internal(
        scratchpad,
        quantization_name,
        backend,
        dataset_name,
        force_regenerate=force_regenerate,
        force_rerun=force_rerun,
        predictor_factory=predictor_factory,
        allow_reference_recursion=True,
    )


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
