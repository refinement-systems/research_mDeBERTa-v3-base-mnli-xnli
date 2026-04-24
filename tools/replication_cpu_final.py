#!/usr/bin/env python3

"""Standalone Python runner for the bounded attempt4 CPU replication workflow.

This script is the public entrypoint for the resumable Python-only attempt4 flow:

1. Prepare the bounded attempt4 dataset pack.
2. Initialize or refresh the SQLite study workspace and runtime assets.
3. Materialize or reuse bounded catalog artifacts and evaluate them on CPU.
4. Benchmark complete candidates, build partial/final reports, and write the
   attempt4 manifest without invoking the legacy C++ study binaries.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import email.utils
import hashlib
import json
import math
import os
import pathlib
import resource
import shlex
import shutil
import sqlite3
import statistics
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
class BenchmarkExample:
    benchmark: str
    example_id: str
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


@dataclass(frozen=True)
class SummaryRow:
    dataset: str
    role: str
    backend: str
    quantization: str
    artifact_path: str
    stdout_log_path: str
    size_bytes: int
    example_count: int
    labeled_example_count: int
    correct_prediction_count: int
    gold_accuracy: float | None
    float_label_agreement: float
    mean_abs_logit_delta: float
    max_abs_logit_delta: float
    disagreement_count: int
    smooth_quant_disabled: bool | None
    retry_reason: str
    pareto_frontier: bool


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


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, math.ceil(pct * len(sorted_values)) - 1)
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def summarize_numeric(values: Sequence[float]) -> dict[str, float | None]:
    normalized = [float(value) for value in values]
    return {
        "mean": statistics.fmean(normalized) if normalized else None,
        "median": statistics.median(normalized) if normalized else None,
        "p95": percentile(normalized, 0.95),
        "min": min(normalized) if normalized else None,
        "max": max(normalized) if normalized else None,
    }


def benchmark_csv_fieldnames() -> list[str]:
    return [
        "candidate",
        "backend",
        "mode",
        "examples",
        "file_size_bytes",
        "load_mean_ms",
        "load_median_ms",
        "load_p95_ms",
        "warm_mean_ms",
        "warm_median_ms",
        "warm_p95_ms",
        "warm_min_ms",
        "warm_max_ms",
        "resident_after_load_mean_bytes",
        "resident_after_load_median_bytes",
        "resident_after_load_p95_bytes",
        "resident_after_warmup_mean_bytes",
        "resident_after_warmup_median_bytes",
        "resident_after_warmup_p95_bytes",
        "resident_after_timed_runs_mean_bytes",
        "resident_after_timed_runs_median_bytes",
        "resident_after_timed_runs_p95_bytes",
        "peak_rss_after_load_mean_bytes",
        "peak_rss_after_load_median_bytes",
        "peak_rss_after_load_p95_bytes",
        "peak_rss_after_warmup_mean_bytes",
        "peak_rss_after_warmup_median_bytes",
        "peak_rss_after_warmup_p95_bytes",
        "peak_rss_after_timed_runs_mean_bytes",
        "peak_rss_after_timed_runs_median_bytes",
        "peak_rss_after_timed_runs_p95_bytes",
        "time_l_peak_rss_mean_bytes",
        "time_l_peak_rss_median_bytes",
        "time_l_peak_rss_p95_bytes",
    ]


def write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_benchmark_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_csv_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate": row["candidate"],
                    "backend": row["backend"],
                    "mode": row["mode"],
                    "examples": row["examples"],
                    "file_size_bytes": row["file_size_bytes"],
                    "load_mean_ms": row["load_ms"]["mean"],
                    "load_median_ms": row["load_ms"]["median"],
                    "load_p95_ms": row["load_ms"]["p95"],
                    "warm_mean_ms": row["warm_latency_ms"]["mean"],
                    "warm_median_ms": row["warm_latency_ms"]["median"],
                    "warm_p95_ms": row["warm_latency_ms"]["p95"],
                    "warm_min_ms": row["warm_latency_ms"]["min"],
                    "warm_max_ms": row["warm_latency_ms"]["max"],
                    "resident_after_load_mean_bytes": row["resident_after_load_bytes"]["mean"],
                    "resident_after_load_median_bytes": row["resident_after_load_bytes"]["median"],
                    "resident_after_load_p95_bytes": row["resident_after_load_bytes"]["p95"],
                    "resident_after_warmup_mean_bytes": row["resident_after_warmup_bytes"]["mean"],
                    "resident_after_warmup_median_bytes": row["resident_after_warmup_bytes"]["median"],
                    "resident_after_warmup_p95_bytes": row["resident_after_warmup_bytes"]["p95"],
                    "resident_after_timed_runs_mean_bytes": row["resident_after_timed_runs_bytes"]["mean"],
                    "resident_after_timed_runs_median_bytes": row["resident_after_timed_runs_bytes"]["median"],
                    "resident_after_timed_runs_p95_bytes": row["resident_after_timed_runs_bytes"]["p95"],
                    "peak_rss_after_load_mean_bytes": row["peak_rss_after_load_bytes"]["mean"],
                    "peak_rss_after_load_median_bytes": row["peak_rss_after_load_bytes"]["median"],
                    "peak_rss_after_load_p95_bytes": row["peak_rss_after_load_bytes"]["p95"],
                    "peak_rss_after_warmup_mean_bytes": row["peak_rss_after_warmup_bytes"]["mean"],
                    "peak_rss_after_warmup_median_bytes": row["peak_rss_after_warmup_bytes"]["median"],
                    "peak_rss_after_warmup_p95_bytes": row["peak_rss_after_warmup_bytes"]["p95"],
                    "peak_rss_after_timed_runs_mean_bytes": row["peak_rss_after_timed_runs_bytes"]["mean"],
                    "peak_rss_after_timed_runs_median_bytes": row["peak_rss_after_timed_runs_bytes"]["median"],
                    "peak_rss_after_timed_runs_p95_bytes": row["peak_rss_after_timed_runs_bytes"]["p95"],
                    "time_l_peak_rss_mean_bytes": row["time_l_peak_rss_bytes"]["mean"],
                    "time_l_peak_rss_median_bytes": row["time_l_peak_rss_bytes"]["median"],
                    "time_l_peak_rss_p95_bytes": row["time_l_peak_rss_bytes"]["p95"],
                }
            )


def fetch_dataset_rows(
    connection: sqlite3.Connection,
    dataset_names: list[str],
    roles: list[str],
) -> list[dict[str, object]]:
    where: list[str] = []
    params: list[object] = []
    if dataset_names:
        where.append("d.name IN ({})".format(",".join("?" for _ in dataset_names)))
        params.extend(dataset_names)
    if roles:
        where.append("d.role IN ({})".format(",".join("?" for _ in roles)))
        params.extend(roles)

    query = """
        SELECT
            d.id AS dataset_id,
            d.name AS dataset_name,
            d.role AS dataset_role
        FROM dataset d
    """
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY d.name"
    return [dict(row) for row in connection.execute(query, params)]


def fetch_backend_names(
    connection: sqlite3.Connection,
    requested_backends: list[str],
) -> list[str]:
    query = "SELECT LOWER(name) AS name FROM backend"
    rows = [str(row["name"]) for row in connection.execute(query)]
    if requested_backends:
        requested = set(requested_backends)
        return [name for name in rows if name in requested]
    return rows


def fetch_reference_rows(
    connection: sqlite3.Connection,
    dataset_id: int,
    backend_name: str,
) -> dict[int, tuple[float, float, float, str]]:
    query = """
        SELECT
            e.dataset_row_id,
            e.entailment_logit,
            e.neutral_logit,
            e.contradiction_logit,
            e.predicted_label
        FROM evaluation e
        JOIN evaluation_run er ON er.id = e.evaluation_run_id
        JOIN artifact a ON a.id = er.artifact_id
        JOIN quantization q ON q.id = a.quantization_id
        JOIN backend b ON b.id = er.backend_id
        WHERE q.name = 'reference'
          AND er.dataset_id = ?
          AND LOWER(b.name) = ?
    """
    rows: dict[int, tuple[float, float, float, str]] = {}
    for row in connection.execute(query, (dataset_id, backend_name)):
        rows[int(row["dataset_row_id"])] = (
            float(row["entailment_logit"]),
            float(row["neutral_logit"]),
            float(row["contradiction_logit"]),
            str(row["predicted_label"]),
        )
    return rows


def fetch_candidate_rows(
    connection: sqlite3.Connection,
    dataset_id: int,
    backend_name: str,
) -> list[dict[str, object]]:
    query = """
        SELECT
            q.name AS quantization_name,
            a.path AS artifact_path,
            a.stdout_log_path AS stdout_log_path,
            COALESCE(a.size_bytes, 0) AS size_bytes,
            e.dataset_row_id,
            e.entailment_logit,
            e.neutral_logit,
            e.contradiction_logit,
            e.predicted_label,
            dr.label AS gold_label
        FROM evaluation e
        JOIN evaluation_run er ON er.id = e.evaluation_run_id
        JOIN artifact a ON a.id = er.artifact_id
        JOIN quantization q ON q.id = a.quantization_id
        JOIN backend b ON b.id = er.backend_id
        JOIN dataset_row dr ON dr.id = e.dataset_row_id
        WHERE er.dataset_id = ?
          AND LOWER(b.name) = ?
        ORDER BY q.name, e.dataset_row_id
    """
    return [dict(row) for row in connection.execute(query, (dataset_id, backend_name))]


def compute_frontier_flags(rows: list[SummaryRow]) -> dict[tuple[str, str, str, str], bool]:
    grouped: dict[tuple[str, str, str], list[SummaryRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset, row.role, row.backend), []).append(row)

    flags: dict[tuple[str, str, str, str], bool] = {}
    for bucket in grouped.values():
        for row in bucket:
            dominated = False
            for other in bucket:
                if other.quantization == row.quantization:
                    continue
                if (
                    other.size_bytes <= row.size_bytes
                    and other.float_label_agreement >= row.float_label_agreement
                    and (
                        other.size_bytes < row.size_bytes
                        or other.float_label_agreement > row.float_label_agreement
                    )
                ):
                    dominated = True
                    break
            flags[(row.dataset, row.role, row.backend, row.quantization)] = not dominated
    return flags


def is_hans_dataset(dataset_name: str) -> bool:
    return "hans" in dataset_name.lower()


def collapse_hans_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized == "entailment":
        return "entailment"
    return "non-entailment"


def gold_label_match(dataset_name: str, gold_label: str, predicted_label: str) -> bool:
    if is_hans_dataset(dataset_name):
        return collapse_hans_label(gold_label) == collapse_hans_label(predicted_label)
    return gold_label == predicted_label


def summarize_dataset_backend(
    connection: sqlite3.Connection,
    dataset_name: str,
    dataset_role: str,
    dataset_id: int,
    backend_name: str,
) -> list[SummaryRow]:
    reference_rows = fetch_reference_rows(connection, dataset_id, backend_name)
    if not reference_rows:
        return []

    per_quantization: dict[str, dict[str, object]] = {}
    for row in fetch_candidate_rows(connection, dataset_id, backend_name):
        quantization_name = str(row["quantization_name"])
        candidate = per_quantization.setdefault(
            quantization_name,
            {
                "artifact_path": str(row["artifact_path"]),
                "stdout_log_path": str(row["stdout_log_path"]),
                "size_bytes": int(row["size_bytes"]),
                "examples": 0,
                "labeled_examples": 0,
                "correct_predictions": 0,
                "agreements": 0,
                "abs_delta_sum": 0.0,
                "abs_delta_count": 0,
                "max_abs_delta": 0.0,
                "disagreements": 0,
            },
        )
        dataset_row_id = int(row["dataset_row_id"])
        if dataset_row_id not in reference_rows:
            continue

        ref_entailment, ref_neutral, ref_contradiction, ref_label = reference_rows[dataset_row_id]
        cand_logits = (
            float(row["entailment_logit"]),
            float(row["neutral_logit"]),
            float(row["contradiction_logit"]),
        )
        ref_logits = (ref_entailment, ref_neutral, ref_contradiction)
        candidate["examples"] += 1
        if str(row["predicted_label"]) == ref_label:
            candidate["agreements"] += 1
        else:
            candidate["disagreements"] += 1

        gold_label = str(row["gold_label"] or "").strip()
        if gold_label:
            candidate["labeled_examples"] += 1
            if gold_label_match(dataset_name, gold_label, str(row["predicted_label"])):
                candidate["correct_predictions"] += 1

        deltas = [abs(left - right) for left, right in zip(cand_logits, ref_logits)]
        candidate["abs_delta_sum"] += sum(deltas)
        candidate["abs_delta_count"] += len(deltas)
        candidate["max_abs_delta"] = max(float(candidate["max_abs_delta"]), max(deltas))

    summary_rows: list[SummaryRow] = []
    for quantization_name, item in sorted(per_quantization.items()):
        example_count = int(item["examples"])
        if example_count == 0:
            continue
        labeled_example_count = int(item["labeled_examples"])
        correct_prediction_count = int(item["correct_predictions"])
        smooth_quant_disabled, retry_reason = parse_generation_metadata(pathlib.Path(str(item["stdout_log_path"])))
        summary_rows.append(
            SummaryRow(
                dataset=dataset_name,
                role=dataset_role,
                backend=backend_name,
                quantization=quantization_name,
                artifact_path=str(item["artifact_path"]),
                stdout_log_path=str(item["stdout_log_path"]),
                size_bytes=int(item["size_bytes"]),
                example_count=example_count,
                labeled_example_count=labeled_example_count,
                correct_prediction_count=correct_prediction_count,
                gold_accuracy=(
                    float(correct_prediction_count) / float(labeled_example_count)
                    if labeled_example_count
                    else None
                ),
                float_label_agreement=float(item["agreements"]) / float(example_count),
                mean_abs_logit_delta=float(item["abs_delta_sum"]) / float(int(item["abs_delta_count"])),
                max_abs_logit_delta=float(item["max_abs_delta"]),
                disagreement_count=int(item["disagreements"]),
                smooth_quant_disabled=smooth_quant_disabled,
                retry_reason=retry_reason,
                pareto_frontier=False,
            )
        )
    return summary_rows


def summary_rows_with_frontier(rows: list[SummaryRow]) -> list[SummaryRow]:
    flags = compute_frontier_flags(rows)
    return [
        SummaryRow(
            dataset=row.dataset,
            role=row.role,
            backend=row.backend,
            quantization=row.quantization,
            artifact_path=row.artifact_path,
            stdout_log_path=row.stdout_log_path,
            size_bytes=row.size_bytes,
            example_count=row.example_count,
            labeled_example_count=row.labeled_example_count,
            correct_prediction_count=row.correct_prediction_count,
            gold_accuracy=row.gold_accuracy,
            float_label_agreement=row.float_label_agreement,
            mean_abs_logit_delta=row.mean_abs_logit_delta,
            max_abs_logit_delta=row.max_abs_logit_delta,
            disagreement_count=row.disagreement_count,
            smooth_quant_disabled=row.smooth_quant_disabled,
            retry_reason=row.retry_reason,
            pareto_frontier=flags[(row.dataset, row.role, row.backend, row.quantization)],
        )
        for row in rows
    ]


def summary_csv_fieldnames() -> list[str]:
    return [
        "dataset",
        "role",
        "backend",
        "quantization",
        "artifact_path",
        "stdout_log_path",
        "size_bytes",
        "example_count",
        "labeled_example_count",
        "correct_prediction_count",
        "gold_accuracy",
        "float_label_agreement",
        "mean_abs_logit_delta",
        "max_abs_logit_delta",
        "disagreement_count",
        "smooth_quant_disabled",
        "retry_reason",
        "pareto_frontier",
    ]


def write_summary_outputs(rows: list[SummaryRow], output_prefix: pathlib.Path | str) -> tuple[pathlib.Path, pathlib.Path]:
    prefix = resolve_absolute_path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    finalized_rows = summary_rows_with_frontier(rows)

    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_csv_fieldnames())
        writer.writeheader()
        for row in finalized_rows:
            writer.writerow(row.__dict__)

    json_path = prefix.with_suffix(".json")
    write_json(json_path, {"rows": [row.__dict__ for row in finalized_rows]})
    return csv_path, json_path


def summarize_study_db_to_prefix(
    scratchpad_root: pathlib.Path | str,
    *,
    db_path: pathlib.Path | str | None = None,
    dataset_names: list[str] | None = None,
    roles: list[str] | None = None,
    backends: list[str] | None = None,
    output_prefix: pathlib.Path | str,
) -> list[SummaryRow]:
    scratchpad = resolve_absolute_path(scratchpad_root)
    resolved_db_path = resolve_absolute_path(db_path) if db_path else scratchpad / "db.sqlite3"
    connection = sqlite3.connect(str(resolved_db_path))
    connection.row_factory = sqlite3.Row
    try:
        selected_roles = roles or ["fidelity_validation"]
        datasets = fetch_dataset_rows(connection, dataset_names or [], selected_roles)
        backend_names = fetch_backend_names(connection, backends or [])

        summary_rows: list[SummaryRow] = []
        for dataset in datasets:
            for backend_name in backend_names:
                summary_rows.extend(
                    summarize_dataset_backend(
                        connection,
                        dataset_name=str(dataset["dataset_name"]),
                        dataset_role=str(dataset["dataset_role"]),
                        dataset_id=int(dataset["dataset_id"]),
                        backend_name=backend_name,
                    )
                )
    finally:
        connection.close()

    write_summary_outputs(summary_rows, output_prefix)
    return summary_rows


def read_json(path: pathlib.Path | str) -> Any:
    resolved = resolve_absolute_path(path)
    return json.loads(resolved.read_text(encoding="utf-8"))


def read_summary_rows(path: pathlib.Path | str) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Summary JSON does not contain a row list: {path}")
    return [dict(row) for row in rows]


def read_benchmark_rows(path: pathlib.Path | str) -> dict[str, dict[str, str]]:
    resolved = resolve_absolute_path(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {str(row["candidate"]): dict(row) for row in reader}


def optional_float(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def percent_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def ms_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f} ms"


def mib_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / (1024.0 * 1024.0):.1f} MiB"


def lower_name(value: str) -> str:
    return value.lower()


def is_mnli_dataset(name: str) -> bool:
    return lower_name(name).startswith("mnli-")


def is_anli_dev_dataset(name: str) -> bool:
    lowered = lower_name(name)
    return lowered.startswith("anli-") and "-dev-" in lowered


def is_anli_test_dataset(name: str) -> bool:
    lowered = lower_name(name)
    return lowered.startswith("anli-") and "-test-" in lowered


def is_xnli_dataset(name: str) -> bool:
    return lower_name(name).startswith("xnli-")


def xnli_language(name: str) -> str | None:
    if not is_xnli_dataset(name):
        return None
    parts = name.split("-")
    if len(parts) < 2:
        return None
    return parts[1]


def aggregate_required_rows(
    rows: list[dict[str, Any]],
    required_datasets: list[str],
) -> dict[str, dict[str, Any]]:
    required_dataset_set = set(required_datasets)
    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in required_dataset_set:
            continue
        quantization = str(row["quantization"])
        item = grouped.setdefault(
            quantization,
            {
                "quantization": quantization,
                "artifact_path": row["artifact_path"],
                "size_bytes": int(row["size_bytes"]),
                "smooth_quant_disabled": row.get("smooth_quant_disabled"),
                "retry_reason": row.get("retry_reason", ""),
                "datasets": {},
                "example_count": 0,
                "labeled_example_count": 0,
                "correct_prediction_count": 0,
                "disagreement_count": 0,
                "max_abs_logit_delta": 0.0,
                "_weighted_mean_sum": 0.0,
            },
        )
        item["datasets"][dataset] = row
        item["example_count"] += int(row["example_count"])
        item["labeled_example_count"] += int(row.get("labeled_example_count", 0))
        item["correct_prediction_count"] += int(row.get("correct_prediction_count", 0))
        item["disagreement_count"] += int(row["disagreement_count"])
        item["max_abs_logit_delta"] = max(
            float(item["max_abs_logit_delta"]),
            float(row["max_abs_logit_delta"]),
        )
        item["_weighted_mean_sum"] += float(row["mean_abs_logit_delta"]) * int(row["example_count"])

    aggregated: dict[str, dict[str, Any]] = {}
    for quantization, item in grouped.items():
        present_dataset_set = set(item["datasets"].keys())
        complete = present_dataset_set == required_dataset_set
        example_count = int(item["example_count"])
        labeled_example_count = int(item["labeled_example_count"])
        aggregated[quantization] = {
            "quantization": quantization,
            "artifact_path": item["artifact_path"],
            "size_bytes": int(item["size_bytes"]),
            "smooth_quant_disabled": item["smooth_quant_disabled"],
            "retry_reason": item["retry_reason"],
            "complete": complete,
            "dataset_names": sorted(present_dataset_set),
            "datasets": item["datasets"],
            "example_count": example_count,
            "labeled_example_count": labeled_example_count,
            "correct_prediction_count": int(item["correct_prediction_count"]),
            "gold_accuracy": (
                float(item["correct_prediction_count"]) / float(labeled_example_count)
                if labeled_example_count
                else None
            ),
            "float_label_agreement": (
                float(example_count - int(item["disagreement_count"])) / float(example_count)
                if example_count
                else None
            ),
            "mean_abs_logit_delta": (
                float(item["_weighted_mean_sum"]) / float(example_count)
                if example_count
                else None
            ),
            "max_abs_logit_delta": float(item["max_abs_logit_delta"]),
            "disagreement_count": int(item["disagreement_count"]),
        }
    return aggregated


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def validation_gate(
    quantization: str,
    aggregated_validation: dict[str, dict[str, Any]],
    runtime_rows: dict[str, dict[str, str]],
) -> tuple[bool, list[str], dict[str, float | None]]:
    candidate = aggregated_validation[quantization]
    reference = aggregated_validation.get("reference")
    reasons: list[str] = []
    metrics: dict[str, float | None] = {
        "mnli_macro_accuracy": None,
        "mnli_macro_accuracy_drop": None,
        "anli_macro_accuracy": None,
        "anli_macro_accuracy_drop": None,
        "validation_float_label_agreement": candidate["float_label_agreement"],
        "peak_rss_ratio_vs_reference": None,
    }

    if not candidate["complete"]:
        reasons.append("missing validation datasets")
        return False, reasons, metrics
    if reference is None or not reference["complete"]:
        reasons.append("reference validation summary is incomplete")
        return False, reasons, metrics

    candidate_runtime = runtime_rows.get(quantization)
    reference_runtime = runtime_rows.get("reference")
    if candidate_runtime is None:
        reasons.append("missing CPU persistent benchmark row")
    if reference_runtime is None:
        reasons.append("missing reference CPU persistent benchmark row")

    candidate_dataset_rows = candidate["datasets"]
    reference_dataset_rows = reference["datasets"]

    candidate_mnli_values = [
        float(row["gold_accuracy"])
        for dataset_name, row in candidate_dataset_rows.items()
        if is_mnli_dataset(dataset_name) and row.get("gold_accuracy") is not None
    ]
    reference_mnli_values = [
        float(row["gold_accuracy"])
        for dataset_name, row in reference_dataset_rows.items()
        if is_mnli_dataset(dataset_name) and row.get("gold_accuracy") is not None
    ]
    candidate_anli_values = [
        float(row["gold_accuracy"])
        for dataset_name, row in candidate_dataset_rows.items()
        if is_anli_dev_dataset(dataset_name) and row.get("gold_accuracy") is not None
    ]
    reference_anli_values = [
        float(row["gold_accuracy"])
        for dataset_name, row in reference_dataset_rows.items()
        if is_anli_dev_dataset(dataset_name) and row.get("gold_accuracy") is not None
    ]

    candidate_mnli_macro = average(candidate_mnli_values)
    reference_mnli_macro = average(reference_mnli_values)
    candidate_anli_macro = average(candidate_anli_values)
    reference_anli_macro = average(reference_anli_values)

    metrics["mnli_macro_accuracy"] = candidate_mnli_macro
    metrics["anli_macro_accuracy"] = candidate_anli_macro

    if candidate_mnli_macro is None or reference_mnli_macro is None:
        reasons.append("missing MNLI development accuracy")
    else:
        metrics["mnli_macro_accuracy_drop"] = reference_mnli_macro - candidate_mnli_macro
        if reference_mnli_macro - candidate_mnli_macro > 0.005:
            reasons.append("MNLI macro accuracy drop exceeds 0.5 points")

    if candidate_anli_macro is None or reference_anli_macro is None:
        reasons.append("missing ANLI development accuracy")
    else:
        metrics["anli_macro_accuracy_drop"] = reference_anli_macro - candidate_anli_macro
        if reference_anli_macro - candidate_anli_macro > 0.010:
            reasons.append("ANLI macro accuracy drop exceeds 1.0 point")

    for dataset_name, row in candidate_dataset_rows.items():
        if not (is_mnli_dataset(dataset_name) or is_anli_dev_dataset(dataset_name)):
            continue
        reference_row = reference_dataset_rows.get(dataset_name)
        if reference_row is None:
            reasons.append(f"missing reference row for {dataset_name}")
            continue
        candidate_accuracy = row.get("gold_accuracy")
        reference_accuracy = reference_row.get("gold_accuracy")
        if candidate_accuracy is None or reference_accuracy is None:
            reasons.append(f"missing gold accuracy for {dataset_name}")
            continue
        if float(reference_accuracy) - float(candidate_accuracy) > 0.015:
            reasons.append(f"{dataset_name} accuracy drop exceeds 1.5 points")

    if candidate["float_label_agreement"] is None or float(candidate["float_label_agreement"]) < 0.98:
        reasons.append("aggregate float-label agreement is below 98.0%")

    if candidate_runtime is not None and reference_runtime is not None:
        candidate_peak_rss = optional_float(candidate_runtime, "peak_rss_after_timed_runs_median_bytes")
        reference_peak_rss = optional_float(reference_runtime, "peak_rss_after_timed_runs_median_bytes")
        if candidate_peak_rss is None or reference_peak_rss is None:
            reasons.append("missing peak RSS for CPU persistent benchmark")
        else:
            metrics["peak_rss_ratio_vs_reference"] = (
                candidate_peak_rss / reference_peak_rss if reference_peak_rss else None
            )
            if reference_peak_rss and candidate_peak_rss > reference_peak_rss * 1.25:
                reasons.append("peak RSS exceeds reference by more than 25%")

    return not reasons, reasons, metrics


def dominates_final(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["quantization"] == right["quantization"]:
        return False

    left_values = (
        left.get("size_bytes"),
        left.get("cpu_persistent_warm_median_ms"),
        left.get("cpu_persistent_resident_after_warmup_bytes"),
    )
    right_values = (
        right.get("size_bytes"),
        right.get("cpu_persistent_warm_median_ms"),
        right.get("cpu_persistent_resident_after_warmup_bytes"),
    )
    if any(value is None for value in left_values + right_values):
        return False
    if any(float(left_value) > float(right_value) for left_value, right_value in zip(left_values, right_values)):
        return False
    return any(float(left_value) < float(right_value) for left_value, right_value in zip(left_values, right_values))


def choose_recommendation(frontier_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not frontier_rows:
        return None
    chosen = min(
        frontier_rows,
        key=lambda row: (
            float(row["size_bytes"]),
            float(row.get("cpu_persistent_warm_median_ms") or float("inf")),
            float(row.get("cpu_persistent_resident_after_warmup_bytes") or float("inf")),
            -float(row.get("final_test_gold_accuracy") or 0.0),
            row["quantization"],
        ),
    )
    return {
        "quantization": chosen["quantization"],
        "artifact_path": chosen["artifact_path"],
    }


def build_candidate_rows(
    validation_aggregates: dict[str, dict[str, Any]],
    test_aggregates: dict[str, dict[str, Any]],
    stress_aggregates: dict[str, dict[str, Any]],
    runtime_rows: dict[str, dict[str, str]],
    cold_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    candidate_names = sorted(
        set(validation_aggregates) | set(test_aggregates) | set(stress_aggregates) | set(runtime_rows) | set(cold_rows)
    )
    rows: list[dict[str, Any]] = []
    for candidate_name in candidate_names:
        validation = validation_aggregates.get(candidate_name, {})
        test = test_aggregates.get(candidate_name, {})
        stress = stress_aggregates.get(candidate_name, {})
        runtime = runtime_rows.get(candidate_name, {})
        cold = cold_rows.get(candidate_name, {})
        rows.append(
            {
                "quantization": candidate_name,
                "artifact_path": validation.get("artifact_path", test.get("artifact_path", "")),
                "size_bytes": validation.get("size_bytes", test.get("size_bytes")),
                "smooth_quant_disabled": validation.get("smooth_quant_disabled"),
                "retry_reason": validation.get("retry_reason", ""),
                "validation_complete": validation.get("complete", False),
                "validation_gold_accuracy": validation.get("gold_accuracy"),
                "validation_float_label_agreement": validation.get("float_label_agreement"),
                "final_test_complete": test.get("complete", False),
                "final_test_gold_accuracy": test.get("gold_accuracy"),
                "final_test_float_label_agreement": test.get("float_label_agreement"),
                "stress_complete": stress.get("complete", False),
                "stress_gold_accuracy": stress.get("gold_accuracy"),
                "stress_float_label_agreement": stress.get("float_label_agreement"),
                "cpu_persistent_load_median_ms": optional_float(runtime, "load_median_ms"),
                "cpu_persistent_warm_median_ms": optional_float(runtime, "warm_median_ms"),
                "cpu_persistent_resident_after_warmup_bytes": optional_float(
                    runtime, "resident_after_warmup_median_bytes"
                ),
                "cpu_persistent_peak_rss_bytes": optional_float(
                    runtime, "peak_rss_after_timed_runs_median_bytes"
                ),
                "cpu_cold_load_median_ms": optional_float(cold, "load_median_ms"),
                "cpu_cold_warm_median_ms": optional_float(cold, "warm_median_ms"),
            }
        )
    return rows


def write_flat_csv(path: pathlib.Path | str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    resolved = resolve_absolute_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def relative_link(target: pathlib.Path, source_dir: pathlib.Path) -> str:
    return os.path.relpath(target, start=source_dir)


def repo_relative_label(target: pathlib.Path) -> str:
    try:
        return target.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return target.name


def markdown_link(target: pathlib.Path, report_dir: pathlib.Path) -> str:
    return f"[{repo_relative_label(target)}]({relative_link(target, report_dir)})"


def attempt4_report_paths(
    output_prefix: pathlib.Path | str,
    *,
    partial: bool,
) -> dict[str, pathlib.Path]:
    prefix = resolve_absolute_path(output_prefix)
    stem = f"{prefix.name}-partial" if partial else prefix.name
    return {
        "candidate_csv": prefix.parent / f"{stem}.csv",
        "candidate_json": prefix.parent / f"{stem}.json",
        "per_dataset_csv": prefix.parent / f"{stem}-per-dataset.csv",
        "per_dataset_json": prefix.parent / f"{stem}-per-dataset.json",
        "per_language_csv": prefix.parent / f"{stem}-per-language.csv",
        "per_language_json": prefix.parent / f"{stem}-per-language.json",
        "report_markdown": prefix.parent / f"{stem}.md",
    }


def delete_attempt4_report_paths(output_prefix: pathlib.Path | str, *, partial: bool) -> None:
    for path in attempt4_report_paths(output_prefix, partial=partial).values():
        path.unlink(missing_ok=True)


def build_attempt4_cpu_report(
    datasets_manifest_path: pathlib.Path | str,
    validation_summary_path: pathlib.Path | str,
    validation_runtime_path: pathlib.Path | str,
    output_prefix: pathlib.Path | str,
    *,
    test_summary_path: pathlib.Path | str | None = None,
    stress_summary_path: pathlib.Path | str | None = None,
    cold_benchmark_path: pathlib.Path | str | None = None,
    report_markdown_path: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    datasets_manifest = dict(read_json(datasets_manifest_path))
    validation_summary_rows = read_summary_rows(validation_summary_path)
    test_summary_rows = read_summary_rows(test_summary_path) if test_summary_path else []
    stress_summary_rows = read_summary_rows(stress_summary_path) if stress_summary_path else []
    validation_runtime_rows = read_benchmark_rows(validation_runtime_path)
    cold_rows = read_benchmark_rows(cold_benchmark_path) if cold_benchmark_path else {}

    validation_datasets = list(datasets_manifest["validation_datasets"])
    test_datasets = list(datasets_manifest["test_datasets"])
    stress_datasets = list(datasets_manifest["stress_datasets"])

    validation_aggregates = aggregate_required_rows(validation_summary_rows, validation_datasets)
    test_aggregates = aggregate_required_rows(test_summary_rows, test_datasets) if test_summary_rows else {}
    stress_aggregates = aggregate_required_rows(stress_summary_rows, stress_datasets) if stress_summary_rows else {}

    candidate_rows = build_candidate_rows(
        validation_aggregates,
        test_aggregates,
        stress_aggregates,
        validation_runtime_rows,
        cold_rows,
    )
    candidate_rows_by_name = {str(row["quantization"]): row for row in candidate_rows}

    development_complete = all(bool(item.get("complete")) for item in validation_aggregates.values())
    write_partial = not development_complete

    locked_quantizations: list[str] = []
    final_frontier: list[dict[str, Any]] = []
    recommendation = None

    for candidate_name in sorted(validation_aggregates):
        if not write_partial:
            gate_pass, reasons, gate_metrics = validation_gate(
                candidate_name,
                validation_aggregates,
                validation_runtime_rows,
            )
        else:
            gate_pass = False
            gate_metrics = {
                "mnli_macro_accuracy": None,
                "mnli_macro_accuracy_drop": None,
                "anli_macro_accuracy": None,
                "anli_macro_accuracy_drop": None,
                "validation_float_label_agreement": validation_aggregates[candidate_name].get("float_label_agreement"),
                "peak_rss_ratio_vs_reference": None,
            }
            if validation_aggregates[candidate_name].get("complete"):
                reasons = ["pending incomplete development summary"]
            else:
                reasons = ["missing validation datasets"]

        row = candidate_rows_by_name.setdefault(
            candidate_name,
            {
                "quantization": candidate_name,
                "artifact_path": validation_aggregates[candidate_name]["artifact_path"],
                "size_bytes": validation_aggregates[candidate_name]["size_bytes"],
            },
        )
        row["validation_gate_pass"] = gate_pass
        row["validation_gate_reasons"] = reasons
        row["validation_gate_reason_text"] = "; ".join(reasons)
        row.update(gate_metrics)
        row["final_frontier"] = False
        if not write_partial and (candidate_name == "reference" or gate_pass):
            locked_quantizations.append(candidate_name)

    if not write_partial and test_aggregates:
        locked_rows = []
        for candidate_name in locked_quantizations:
            candidate_row = candidate_rows_by_name[candidate_name]
            if not candidate_row.get("final_test_complete"):
                continue
            locked_rows.append(candidate_row)
        for row in locked_rows:
            row["final_frontier"] = not any(dominates_final(other, row) for other in locked_rows)
            if row["final_frontier"]:
                final_frontier.append(row)
        recommendation = choose_recommendation(final_frontier)

    per_dataset_rows: list[dict[str, Any]] = []
    for phase_name, rows in (
        ("validation", validation_summary_rows),
        ("test", test_summary_rows),
        ("stress", stress_summary_rows),
    ):
        for row in rows:
            enriched = dict(row)
            enriched["phase"] = phase_name
            enriched["language"] = xnli_language(str(row["dataset"]))
            per_dataset_rows.append(enriched)

    per_language_rows = [
        row
        for row in per_dataset_rows
        if row["language"] is not None and row["phase"] == "test"
    ]

    candidate_summary_rows = sorted(candidate_rows_by_name.values(), key=lambda row: row["quantization"])
    output_paths = attempt4_report_paths(output_prefix, partial=write_partial)
    if report_markdown_path:
        output_paths["report_markdown"] = resolve_absolute_path(report_markdown_path)

    write_flat_csv(
        output_paths["candidate_csv"],
        candidate_summary_rows,
        [
            "quantization",
            "artifact_path",
            "size_bytes",
            "smooth_quant_disabled",
            "retry_reason",
            "validation_complete",
            "validation_gate_pass",
            "validation_gate_reason_text",
            "validation_gold_accuracy",
            "validation_float_label_agreement",
            "mnli_macro_accuracy",
            "mnli_macro_accuracy_drop",
            "anli_macro_accuracy",
            "anli_macro_accuracy_drop",
            "peak_rss_ratio_vs_reference",
            "cpu_persistent_load_median_ms",
            "cpu_persistent_warm_median_ms",
            "cpu_persistent_resident_after_warmup_bytes",
            "cpu_persistent_peak_rss_bytes",
            "cpu_cold_load_median_ms",
            "cpu_cold_warm_median_ms",
            "final_test_complete",
            "final_test_gold_accuracy",
            "final_test_float_label_agreement",
            "stress_complete",
            "stress_gold_accuracy",
            "stress_float_label_agreement",
            "final_frontier",
        ],
    )
    write_json(
        output_paths["candidate_json"],
        {
            "locked_quantizations": locked_quantizations,
            "recommendation": recommendation,
            "candidates": candidate_summary_rows,
            "final_frontier": [row["quantization"] for row in final_frontier],
        },
    )
    write_flat_csv(
        output_paths["per_dataset_csv"],
        per_dataset_rows,
        [
            "phase",
            "dataset",
            "role",
            "backend",
            "quantization",
            "language",
            "gold_accuracy",
            "float_label_agreement",
            "mean_abs_logit_delta",
            "max_abs_logit_delta",
            "example_count",
            "labeled_example_count",
            "correct_prediction_count",
            "disagreement_count",
            "pareto_frontier",
        ],
    )
    write_json(output_paths["per_dataset_json"], {"rows": per_dataset_rows})
    write_flat_csv(
        output_paths["per_language_csv"],
        per_language_rows,
        [
            "phase",
            "dataset",
            "language",
            "quantization",
            "gold_accuracy",
            "float_label_agreement",
            "example_count",
        ],
    )
    write_json(output_paths["per_language_json"], {"rows": per_language_rows})

    report_path = output_paths["report_markdown"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = report_path.parent
    lines = [
        "# Attempt4 CPU Deployment Study",
        "",
        "## Locked Quantizations",
        "",
        (
            "- Locked after development gates: pending incomplete development summary"
            if write_partial
            else f"- Locked after development gates: {', '.join(f'`{name}`' for name in locked_quantizations)}"
        ),
        (
            "- Recommendation: pending incomplete development summary"
            if write_partial
            else f"- Recommendation: `{recommendation['quantization']}`"
            if recommendation
            else "- Recommendation: pending locked-final test"
        ),
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Gate | Dev Acc | Dev Float Agree | CPU Warm | CPU Steady RSS | CPU Peak RSS | Cold Load |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidate_summary_rows:
        gate_value = row.get("validation_gate_pass")
        gate_text = "pass" if gate_value else "pending" if write_partial else "fail"
        lines.append(
            "| "
            f"`{row['quantization']}` | "
            f"{gate_text} | "
            f"{percent_text(row.get('validation_gold_accuracy'))} | "
            f"{percent_text(row.get('validation_float_label_agreement'))} | "
            f"{ms_text(row.get('cpu_persistent_warm_median_ms'))} | "
            f"{mib_text(row.get('cpu_persistent_resident_after_warmup_bytes'))} | "
            f"{mib_text(row.get('cpu_persistent_peak_rss_bytes'))} | "
            f"{ms_text(row.get('cpu_cold_load_median_ms'))} |"
        )

    if not write_partial and test_aggregates:
        lines.extend(
            [
                "",
                "## Locked Final Frontier",
                "",
                "| Candidate | Size | Final Acc | Final Float Agree | CPU Warm | CPU Steady RSS | Frontier |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in candidate_summary_rows:
            if row["quantization"] not in locked_quantizations or not row.get("final_test_complete"):
                continue
            lines.append(
                "| "
                f"`{row['quantization']}` | "
                f"{mib_text(row.get('size_bytes'))} | "
                f"{percent_text(row.get('final_test_gold_accuracy'))} | "
                f"{percent_text(row.get('final_test_float_label_agreement'))} | "
                f"{ms_text(row.get('cpu_persistent_warm_median_ms'))} | "
                f"{mib_text(row.get('cpu_persistent_resident_after_warmup_bytes'))} | "
                f"{'frontier' if row.get('final_frontier') else '-'} |"
            )

    if per_language_rows:
        lines.extend(
            [
                "",
                "## XNLI Per-Language Rows",
                "",
                f"- {markdown_link(output_paths['per_language_csv'], report_dir)}",
                f"- {markdown_link(output_paths['per_language_json'], report_dir)}",
            ]
        )

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- {markdown_link(output_paths['candidate_csv'], report_dir)}",
            f"- {markdown_link(output_paths['candidate_json'], report_dir)}",
            f"- {markdown_link(output_paths['per_dataset_csv'], report_dir)}",
            f"- {markdown_link(output_paths['per_dataset_json'], report_dir)}",
            f"- {markdown_link(resolve_absolute_path(validation_summary_path), report_dir)}",
            f"- {markdown_link(resolve_absolute_path(validation_runtime_path), report_dir)}",
            f"- {markdown_link(resolve_absolute_path(datasets_manifest_path), report_dir)}",
        ]
    )
    if test_summary_path:
        lines.append(f"- {markdown_link(resolve_absolute_path(test_summary_path), report_dir)}")
    if stress_summary_path:
        lines.append(f"- {markdown_link(resolve_absolute_path(stress_summary_path), report_dir)}")
    if cold_benchmark_path:
        lines.append(f"- {markdown_link(resolve_absolute_path(cold_benchmark_path), report_dir)}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if write_partial:
        delete_attempt4_report_paths(output_prefix, partial=False)
    else:
        delete_attempt4_report_paths(output_prefix, partial=True)

    return {
        "partial": write_partial,
        "paths": output_paths,
        "locked_quantizations": locked_quantizations,
        "recommendation": recommendation,
        "candidate_rows": candidate_summary_rows,
        "final_frontier": [row["quantization"] for row in final_frontier],
    }


def read_benchmark_examples(
    tsv_paths: Sequence[pathlib.Path],
    sample_mode: str,
    max_examples: int,
    *,
    seed: int = 0,
) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for tsv_path in tsv_paths:
        with tsv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "premise" not in (reader.fieldnames or ()) or "hypothesis" not in (reader.fieldnames or ()):
                raise RuntimeError(f"TSV must include premise and hypothesis columns: {tsv_path}")
            for index, row in enumerate(reader):
                examples.append(
                    BenchmarkExample(
                        benchmark=(row.get("benchmark") or "").strip() or tsv_path.name,
                        example_id=(row.get("id") or f"{tsv_path.stem}-{index + 1}").strip(),
                        premise=str(row["premise"]),
                        hypothesis=str(row["hypothesis"]),
                    )
                )

    if sample_mode != "first":
        raise RuntimeError(f"Unsupported benchmark sample mode: {sample_mode}")
    if max_examples > 0:
        examples = examples[:max_examples]
    if not examples:
        raise RuntimeError("No runtime benchmark examples loaded")
    return examples


def peak_rss_bytes() -> float | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (AttributeError, ValueError, OSError):
        return None
    value = getattr(usage, "ru_maxrss", 0)
    if value <= 0:
        return None
    if sys.platform == "darwin":
        return float(value)
    return float(value) * 1024.0


def process_rss_bytes(pid: int) -> float | None:
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    if not value:
        return None
    try:
        return float(value) * 1024.0
    except ValueError:
        return None


def benchmark_warm_runs(
    predictor: CpuOrtPredictor,
    example: BenchmarkExample,
    repeat: int,
) -> list[float]:
    runs_ms: list[float] = []
    for _ in range(repeat):
        started_at = time.perf_counter()
        predictor.predict_logits(example.premise, example.hypothesis)
        runs_ms.append((time.perf_counter() - started_at) * 1000.0)
    return runs_ms


def benchmark_examples_payload(examples: Sequence[BenchmarkExample]) -> list[dict[str, str]]:
    return [
        {
            "benchmark": example.benchmark,
            "id": example.example_id,
            "premise": example.premise,
            "hypothesis": example.hypothesis,
        }
        for example in examples
    ]


def parse_benchmark_examples(payload: Sequence[dict[str, object]]) -> list[BenchmarkExample]:
    return [
        BenchmarkExample(
            benchmark=str(item["benchmark"]),
            example_id=str(item["id"]),
            premise=str(item["premise"]),
            hypothesis=str(item["hypothesis"]),
        )
        for item in payload
    ]


def benchmark_worker_persistent(request: dict[str, object]) -> dict[str, object]:
    model_path = pathlib.Path(str(request["model_path"]))
    tokenizer_root = pathlib.Path(str(request["tokenizer_root"]))
    backend = str(request["backend"])
    repeat = int(request["repeat"])
    warmup = int(request["warmup"])
    examples = parse_benchmark_examples(list(request["examples"]))

    load_started_at = time.perf_counter()
    predictor = CpuOrtPredictor(model_path, tokenizer_root, backend)
    load_ms = (time.perf_counter() - load_started_at) * 1000.0

    pid = os.getpid()
    resident_after_load = process_rss_bytes(pid)
    peak_after_load = peak_rss_bytes()

    for example in examples:
        for _ in range(warmup):
            predictor.predict_logits(example.premise, example.hypothesis)

    resident_after_warmup = process_rss_bytes(pid)
    peak_after_warmup = peak_rss_bytes()

    warm_run_values: list[float] = []
    per_example: list[dict[str, object]] = []
    per_benchmark_runs: dict[str, list[float]] = {}

    for example in examples:
        runs_ms = benchmark_warm_runs(predictor, example, repeat)
        warm_run_values.extend(runs_ms)
        per_benchmark_runs.setdefault(example.benchmark, []).extend(runs_ms)
        summary = summarize_numeric(runs_ms)
        per_example.append(
            {
                "benchmark": example.benchmark,
                "id": example.example_id,
                "timing_mean_ms": summary["mean"],
                "timing_median_ms": summary["median"],
                "timing_p95_ms": summary["p95"],
                "timing_min_ms": summary["min"],
                "timing_max_ms": summary["max"],
            }
        )

    resident_after_timed_runs = process_rss_bytes(pid)
    peak_after_timed_runs = peak_rss_bytes()

    per_benchmark: dict[str, dict[str, object]] = {}
    for benchmark_name in sorted(per_benchmark_runs):
        benchmark_values = per_benchmark_runs[benchmark_name]
        per_benchmark[benchmark_name] = {
            "examples": sum(1 for example in examples if example.benchmark == benchmark_name),
            "warm_latency_ms": summarize_numeric(benchmark_values),
        }

    return {
        "examples": len(examples),
        "file_size_bytes": model_path.stat().st_size,
        "load_ms": summarize_numeric([load_ms]),
        "warm_latency_ms": summarize_numeric(warm_run_values),
        "resident_after_load_bytes": summarize_numeric([resident_after_load] if resident_after_load is not None else []),
        "resident_after_warmup_bytes": summarize_numeric(
            [resident_after_warmup] if resident_after_warmup is not None else []
        ),
        "resident_after_timed_runs_bytes": summarize_numeric(
            [resident_after_timed_runs] if resident_after_timed_runs is not None else []
        ),
        "peak_rss_after_load_bytes": summarize_numeric([peak_after_load] if peak_after_load is not None else []),
        "peak_rss_after_warmup_bytes": summarize_numeric(
            [peak_after_warmup] if peak_after_warmup is not None else []
        ),
        "peak_rss_after_timed_runs_bytes": summarize_numeric(
            [peak_after_timed_runs] if peak_after_timed_runs is not None else []
        ),
        "time_l_peak_rss_bytes": summarize_numeric([]),
        "per_benchmark": per_benchmark,
        "per_example": per_example,
    }


def benchmark_worker_coldstart(request: dict[str, object]) -> dict[str, object]:
    examples = parse_benchmark_examples(list(request["examples"]))
    if len(examples) != 1:
        raise RuntimeError("Coldstart benchmark worker requires exactly one example")

    example = examples[0]
    model_path = pathlib.Path(str(request["model_path"]))
    tokenizer_root = pathlib.Path(str(request["tokenizer_root"]))
    backend = str(request["backend"])
    repeat = int(request["repeat"])
    warmup = int(request["warmup"])

    load_started_at = time.perf_counter()
    predictor = CpuOrtPredictor(model_path, tokenizer_root, backend)
    load_ms = (time.perf_counter() - load_started_at) * 1000.0
    pid = os.getpid()
    resident_after_load = process_rss_bytes(pid)
    peak_after_load = peak_rss_bytes()

    for _ in range(warmup):
        predictor.predict_logits(example.premise, example.hypothesis)
    resident_after_warmup = process_rss_bytes(pid)
    peak_after_warmup = peak_rss_bytes()
    runs_ms = benchmark_warm_runs(predictor, example, repeat)
    resident_after_timed_runs = process_rss_bytes(pid)
    peak_after_timed_runs = peak_rss_bytes()
    run_summary = summarize_numeric(runs_ms)
    return {
        "benchmark": example.benchmark,
        "id": example.example_id,
        "load_ms": load_ms,
        "timing_mean_ms": run_summary["mean"],
        "timing_median_ms": run_summary["median"],
        "timing_p95_ms": run_summary["p95"],
        "timing_min_ms": run_summary["min"],
        "timing_max_ms": run_summary["max"],
        "timing_runs_ms": runs_ms,
        "resident_after_load_bytes": resident_after_load,
        "resident_after_warmup_bytes": resident_after_warmup,
        "resident_after_timed_runs_bytes": resident_after_timed_runs,
        "peak_rss_after_load_bytes": peak_after_load,
        "peak_rss_after_warmup_bytes": peak_after_warmup,
        "peak_rss_after_timed_runs_bytes": peak_after_timed_runs,
        "time_l_peak_rss_bytes": None,
    }


def aggregate_coldstart_worker_results(
    model_path: pathlib.Path,
    examples: Sequence[BenchmarkExample],
    worker_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    load_values: list[float] = []
    warm_run_values: list[float] = []
    resident_after_load_values: list[float] = []
    resident_after_warmup_values: list[float] = []
    resident_after_timed_runs_values: list[float] = []
    peak_rss_after_load_values: list[float] = []
    peak_rss_after_warmup_values: list[float] = []
    peak_rss_after_timed_runs_values: list[float] = []
    time_l_peak_rss_values: list[float] = []
    per_example: list[dict[str, object]] = []

    for row in worker_rows:
        load_ms = float(row["load_ms"])
        runs_ms = [float(value) for value in list(row["timing_runs_ms"])]
        load_values.append(load_ms)
        warm_run_values.extend(runs_ms)

        resident_after_load = row.get("resident_after_load_bytes")
        resident_after_warmup = row.get("resident_after_warmup_bytes")
        resident_after_timed_runs = row.get("resident_after_timed_runs_bytes")
        peak_rss_after_load = row.get("peak_rss_after_load_bytes")
        peak_rss_after_warmup = row.get("peak_rss_after_warmup_bytes")
        peak_rss_after_timed_runs = row.get("peak_rss_after_timed_runs_bytes")
        time_l_peak_rss = row.get("time_l_peak_rss_bytes")

        if resident_after_load is not None:
            resident_after_load_values.append(float(resident_after_load))
        if resident_after_warmup is not None:
            resident_after_warmup_values.append(float(resident_after_warmup))
        if resident_after_timed_runs is not None:
            resident_after_timed_runs_values.append(float(resident_after_timed_runs))
        if peak_rss_after_load is not None:
            peak_rss_after_load_values.append(float(peak_rss_after_load))
        if peak_rss_after_warmup is not None:
            peak_rss_after_warmup_values.append(float(peak_rss_after_warmup))
        if peak_rss_after_timed_runs is not None:
            peak_rss_after_timed_runs_values.append(float(peak_rss_after_timed_runs))
        if time_l_peak_rss is not None:
            time_l_peak_rss_values.append(float(time_l_peak_rss))

        per_example.append(
            {
                "benchmark": str(row["benchmark"]),
                "id": str(row["id"]),
                "load_ms": load_ms,
                "timing_mean_ms": float(row["timing_mean_ms"]),
                "timing_median_ms": float(row["timing_median_ms"]),
                "timing_p95_ms": float(row["timing_p95_ms"]),
                "timing_min_ms": None if row.get("timing_min_ms") is None else float(row["timing_min_ms"]),
                "timing_max_ms": None if row.get("timing_max_ms") is None else float(row["timing_max_ms"]),
                "timing_runs_ms": runs_ms,
                "resident_after_load_bytes": None if resident_after_load is None else float(resident_after_load),
                "resident_after_warmup_bytes": None if resident_after_warmup is None else float(resident_after_warmup),
                "resident_after_timed_runs_bytes": None
                if resident_after_timed_runs is None
                else float(resident_after_timed_runs),
                "peak_rss_after_load_bytes": None if peak_rss_after_load is None else float(peak_rss_after_load),
                "peak_rss_after_warmup_bytes": None if peak_rss_after_warmup is None else float(peak_rss_after_warmup),
                "peak_rss_after_timed_runs_bytes": None
                if peak_rss_after_timed_runs is None
                else float(peak_rss_after_timed_runs),
                "time_l_peak_rss_bytes": None if time_l_peak_rss is None else float(time_l_peak_rss),
            }
        )

    per_benchmark: dict[str, dict[str, object]] = {}
    for benchmark_name in sorted({row["benchmark"] for row in per_example}):
        benchmark_rows = [row for row in per_example if row["benchmark"] == benchmark_name]
        benchmark_loads = [float(row["load_ms"]) for row in benchmark_rows]
        benchmark_runs = [run_ms for row in benchmark_rows for run_ms in row["timing_runs_ms"]]
        per_benchmark[benchmark_name] = {
            "examples": len(benchmark_rows),
            "load_ms": summarize_numeric(benchmark_loads),
            "warm_latency_ms": summarize_numeric(benchmark_runs),
        }

    return {
        "examples": len(examples),
        "file_size_bytes": model_path.stat().st_size,
        "load_ms": summarize_numeric(load_values),
        "warm_latency_ms": summarize_numeric(warm_run_values),
        "resident_after_load_bytes": summarize_numeric(resident_after_load_values),
        "resident_after_warmup_bytes": summarize_numeric(resident_after_warmup_values),
        "resident_after_timed_runs_bytes": summarize_numeric(resident_after_timed_runs_values),
        "peak_rss_after_load_bytes": summarize_numeric(peak_rss_after_load_values),
        "peak_rss_after_warmup_bytes": summarize_numeric(peak_rss_after_warmup_values),
        "peak_rss_after_timed_runs_bytes": summarize_numeric(peak_rss_after_timed_runs_values),
        "time_l_peak_rss_bytes": summarize_numeric(time_l_peak_rss_values),
        "per_benchmark": per_benchmark,
        "per_example": per_example,
    }


def internal_benchmark_request(
    request: dict[str, object],
) -> dict[str, object]:
    mode = str(request["mode"])
    if mode == "persistent":
        return benchmark_worker_persistent(request)
    if mode == "coldstart":
        return benchmark_worker_coldstart(request)
    raise RuntimeError(f"Unknown internal benchmark mode: {mode}")


def invoke_internal_benchmark_request(request: dict[str, object]) -> dict[str, object]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "replication_cpu_final.py"),
        "--internal-benchmark-worker",
        "--internal-benchmark-request",
        json.dumps(request),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Internal benchmark worker failed:\n"
            f"command={' '.join(command)}\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )
    return json.loads(completed.stdout)


def load_existing_benchmark_cache(
    output_prefix: pathlib.Path,
) -> dict[str, object] | None:
    candidates = [
        output_prefix.with_suffix(".json"),
        output_prefix.parent / f"{output_prefix.name}-partial.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def benchmark_payload(
    *,
    examples: Sequence[BenchmarkExample],
    mode: str,
    repeat: int,
    warmup: int,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "examples": len(examples),
        "sources": sorted({example.benchmark for example in examples}),
        "mode": mode,
        "repeat": repeat,
        "warmup": warmup,
        "sample_mode": "first",
        "max_examples": 0,
        "results": rows,
    }


def benchmark_partial_paths(output_prefix: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    return (
        output_prefix.parent / f"{output_prefix.name}-partial.json",
        output_prefix.parent / f"{output_prefix.name}-partial.csv",
    )


def refresh_quantization_for_benchmark(
    connection: sqlite3.Connection,
    quantization_name: str,
) -> QuantizationRecord:
    quantization = load_quantization_record(connection, quantization_name)
    current_state = inspect_artifact(quantization.artifact_path, None)
    update_artifact_state(
        connection,
        quantization.artifact_id,
        current_state,
        quantization.stdout_log_path,
        quantization.stderr_log_path,
    )
    refreshed = load_quantization_record(connection, quantization_name)
    if refreshed.artifact_status != STUDY_STATUS_MATERIALIZED or not refreshed.artifact_sha256:
        raise RuntimeError(f"Benchmark artifact is not materialized: {quantization_name}")
    return refreshed


def benchmark_result_sort_key(row: dict[str, object]) -> tuple[object, object, object]:
    return (
        row["backend"],
        row["warm_latency_ms"]["median"],
        row["load_ms"]["median"],
    )


def benchmark_runtime_phase(
    scratchpad_root: pathlib.Path | str,
    mode: str,
    quantizations: Sequence[str],
    output_prefix: pathlib.Path | str,
    *,
    repeat: int = 5,
    warmup: int = 1,
    worker_runner: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> pathlib.Path:
    scratchpad = resolve_absolute_path(scratchpad_root)
    prefix = resolve_absolute_path(output_prefix)
    if mode not in {"persistent", "coldstart"}:
        raise RuntimeError(f"Unsupported runtime benchmark mode: {mode}")

    probe_path = scratchpad / "datasets" / "hf-core-probe.tsv"
    examples = read_benchmark_examples([probe_path], "first", 0)
    runner = worker_runner or invoke_internal_benchmark_request
    cached_payload = load_existing_benchmark_cache(prefix)
    cached_rows = {
        str(row["candidate"]): dict(row)
        for row in list(cached_payload.get("results", []))
    } if isinstance(cached_payload, dict) else {}

    connection = open_study_connection(scratchpad / "db.sqlite3")
    try:
        completed_rows: list[dict[str, object]] = []
        partial_json_path, partial_csv_path = benchmark_partial_paths(prefix)

        for quantization_name in quantizations:
            quantization = refresh_quantization_for_benchmark(connection, quantization_name)
            cached_row = cached_rows.get(quantization_name)
            if (
                cached_row is not None
                and cached_row.get("candidate") == quantization_name
                and cached_row.get("mode") == mode
                and cached_row.get("artifact_path") == str(quantization.artifact_path)
                and cached_row.get("artifact_sha256") == quantization.artifact_sha256
            ):
                row = cached_row
            else:
                request_base = {
                    "mode": mode,
                    "backend": "cpu",
                    "model_path": str(quantization.artifact_path),
                    "tokenizer_root": str(study_tokenizer_root(scratchpad)),
                    "repeat": repeat,
                    "warmup": warmup,
                }
                if mode == "persistent":
                    summary = runner(
                        {
                            **request_base,
                            "examples": benchmark_examples_payload(examples),
                        }
                    )
                else:
                    summary = aggregate_coldstart_worker_results(
                        quantization.artifact_path,
                        examples,
                        [
                            runner(
                                {
                                    **request_base,
                                    "examples": benchmark_examples_payload([example]),
                                }
                            )
                            for example in examples
                        ],
                    )
                row = {
                    "candidate": quantization_name,
                    "backend": "cpu",
                    "mode": mode,
                    "artifact_path": str(quantization.artifact_path),
                    "artifact_sha256": quantization.artifact_sha256,
                    **summary,
                }

            completed_rows.append(row)
            partial_rows = sorted(completed_rows, key=benchmark_result_sort_key)
            partial_payload = benchmark_payload(
                examples=examples,
                mode=mode,
                repeat=repeat,
                warmup=warmup,
                rows=partial_rows,
            )
            write_json(partial_json_path, partial_payload)
            write_benchmark_csv(partial_csv_path, partial_rows)

        final_rows = sorted(completed_rows, key=benchmark_result_sort_key)
        write_json(prefix.with_suffix(".json"), benchmark_payload(
            examples=examples,
            mode=mode,
            repeat=repeat,
            warmup=warmup,
            rows=final_rows,
        ))
        write_benchmark_csv(prefix.with_suffix(".csv"), final_rows)
        partial_json_path.unlink(missing_ok=True)
        partial_csv_path.unlink(missing_ok=True)
        return prefix.with_suffix(".csv")
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded attempt4 CPU deployment study end-to-end using the "
            "Python-only replication workflow."
        )
    )
    parser.add_argument(
        "--workspace",
        default=str(REPO_ROOT / "scratchpad" / "replication_cpu_final"),
        help="Directory for generated datasets, manifests, reports, and logs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite/rebuild existing outputs where supported.",
    )
    parser.add_argument(
        "--force-datasets",
        action="store_true",
        help="Regenerate the bounded attempt4 dataset pack instead of reusing local files.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Stop after the interim report and locked-candidate selection.",
    )
    parser.add_argument(
        "--internal-benchmark-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-benchmark-request",
        default="",
        help=argparse.SUPPRESS,
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


def catalog_quantization_names(
    catalog_path: pathlib.Path | str = DEFAULT_STUDY_CATALOG_PATH,
) -> list[str]:
    return [entry.name for entry in load_study_catalog(resolve_absolute_path(catalog_path))]


def complete_quantizations(
    summary_json_path: pathlib.Path | str,
    required_datasets: Sequence[str],
) -> list[str]:
    payload = read_json(summary_json_path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Summary JSON does not contain a row list: {summary_json_path}")

    required_dataset_set = set(required_datasets)
    grouped: dict[str, set[str]] = {}
    for row in rows:
        quantization = str(row["quantization"])
        grouped.setdefault(quantization, set()).add(str(row["dataset"]))

    return sorted(
        quantization
        for quantization, datasets in grouped.items()
        if datasets == required_dataset_set
    )


def verify_role_assignments(
    scratchpad_root: pathlib.Path | str,
    expected: dict[str, Sequence[str]],
) -> None:
    scratchpad = resolve_absolute_path(scratchpad_root)
    connection = open_study_connection(scratchpad / "db.sqlite3")
    try:
        rows = connection.execute("SELECT name, role FROM dataset ORDER BY name").fetchall()
    finally:
        connection.close()

    actual_by_role: dict[str, list[str]] = {}
    for name, role in rows:
        actual_by_role.setdefault(str(role), []).append(str(name))

    for role, expected_names in expected.items():
        actual_names = sorted(actual_by_role.get(role, []))
        if sorted(expected_names) != actual_names:
            raise RuntimeError(
                f"Dataset-role mismatch for {role}: expected {sorted(expected_names)}, got {actual_names}"
            )


def report_artifact_paths(
    report_result: dict[str, Any],
) -> dict[str, str]:
    paths = dict(report_result["paths"])
    return {name: str(path) for name, path in paths.items()}


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


def run_attempt4_pipeline(
    workspace: pathlib.Path | str,
    *,
    catalog_path: pathlib.Path | str = DEFAULT_STUDY_CATALOG_PATH,
    force: bool = False,
    force_datasets: bool = False,
    skip_test: bool = False,
) -> dict[str, object]:
    scratchpad_root = resolve_absolute_path(workspace)
    resolved_catalog_path = resolve_absolute_path(catalog_path)
    if not resolved_catalog_path.is_file():
        raise FileNotFoundError(f"Study catalog not found: {resolved_catalog_path}")

    scratchpad_root.mkdir(parents=True, exist_ok=True)
    reports_root = scratchpad_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    datasets_manifest_path = prepare_attempt4_datasets(scratchpad_root, force_datasets)
    dataset_manifest = dict(read_json(datasets_manifest_path))
    stage_runtime_assets(scratchpad_root)
    initialize_study_workspace(
        scratchpad_root,
        catalog_path=resolved_catalog_path,
        force=force,
    )
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

    quantizations = catalog_quantization_names(resolved_catalog_path)
    development_datasets = [
        *list(dataset_manifest["smoke_datasets"]),
        *list(dataset_manifest["validation_datasets"]),
    ]
    print("Running smoke and development evaluation for the bounded CPU catalog.", flush=True)
    for quantization in quantizations:
        for dataset_name in development_datasets:
            run_study_evaluation(
                scratchpad_root,
                quantization,
                dataset_name,
                backend="cpu",
                force_regenerate=force,
                force_rerun=force,
            )

    validation_summary_prefix = reports_root / "attempt4-validation-summary"
    summarize_study_db_to_prefix(
        scratchpad_root,
        roles=["fidelity_validation"],
        backends=["cpu"],
        output_prefix=validation_summary_prefix,
    )
    validation_summary_path = validation_summary_prefix.with_suffix(".json")
    validation_complete = complete_quantizations(
        validation_summary_path,
        list(dataset_manifest["validation_datasets"]),
    )

    print("Benchmarking complete development candidates on persistent CPU runtime and RSS.", flush=True)
    validation_runtime_csv_path = benchmark_runtime_phase(
        scratchpad_root,
        "persistent",
        validation_complete,
        reports_root / "attempt4-validation-cpu-persistent",
    )

    report_output_prefix = reports_root / "attempt4-cpu-summary"
    report_result = build_attempt4_cpu_report(
        datasets_manifest_path,
        validation_summary_path,
        validation_runtime_csv_path,
        report_output_prefix,
    )
    locked_quantizations = list(report_result["locked_quantizations"])

    manifest: dict[str, object] = {
        "scratchpad_root": str(scratchpad_root),
        "catalog": str(resolved_catalog_path),
        "validation_complete_quantizations": validation_complete,
        "locked_quantizations": locked_quantizations,
        "datasets_manifest": str(datasets_manifest_path),
        "validation_summary": str(validation_summary_path),
        "validation_runtime_csv": str(validation_runtime_csv_path),
        "report_artifacts": report_artifact_paths(report_result),
    }

    if skip_test:
        print("Skipping final test and stress evaluation because --skip-test was requested.", flush=True)
        manifest["final_report"] = str(report_result["paths"]["candidate_json"])
    else:
        print("Running locked final-test and stress-test evaluation.", flush=True)
        for dataset_name in dataset_manifest["test_datasets"]:
            for quantization in locked_quantizations:
                run_study_evaluation(
                    scratchpad_root,
                    quantization,
                    dataset_name,
                    backend="cpu",
                    force_regenerate=force,
                    force_rerun=force,
                )
        for dataset_name in dataset_manifest["stress_datasets"]:
            for quantization in locked_quantizations:
                run_study_evaluation(
                    scratchpad_root,
                    quantization,
                    dataset_name,
                    backend="cpu",
                    force_regenerate=force,
                    force_rerun=force,
                )

        test_summary_prefix = reports_root / "attempt4-test-summary"
        summarize_study_db_to_prefix(
            scratchpad_root,
            roles=["fidelity_test"],
            backends=["cpu"],
            output_prefix=test_summary_prefix,
        )
        test_summary_path = test_summary_prefix.with_suffix(".json")

        stress_summary_prefix = reports_root / "attempt4-stress-summary"
        summarize_study_db_to_prefix(
            scratchpad_root,
            roles=["stress_test"],
            backends=["cpu"],
            output_prefix=stress_summary_prefix,
        )
        stress_summary_path = stress_summary_prefix.with_suffix(".json")

        cold_benchmark_csv_path = benchmark_runtime_phase(
            scratchpad_root,
            "coldstart",
            locked_quantizations,
            reports_root / "attempt4-test-cpu-cold",
        )

        report_result = build_attempt4_cpu_report(
            datasets_manifest_path,
            validation_summary_path,
            validation_runtime_csv_path,
            report_output_prefix,
            test_summary_path=test_summary_path,
            stress_summary_path=stress_summary_path,
            cold_benchmark_path=cold_benchmark_csv_path,
        )
        manifest["test_summary"] = str(test_summary_path)
        manifest["stress_summary"] = str(stress_summary_path)
        manifest["cold_benchmark_csv"] = str(cold_benchmark_csv_path)
        manifest["final_report"] = str(report_result["paths"]["candidate_json"])
        manifest["report_artifacts"] = report_artifact_paths(report_result)

    manifest_path = reports_root / "attempt4-manifest.json"
    write_json(manifest_path, manifest)
    print(f"manifest: {manifest_path}", flush=True)
    return manifest


def main() -> int:
    args = parse_args()
    if args.internal_benchmark_worker:
        if not args.internal_benchmark_request:
            raise RuntimeError("--internal-benchmark-request is required with --internal-benchmark-worker")
        request = json.loads(args.internal_benchmark_request)
        print(json.dumps(internal_benchmark_request(request)))
        return 0

    manifest = run_attempt4_pipeline(
        args.workspace,
        force=args.force,
        force_datasets=args.force_datasets,
        skip_test=args.skip_test,
    )
    print("\nDone.")
    print(f"workspace: {manifest['scratchpad_root']}")
    print(f"manifest: {pathlib.Path(str(manifest['scratchpad_root'])) / 'reports' / 'attempt4-manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
