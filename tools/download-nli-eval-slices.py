#!/usr/bin/env python3

import argparse
import csv
import datetime
import email.utils
import json
import pathlib
import random
import sys
import time
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
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_RETRIES = 8
DEFAULT_INITIAL_BACKOFF_SECONDS = 5.0
DEFAULT_MAX_BACKOFF_SECONDS = 120.0
LABELS = ("entailment", "neutral", "contradiction")


@dataclass(frozen=True)
class ExportTarget:
    dataset: str
    config: str
    split: str
    per_label: int
    skip_per_label: int
    output_name: str


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
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "nli-eval-slice-downloader/1.0",
            },
        )

        for attempt in range(self.max_retries + 1):
            self._wait_for_request_slot()
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                    return json.load(response)
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
        "--mnli-split",
        action="append",
        dest="mnli_splits",
        help=(
            "MNLI split to fetch. Repeat to override the defaults of validation_matched and "
            "validation_mismatched."
        ),
    )
    parser.add_argument(
        "--xnli-split",
        default="test",
        help="XNLI split to fetch (default: test).",
    )
    parser.add_argument(
        "--skip-per-label",
        type=int,
        default=0,
        help="Skip this many examples per label before collecting the balanced slice (default: 0).",
    )
    parser.add_argument(
        "--mnli-skip-per-label",
        type=int,
        default=None,
        help="Optional MNLI-specific per-label skip count. Defaults to --skip-per-label.",
    )
    parser.add_argument(
        "--xnli-skip-per-label",
        type=int,
        default=None,
        help="Optional XNLI-specific per-label skip count. Defaults to --skip-per-label.",
    )
    parser.add_argument(
        "--name-tag",
        default="",
        help="Optional tag inserted into generated filenames, e.g. calibration or search-validation.",
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
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=(
            "Per-request timeout for datasets-server calls "
            f"(default: {DEFAULT_REQUEST_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
        help=(
            "Minimum time between request starts, used to avoid hitting rate limits "
            f"(default: {DEFAULT_MIN_REQUEST_INTERVAL_SECONDS})"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retry count for 429/5xx/network errors (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--initial-backoff-seconds",
        type=float,
        default=DEFAULT_INITIAL_BACKOFF_SECONDS,
        help=(
            "Initial retry backoff, doubled on each retry until the max backoff is reached "
            f"(default: {DEFAULT_INITIAL_BACKOFF_SECONDS})"
        ),
    )
    parser.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=DEFAULT_MAX_BACKOFF_SECONDS,
        help=f"Maximum retry backoff in seconds (default: {DEFAULT_MAX_BACKOFF_SECONDS})",
    )

    args = parser.parse_args()

    if args.page_size < 1 or args.page_size > 100:
        parser.error("--page-size must be between 1 and 100")
    if args.mnli_per_label < 0:
        parser.error("--mnli-per-label must be non-negative")
    if args.xnli_per_label < 0:
        parser.error("--xnli-per-label must be non-negative")
    if args.skip_per_label < 0:
        parser.error("--skip-per-label must be non-negative")
    if args.mnli_skip_per_label is not None and args.mnli_skip_per_label < 0:
        parser.error("--mnli-skip-per-label must be non-negative")
    if args.xnli_skip_per_label is not None and args.xnli_skip_per_label < 0:
        parser.error("--xnli-skip-per-label must be non-negative")
    if args.request_timeout_seconds <= 0:
        parser.error("--request-timeout-seconds must be positive")
    if args.min_request_interval_seconds < 0:
        parser.error("--min-request-interval-seconds must be non-negative")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if args.initial_backoff_seconds <= 0:
        parser.error("--initial-backoff-seconds must be positive")
    if args.max_backoff_seconds <= 0:
        parser.error("--max-backoff-seconds must be positive")
    if args.initial_backoff_seconds > args.max_backoff_seconds:
        parser.error("--initial-backoff-seconds must not exceed --max-backoff-seconds")

    return args


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


def tagged_output_name(
    prefix: str,
    split: str,
    per_label: int,
    name_tag: str,
    skip_per_label: int,
) -> str:
    parts = [prefix, split]
    if name_tag:
        parts.append(name_tag)
    if skip_per_label > 0:
        parts.append(f"skip{skip_per_label}")
    parts.append(f"{per_label}-per-label.tsv")
    return "-".join(parts)


def build_targets(args: argparse.Namespace) -> list[ExportTarget]:
    targets: list[ExportTarget] = []
    mnli_skip = args.mnli_skip_per_label
    if mnli_skip is None:
        mnli_skip = args.skip_per_label
    xnli_skip = args.xnli_skip_per_label
    if xnli_skip is None:
        xnli_skip = args.skip_per_label

    if not args.skip_mnli and args.mnli_per_label > 0:
        for split in (args.mnli_splits or ["validation_matched", "validation_mismatched"]):
            targets.append(
                ExportTarget(
                    dataset="nyu-mll/multi_nli",
                    config="default",
                    split=split,
                    per_label=args.mnli_per_label,
                    skip_per_label=mnli_skip,
                    output_name=tagged_output_name(
                        "mnli",
                        split,
                        args.mnli_per_label,
                        args.name_tag,
                        mnli_skip,
                    ),
                )
            )

    xnli_languages = args.xnli_languages or list(DEFAULT_XNLI_LANGUAGES)
    if not args.skip_xnli and args.xnli_per_label > 0:
        for language in xnli_languages:
            targets.append(
                ExportTarget(
                    dataset="facebook/xnli",
                    config=language,
                    split=args.xnli_split,
                    per_label=args.xnli_per_label,
                    skip_per_label=xnli_skip,
                    output_name=tagged_output_name(
                        f"xnli-{language}",
                        args.xnli_split,
                        args.xnli_per_label,
                        args.name_tag,
                        xnli_skip,
                    ),
                )
            )

    if not targets:
        raise RuntimeError("Nothing to do; all requested benchmark counts are zero or skipped")

    return targets


def collect_balanced_examples(
    client: DatasetsServerClient,
    target: ExportTarget,
    page_size: int,
    seed: int,
) -> list[dict[str, object]]:
    kept_by_label = {label: [] for label in LABELS}
    skipped_by_label = {label: 0 for label in LABELS}
    offset = 0
    total_rows = None
    label_names: list[str] | None = None
    dataset_slug = target.dataset.replace("/", "-")

    while any(len(examples) < target.per_label for examples in kept_by_label.values()):
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
            if skipped_by_label[label] < target.skip_per_label:
                skipped_by_label[label] += 1
                continue
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
        counts = ", ".join(
            f"{label}=kept:{len(kept_by_label[label])}/skipped:{skipped_by_label[label]}"
            for label in LABELS
        )
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
    client = DatasetsServerClient(
        args.api_base_url,
        request_timeout_seconds=args.request_timeout_seconds,
        min_request_interval_seconds=args.min_request_interval_seconds,
        max_retries=args.max_retries,
        initial_backoff_seconds=args.initial_backoff_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
    )

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
            client,
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
