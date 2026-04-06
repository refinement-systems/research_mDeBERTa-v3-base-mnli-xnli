#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sqlite3
import sys
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCRATCHPAD_ROOT = REPO_ROOT / "scratchpad"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize stored study evaluations by comparing each artifact against the "
            "backend-specific float reference and computing size-vs-fidelity frontiers."
        )
    )
    parser.add_argument(
        "--scratchpad-root",
        default=str(DEFAULT_SCRATCHPAD_ROOT),
        help=f"Scratchpad root directory (default: {DEFAULT_SCRATCHPAD_ROOT})",
    )
    parser.add_argument(
        "--db-path",
        default="",
        help="Optional path to the study SQLite database. Defaults to <scratchpad>/db.sqlite3.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Dataset name to summarize. Repeat to add more. Defaults to all datasets for the selected roles.",
    )
    parser.add_argument(
        "--role",
        dest="roles",
        action="append",
        default=[],
        choices=["calibration", "fidelity_validation", "fidelity_test", "stress_test", "smoke"],
        help="Dataset role to summarize. Repeat to add more. Defaults to fidelity_validation only.",
    )
    parser.add_argument(
        "--backend",
        dest="backends",
        action="append",
        default=[],
        choices=["cpu", "coreml"],
        help="Backend to summarize. Repeat to add more. Defaults to all backends present in the DB.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional output prefix. Defaults to <scratchpad>/reports/study-summary.",
    )
    return parser.parse_args()


def resolve_output_prefix(args: argparse.Namespace, scratchpad_root: pathlib.Path) -> pathlib.Path:
    if args.output_prefix:
        return pathlib.Path(args.output_prefix)
    return scratchpad_root / "reports" / "study-summary"


def fetch_dataset_rows(conn: sqlite3.Connection, dataset_names: list[str], roles: list[str]) -> list[dict[str, object]]:
    where = []
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
    return [dict(row) for row in conn.execute(query, params)]


def fetch_backend_names(conn: sqlite3.Connection, requested_backends: list[str]) -> list[str]:
    query = "SELECT LOWER(name) AS name FROM backend"
    rows = [str(row["name"]) for row in conn.execute(query)]
    if requested_backends:
        requested = set(requested_backends)
        return [name for name in rows if name in requested]
    return rows


def fetch_reference_rows(
    conn: sqlite3.Connection,
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
    rows = {}
    for row in conn.execute(query, (dataset_id, backend_name)):
        rows[int(row["dataset_row_id"])] = (
            float(row["entailment_logit"]),
            float(row["neutral_logit"]),
            float(row["contradiction_logit"]),
            str(row["predicted_label"]),
        )
    return rows


def fetch_candidate_rows(
    conn: sqlite3.Connection,
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
    return [dict(row) for row in conn.execute(query, (dataset_id, backend_name))]


def compute_frontier_flags(rows: list[SummaryRow]) -> dict[tuple[str, str, str, str], bool]:
    grouped: dict[tuple[str, str, str], list[SummaryRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset, row.role, row.backend), []).append(row)

    flags: dict[tuple[str, str, str, str], bool] = {}
    for key, bucket in grouped.items():
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


def parse_generation_metadata(stdout_log_path: str) -> tuple[bool | None, str]:
    if not stdout_log_path:
        return None, ""

    path = pathlib.Path(stdout_log_path)
    if not path.is_file():
        return None, ""

    text = path.read_text(encoding="utf-8", errors="replace")
    smooth_quant_disabled: bool | None = None
    retry_reason = ""

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
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

    match = re.search(r"smooth_quant_disabled=(True|False)", text)
    if match:
        smooth_quant_disabled = match.group(1) == "True"

    for line in text.splitlines():
        if line.startswith("  retry_reason: "):
            retry_reason = line.split(": ", 1)[1].strip()
            break

    return smooth_quant_disabled, retry_reason


def summarize_dataset_backend(
    conn: sqlite3.Connection,
    dataset_name: str,
    dataset_role: str,
    dataset_id: int,
    backend_name: str,
) -> list[SummaryRow]:
    reference_rows = fetch_reference_rows(conn, dataset_id, backend_name)
    if not reference_rows:
        return []

    per_quantization: dict[str, dict[str, object]] = {}
    for row in fetch_candidate_rows(conn, dataset_id, backend_name):
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
        abs_delta_count = int(item["abs_delta_count"])
        labeled_example_count = int(item["labeled_examples"])
        correct_prediction_count = int(item["correct_predictions"])
        smooth_quant_disabled, retry_reason = parse_generation_metadata(str(item["stdout_log_path"]))
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
                mean_abs_logit_delta=float(item["abs_delta_sum"]) / float(abs_delta_count),
                max_abs_logit_delta=float(item["max_abs_delta"]),
                disagreement_count=int(item["disagreements"]),
                smooth_quant_disabled=smooth_quant_disabled,
                retry_reason=retry_reason,
                pareto_frontier=False,
            )
        )
    return summary_rows


def write_outputs(rows: list[SummaryRow], output_prefix: pathlib.Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    flags = compute_frontier_flags(rows)
    rows = [
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

    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps({"rows": [row.__dict__ for row in rows]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def main() -> int:
    args = parse_args()
    scratchpad_root = pathlib.Path(args.scratchpad_root).resolve()
    db_path = pathlib.Path(args.db_path).resolve() if args.db_path else scratchpad_root / "db.sqlite3"
    output_prefix = resolve_output_prefix(args, scratchpad_root)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        roles = args.roles or ["fidelity_validation"]
        datasets = fetch_dataset_rows(conn, args.datasets, roles)
        backends = fetch_backend_names(conn, args.backends)

        summary_rows: list[SummaryRow] = []
        for dataset in datasets:
            for backend_name in backends:
                summary_rows.extend(
                    summarize_dataset_backend(
                        conn,
                        dataset_name=str(dataset["dataset_name"]),
                        dataset_role=str(dataset["dataset_role"]),
                        dataset_id=int(dataset["dataset_id"]),
                        backend_name=backend_name,
                    )
                )

        write_outputs(summary_rows, output_prefix)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
