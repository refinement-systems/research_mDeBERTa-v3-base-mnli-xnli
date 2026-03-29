#!/usr/bin/env python3

import argparse
import csv
import json
import math
import pathlib
import random
import sys
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_HF_SOURCE = str(REPO_ROOT / "models/mdeberta")
DEFAULT_FLOAT_MODEL = str(REPO_ROOT / "models/mdeberta/onnx/model.onnx")
DEFAULT_COMPARE_MODELS = [
    (
        "attention_only",
        str(
            REPO_ROOT
            / "models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx"
        ),
    ),
    (
        "attention_proj_only",
        str(
            REPO_ROOT
            / "models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx"
        ),
    ),
]
DEFAULT_TSVS = [
    REPO_ROOT / "benchmarks/nli/mnli-validation_matched-200-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/mnli-validation_mismatched-200-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-de-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-en-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-es-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-fr-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-zh-test-50-per-label.tsv",
]
LABELS = ["entailment", "neutral", "contradiction"]
XNLI_ZH_BENCHMARK = "xnli-zh-test-50-per-label.tsv"


@dataclass
class Example:
    benchmark: str
    example_id: str
    premise: str
    hypothesis: str
    gold_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HF/PyTorch, float ONNX, and selected quantized ONNX models on "
            "a multilingual TSV sample, summarizing accuracy, label agreement, and logit drift."
        )
    )
    parser.add_argument(
        "--hf-source",
        default=DEFAULT_HF_SOURCE,
        help=f"HF model/tokenizer source (default: {DEFAULT_HF_SOURCE})",
    )
    parser.add_argument(
        "--float-model",
        default=DEFAULT_FLOAT_MODEL,
        help=f"Float ONNX model (default: {DEFAULT_FLOAT_MODEL})",
    )
    parser.add_argument(
        "--compare-model",
        dest="compare_models",
        action="append",
        default=[],
        help="Comparison model in NAME=PATH form. Repeat to add more.",
    )
    parser.add_argument(
        "--tsv",
        dest="tsv_paths",
        action="append",
        default=[],
        help="Benchmark TSV. Repeat to add more. Defaults to the non-overlapping suite.",
    )
    parser.add_argument(
        "--max-examples-per-source",
        type=int,
        default=10,
        help="Maximum number of examples to sample from each TSV (default: 10)",
    )
    parser.add_argument(
        "--max-total-examples",
        type=int,
        default=0,
        help="Optional cap across all loaded examples after per-source sampling (default: unlimited)",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="random",
        help="How to choose examples from each TSV (default: random)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --sample-mode=random (default: 0)",
    )
    parser.add_argument(
        "--show-slices",
        action="store_true",
        help="Print per-benchmark rows in addition to the aggregate summary",
    )
    parser.add_argument(
        "--max-disagreements",
        type=int,
        default=5,
        help="Maximum HF-label disagreements to print per candidate (default: 5)",
    )
    parser.add_argument(
        "--max-drift-examples",
        type=int,
        default=3,
        help="Maximum top-drift examples to print per candidate (default: 3)",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def resolve_compare_models(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.compare_models:
        return DEFAULT_COMPARE_MODELS

    resolved = []
    for item in args.compare_models:
        if "=" not in item:
            raise RuntimeError(f"--compare-model must be NAME=PATH: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise RuntimeError(f"--compare-model must be NAME=PATH: {item}")
        resolved.append((name, path))
    return resolved


def read_examples(
    tsv_paths: list[pathlib.Path],
    sample_mode: str,
    max_examples_per_source: int,
    max_total_examples: int,
    seed: int,
) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []

    for tsv_path in tsv_paths:
        with tsv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "premise" not in reader.fieldnames or "hypothesis" not in reader.fieldnames:
                raise RuntimeError(f"TSV must include premise and hypothesis columns: {tsv_path}")

            source_examples: list[Example] = []
            for index, row in enumerate(reader):
                label = row.get("label") or row.get("gold_label") or ""
                benchmark_name = (row.get("benchmark") or "").strip() or tsv_path.name
                source_examples.append(
                    Example(
                        benchmark=benchmark_name,
                        example_id=row.get("id") or f"{tsv_path.stem}-{index + 1}",
                        premise=row["premise"],
                        hypothesis=row["hypothesis"],
                        gold_label=label.strip(),
                    )
                )

        if sample_mode == "random":
            rng.shuffle(source_examples)

        if max_examples_per_source > 0:
            source_examples = source_examples[:max_examples_per_source]

        examples.extend(source_examples)
        if max_total_examples > 0 and len(examples) >= max_total_examples:
            examples = examples[:max_total_examples]
            break

    if not examples:
        raise RuntimeError("No examples loaded for HF/ONNX benchmarking")
    return examples


def softmax(logits: list[float]) -> list[float]:
    max_value = max(logits)
    exps = [math.exp(value - max_value) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def predicted_label(logits: list[float]) -> str:
    return LABELS[max(range(len(logits)), key=lambda index: logits[index])]


def max_abs_delta(left: list[float], right: list[float]) -> float:
    return max(abs(left_value - right_value) for left_value, right_value in zip(left, right))


def mean_abs_delta(left: list[float], right: list[float]) -> float:
    return sum(abs(left_value - right_value) for left_value, right_value in zip(left, right)) / max(
        len(left), 1
    )


def blank_confusion() -> dict[str, dict[str, int]]:
    return {left: {right: 0 for right in LABELS} for left in LABELS}


def update_confusion(
    matrix: dict[str, dict[str, int]],
    reference_label: str,
    model_label: str,
) -> None:
    if reference_label not in matrix or model_label not in matrix[reference_label]:
        return
    matrix[reference_label][model_label] += 1


def benchmark_language(benchmark_name: str) -> str | None:
    if not benchmark_name.startswith("xnli-"):
        return None
    parts = benchmark_name.split("-")
    if len(parts) < 3:
        return None
    return parts[1]


def build_sessions(args: argparse.Namespace, compare_models: list[tuple[str, str]]):
    import onnxruntime
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_source, local_files_only=True, use_fast=True)
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        args.hf_source,
        local_files_only=True,
    )
    hf_model.eval()

    def make_session(path: str):
        return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

    sessions = {"float": make_session(args.float_model)}
    for name, path in compare_models:
        sessions[name] = make_session(path)
    return tokenizer, hf_model, sessions, torch


def run_hf(tokenizer, hf_model, torch_module, example: Example) -> dict[str, object]:
    encoded = tokenizer(example.premise, example.hypothesis, truncation=True, return_tensors="pt")
    with torch_module.no_grad():
        logits_tensor = hf_model(**encoded).logits[0]
    logits = [float(value) for value in logits_tensor.tolist()]
    return {
        "logits": logits,
        "scores": softmax(logits),
        "label": predicted_label(logits),
    }


def run_onnx(tokenizer, session, example: Example):
    import numpy

    encoded = tokenizer(example.premise, example.hypothesis, truncation=True, return_tensors="np")
    feed = {}
    for input_meta in session.get_inputs():
        if input_meta.name not in encoded:
            continue
        value = encoded[input_meta.name]
        if value.dtype != numpy.int64:
            value = value.astype(numpy.int64, copy=False)
        feed[input_meta.name] = value
    outputs = session.run(None, feed)
    logits = [float(value) for value in outputs[0][0].tolist()]
    return {
        "logits": logits,
        "scores": softmax(logits),
        "label": predicted_label(logits),
    }


def summarize_model(
    name: str,
    example_results: list[dict[str, object]],
) -> dict[str, object]:
    example_count = len(example_results)
    labeled_examples = sum(1 for item in example_results if item["gold_label"])
    accuracy_hits = sum(
        1 for item in example_results if item["gold_label"] and item["model_label"] == item["gold_label"]
    )
    hf_agreements = sum(1 for item in example_results if item["model_label"] == item["hf_label"])
    float_agreements = sum(
        1 for item in example_results if item["model_label"] == item["float_label"]
    )
    mean_max_abs_logit_delta_vs_hf = sum(
        item["max_abs_logit_delta_vs_hf"] for item in example_results
    ) / max(example_count, 1)
    mean_mean_abs_logit_delta_vs_hf = sum(
        item["mean_abs_logit_delta_vs_hf"] for item in example_results
    ) / max(example_count, 1)
    max_abs_logit_delta_vs_hf = max(
        item["max_abs_logit_delta_vs_hf"] for item in example_results
    )
    return {
        "candidate": name,
        "examples": example_count,
        "labeled_examples": labeled_examples,
        "accuracy": accuracy_hits / max(labeled_examples, 1),
        "accuracy_hits": accuracy_hits,
        "hf_agreement": hf_agreements / max(example_count, 1),
        "hf_agreements": hf_agreements,
        "float_agreement": float_agreements / max(example_count, 1),
        "float_agreements": float_agreements,
        "mean_max_abs_logit_delta_vs_hf": mean_max_abs_logit_delta_vs_hf,
        "mean_mean_abs_logit_delta_vs_hf": mean_mean_abs_logit_delta_vs_hf,
        "max_abs_logit_delta_vs_hf": max_abs_logit_delta_vs_hf,
    }


def summarize_pairwise(
    candidate_rows: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
) -> dict[str, object]:
    if len(candidate_rows) != len(baseline_rows):
        raise RuntimeError("Pairwise summaries require aligned example lists")

    example_count = len(candidate_rows)
    labeled_examples = 0
    fixed_baseline_errors = 0
    introduced_new_errors = 0
    both_correct = 0
    both_wrong = 0
    label_agreements = 0

    for candidate_row, baseline_row in zip(candidate_rows, baseline_rows):
        if candidate_row["benchmark"] != baseline_row["benchmark"] or candidate_row["id"] != baseline_row["id"]:
            raise RuntimeError("Pairwise summaries require aligned example identities")

        if candidate_row["model_label"] == baseline_row["model_label"]:
            label_agreements += 1

        gold_label = candidate_row["gold_label"]
        if not gold_label:
            continue

        labeled_examples += 1
        candidate_correct = candidate_row["model_label"] == gold_label
        baseline_correct = baseline_row["model_label"] == gold_label
        if candidate_correct and not baseline_correct:
            fixed_baseline_errors += 1
        elif baseline_correct and not candidate_correct:
            introduced_new_errors += 1
        elif candidate_correct and baseline_correct:
            both_correct += 1
        else:
            both_wrong += 1

    return {
        "examples": example_count,
        "labeled_examples": labeled_examples,
        "fixed_baseline_errors": fixed_baseline_errors,
        "introduced_new_errors": introduced_new_errors,
        "net_accuracy_delta": fixed_baseline_errors - introduced_new_errors,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "label_agreements": label_agreements,
        "label_agreement": label_agreements / max(example_count, 1),
    }


def summarize_pairwise_grouped(
    candidate_rows: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
    group_fn,
) -> dict[str, dict[str, object]]:
    grouped_candidate: dict[str, list[dict[str, object]]] = {}
    grouped_baseline: dict[str, list[dict[str, object]]] = {}

    for candidate_row, baseline_row in zip(candidate_rows, baseline_rows):
        group_name = group_fn(candidate_row)
        if not group_name:
            continue
        grouped_candidate.setdefault(group_name, []).append(candidate_row)
        grouped_baseline.setdefault(group_name, []).append(baseline_row)

    return {
        group_name: summarize_pairwise(grouped_candidate[group_name], grouped_baseline[group_name])
        for group_name in sorted(grouped_candidate)
    }


def summarize_per_benchmark(
    examples: list[Example],
    per_example_models: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    benchmark_names = sorted({example.benchmark for example in examples})
    per_benchmark = {}
    for name, rows in per_example_models.items():
        summaries = []
        for benchmark_name in benchmark_names:
            filtered = [row for row in rows if row["benchmark"] == benchmark_name]
            summaries.append(summarize_model(name, filtered) | {"benchmark": benchmark_name})
        per_benchmark[name] = summaries
    return per_benchmark


def write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate",
        "examples",
        "labeled_examples",
        "accuracy",
        "accuracy_hits",
        "hf_agreement",
        "hf_agreements",
        "float_agreement",
        "float_agreements",
        "fixed_float_errors",
        "introduced_new_errors_vs_float",
        "net_accuracy_delta_vs_float",
        "label_agreement_vs_float",
        "xnli_zh_examples",
        "xnli_zh_accuracy",
        "xnli_zh_accuracy_hits",
        "xnli_zh_hf_agreement",
        "xnli_zh_float_agreement",
        "mean_max_abs_logit_delta_vs_hf",
        "mean_mean_abs_logit_delta_vs_hf",
        "max_abs_logit_delta_vs_hf",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def print_summary_table(rows: list[dict[str, object]]) -> None:
    header = (
        f"{'candidate':<34} {'acc':>8} {'hf_ag':>8} "
        f"{'flt_ag':>8} {'fix_flt':>8} {'new_err':>8} {'zh_acc':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        zh_accuracy = row["xnli_zh_accuracy"]
        print(
            f"{row['candidate']:<34} "
            f"{row['accuracy'] * 100:>7.2f}% "
            f"{row['hf_agreement'] * 100:>7.2f}% "
            f"{row['float_agreement'] * 100:>7.2f}% "
            f"{row['fixed_float_errors']:>8} "
            f"{row['introduced_new_errors_vs_float']:>8} "
            f"{(zh_accuracy * 100 if zh_accuracy is not None else float('nan')):>7.2f}%"
        )


def print_per_benchmark(per_benchmark: dict[str, list[dict[str, object]]]) -> None:
    for candidate, rows in per_benchmark.items():
        print()
        print(candidate)
        for row in rows:
            print(
                f"  {row['benchmark']:<42} "
                f"acc={row['accuracy'] * 100:>6.2f}% "
                f"hf={row['hf_agreement'] * 100:>6.2f}% "
                f"float={row['float_agreement'] * 100:>6.2f}% "
                f"mean_max={row['mean_max_abs_logit_delta_vs_hf']:.6f}"
            )


def print_pairwise_summary(
    pairwise: dict[str, dict[str, object]],
    show_slices: bool,
) -> None:
    for candidate, payload in pairwise.items():
        print()
        print(
            f"{candidate} vs float: "
            f"fixed={payload['vs_float']['fixed_baseline_errors']} "
            f"new={payload['vs_float']['introduced_new_errors']} "
            f"net={payload['vs_float']['net_accuracy_delta']:+d} "
            f"label_agree={payload['vs_float']['label_agreement'] * 100:.2f}%"
        )

        zh_summary = payload["per_benchmark_vs_float"].get(XNLI_ZH_BENCHMARK)
        if zh_summary:
            print(
                f"  {XNLI_ZH_BENCHMARK}: "
                f"fixed={zh_summary['fixed_baseline_errors']} "
                f"new={zh_summary['introduced_new_errors']} "
                f"net={zh_summary['net_accuracy_delta']:+d}"
            )

        for other_name, other_summary in payload["vs_quantized"].items():
            print(
                f"  vs {other_name}: "
                f"fixed={other_summary['fixed_baseline_errors']} "
                f"new={other_summary['introduced_new_errors']} "
                f"net={other_summary['net_accuracy_delta']:+d} "
                f"label_agree={other_summary['label_agreement'] * 100:.2f}%"
            )

        if show_slices:
            print("  per_source_vs_float:")
            for benchmark_name, summary in payload["per_benchmark_vs_float"].items():
                print(
                    f"    {benchmark_name:<42} "
                    f"fixed={summary['fixed_baseline_errors']:>3} "
                    f"new={summary['introduced_new_errors']:>3} "
                    f"net={summary['net_accuracy_delta']:+3d}"
                )

            if payload["per_language_vs_float"]:
                print("  per_language_vs_float:")
                for language_name, summary in payload["per_language_vs_float"].items():
                    print(
                        f"    {language_name:<6} "
                        f"fixed={summary['fixed_baseline_errors']:>3} "
                        f"new={summary['introduced_new_errors']:>3} "
                        f"net={summary['net_accuracy_delta']:+3d}"
                    )


def print_disagreements(
    examples_by_candidate: dict[str, list[dict[str, object]]],
    max_disagreements: int,
    max_drift_examples: int,
) -> None:
    for candidate, rows in examples_by_candidate.items():
        disagreements = [
            row for row in rows if row["model_label"] != row["hf_label"]
        ]
        if disagreements and max_disagreements > 0:
            print()
            print(f"{candidate} disagreements_vs_hf: {len(disagreements)}")
            for row in disagreements[:max_disagreements]:
                print(
                    f"  {row['benchmark']} id={row['id']} gold={row['gold_label'] or '<none>'} "
                    f"hf={row['hf_label']} model={row['model_label']} "
                    f"max_abs={row['max_abs_logit_delta_vs_hf']:.6f}"
                )
                print(f"    premise={row['premise']}")
                print(f"    hypothesis={row['hypothesis']}")

        if max_drift_examples > 0:
            top_drift = sorted(
                rows,
                key=lambda row: row["max_abs_logit_delta_vs_hf"],
                reverse=True,
            )[:max_drift_examples]
            print()
            print(f"{candidate} top_drift_examples:")
            for row in top_drift:
                print(
                    f"  {row['benchmark']} id={row['id']} gold={row['gold_label'] or '<none>'} "
                    f"hf={row['hf_label']} model={row['model_label']} "
                    f"max_abs={row['max_abs_logit_delta_vs_hf']:.6f} "
                    f"mean_abs={row['mean_abs_logit_delta_vs_hf']:.6f}"
                )
                print(f"    premise={row['premise']}")
                print(f"    hypothesis={row['hypothesis']}")


def main() -> int:
    args = parse_args()
    compare_models = resolve_compare_models(args)
    tsv_paths = [pathlib.Path(path) for path in (args.tsv_paths or DEFAULT_TSVS)]
    for path in tsv_paths:
        if not path.is_file():
            raise RuntimeError(f"Benchmark TSV not found: {path}")

    examples = read_examples(
        tsv_paths,
        args.sample_mode,
        args.max_examples_per_source,
        args.max_total_examples,
        args.seed,
    )
    tokenizer, hf_model, sessions, torch_module = build_sessions(args, compare_models)

    model_names = ["float"] + [name for name, _ in compare_models]
    per_example_models: dict[str, list[dict[str, object]]] = {name: [] for name in model_names}

    for example in examples:
        hf_payload = run_hf(tokenizer, hf_model, torch_module, example)
        float_payload = run_onnx(tokenizer, sessions["float"], example)
        candidate_payloads = {
            name: run_onnx(tokenizer, sessions[name], example) for name in model_names
        }

        for name, payload in candidate_payloads.items():
            per_example_models[name].append(
                {
                    "benchmark": example.benchmark,
                    "id": example.example_id,
                    "gold_label": example.gold_label,
                    "hf_label": hf_payload["label"],
                    "float_label": float_payload["label"],
                    "model_label": payload["label"],
                    "premise": example.premise,
                    "hypothesis": example.hypothesis,
                    "max_abs_logit_delta_vs_hf": max_abs_delta(payload["logits"], hf_payload["logits"]),
                    "mean_abs_logit_delta_vs_hf": mean_abs_delta(payload["logits"], hf_payload["logits"]),
                }
            )

    summary_by_name = {name: summarize_model(name, rows) for name, rows in per_example_models.items()}
    per_benchmark = summarize_per_benchmark(examples, per_example_models)

    pairwise: dict[str, dict[str, object]] = {}
    for candidate_name, candidate_rows in per_example_models.items():
        pairwise_vs_float = summarize_pairwise(candidate_rows, per_example_models["float"])
        per_benchmark_vs_float = summarize_pairwise_grouped(
            candidate_rows,
            per_example_models["float"],
            lambda row: row["benchmark"],
        )
        per_language_vs_float = summarize_pairwise_grouped(
            candidate_rows,
            per_example_models["float"],
            lambda row: benchmark_language(row["benchmark"]),
        )
        confusion_vs_hf = blank_confusion()
        confusion_vs_float = blank_confusion()
        for candidate_row, float_row in zip(candidate_rows, per_example_models["float"]):
            update_confusion(confusion_vs_hf, candidate_row["hf_label"], candidate_row["model_label"])
            update_confusion(confusion_vs_float, float_row["model_label"], candidate_row["model_label"])

        vs_quantized = {}
        for other_name, other_rows in per_example_models.items():
            if other_name in {"float", candidate_name}:
                continue
            vs_quantized[other_name] = summarize_pairwise(candidate_rows, other_rows)

        pairwise[candidate_name] = {
            "vs_float": pairwise_vs_float,
            "per_benchmark_vs_float": per_benchmark_vs_float,
            "per_language_vs_float": per_language_vs_float,
            "confusion_vs_hf": confusion_vs_hf,
            "confusion_vs_float": confusion_vs_float,
            "vs_quantized": vs_quantized,
        }

    for candidate_name, row in summary_by_name.items():
        vs_float = pairwise[candidate_name]["vs_float"]
        zh_summary = next(
            (
                benchmark_row
                for benchmark_row in per_benchmark[candidate_name]
                if benchmark_row["benchmark"] == XNLI_ZH_BENCHMARK
            ),
            None,
        )
        row["fixed_float_errors"] = vs_float["fixed_baseline_errors"]
        row["introduced_new_errors_vs_float"] = vs_float["introduced_new_errors"]
        row["net_accuracy_delta_vs_float"] = vs_float["net_accuracy_delta"]
        row["label_agreement_vs_float"] = vs_float["label_agreement"]
        row["xnli_zh_examples"] = zh_summary["examples"] if zh_summary else 0
        row["xnli_zh_accuracy"] = zh_summary["accuracy"] if zh_summary else None
        row["xnli_zh_accuracy_hits"] = zh_summary["accuracy_hits"] if zh_summary else 0
        row["xnli_zh_hf_agreement"] = zh_summary["hf_agreement"] if zh_summary else None
        row["xnli_zh_float_agreement"] = zh_summary["float_agreement"] if zh_summary else None

    summary_rows = list(summary_by_name.values())
    summary_rows.sort(
        key=lambda row: (
            row["accuracy"],
            row["hf_agreement"],
            -row["mean_max_abs_logit_delta_vs_hf"],
        ),
        reverse=True,
    )

    print(f"examples: {len(examples)}")
    print(f"benchmarks: {', '.join(sorted({example.benchmark for example in examples}))}")
    print_summary_table(summary_rows)
    if args.show_slices:
        print_per_benchmark(per_benchmark)
    print_pairwise_summary(
        {
            candidate: payload
            for candidate, payload in pairwise.items()
            if candidate != "float"
        },
        args.show_slices,
    )
    print_disagreements(per_example_models, args.max_disagreements, args.max_drift_examples)

    payload = {
        "examples": len(examples),
        "sources": sorted({example.benchmark for example in examples}),
        "sample_mode": args.sample_mode,
        "max_examples_per_source": args.max_examples_per_source,
        "results": summary_rows,
        "per_benchmark": per_benchmark,
        "pairwise": pairwise,
        "per_example": per_example_models,
    }
    if args.summary_json:
        write_json(pathlib.Path(args.summary_json), payload)
    if args.summary_csv:
        write_csv(pathlib.Path(args.summary_csv), summary_rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
