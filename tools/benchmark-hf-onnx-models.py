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
                source_examples.append(
                    Example(
                        benchmark=tsv_path.name,
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
        f"{'candidate':<40} {'acc':>8} {'hf_agree':>9} "
        f"{'float_ag':>9} {'mean_max':>10} {'max_abs':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['candidate']:<40} "
            f"{row['accuracy'] * 100:>7.2f}% "
            f"{row['hf_agreement'] * 100:>8.2f}% "
            f"{row['float_agreement'] * 100:>8.2f}% "
            f"{row['mean_max_abs_logit_delta_vs_hf']:>10.6f} "
            f"{row['max_abs_logit_delta_vs_hf']:>10.6f}"
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

    summary_rows = [summarize_model(name, rows) for name, rows in per_example_models.items()]
    summary_rows.sort(
        key=lambda row: (
            row["accuracy"],
            row["hf_agreement"],
            -row["mean_max_abs_logit_delta_vs_hf"],
        ),
        reverse=True,
    )
    per_benchmark = summarize_per_benchmark(examples, per_example_models)

    print(f"examples: {len(examples)}")
    print(f"benchmarks: {', '.join(sorted({example.benchmark for example in examples}))}")
    print_summary_table(summary_rows)
    if args.show_slices:
        print_per_benchmark(per_benchmark)
    print_disagreements(per_example_models, args.max_disagreements, args.max_drift_examples)

    payload = {
        "examples": len(examples),
        "sources": sorted({example.benchmark for example in examples}),
        "sample_mode": args.sample_mode,
        "max_examples_per_source": args.max_examples_per_source,
        "results": summary_rows,
        "per_benchmark": per_benchmark,
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
