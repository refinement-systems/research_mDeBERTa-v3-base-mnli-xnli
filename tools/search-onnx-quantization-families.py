#!/usr/bin/env python3

import argparse
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field

from mdeberta_onnx_quantization import (
    LAYER_PATTERN,
    layer_subset,
    load_matmul_families,
    merge_nodes,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_FLOAT_MODEL = REPO_ROOT / "models/mdeberta/onnx/model.onnx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models/mdeberta/onnx/candidates/family_search"
DEFAULT_EVAL_BINARY = REPO_ROOT / "builddir/nli-eval"
DEFAULT_QUANTIZE_SCRIPT = REPO_ROOT / "tools/quantize-onnx-model.py"
DEFAULT_PYTHON = (
    REPO_ROOT / ".venv/bin/python"
    if (REPO_ROOT / ".venv/bin/python").exists()
    else pathlib.Path(sys.executable)
)
DEFAULT_EXISTING_BEST = REPO_ROOT / (
    "models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx"
)
DEFAULT_BASELINE_DYNAMIC = REPO_ROOT / (
    "models/mdeberta/onnx/candidates/dynamic_qint8_matmul.onnx"
)
DEFAULT_BENCHMARKS = [
    REPO_ROOT / "benchmarks/nli/mnli-validation_matched-200-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/mnli-validation_mismatched-200-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-de-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-en-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-es-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-fr-test-50-per-label.tsv",
    REPO_ROOT / "benchmarks/nli/xnli-zh-test-50-per-label.tsv",
]
STATIC_CANDIDATE_DESCRIPTIONS = [
    (
        "current_best_reference",
        "Existing best candidate from the single-example exclusion search",
    ),
    (
        "baseline_dynamic_matmul",
        "Baseline dynamic MatMul quantization with no exclusions",
    ),
    (
        "ffn_layers_8_11_float",
        "Keep FFN intermediate/output dense matmuls float in layers 8-11",
    ),
    (
        "ffn_layers_6_11_float",
        "Keep FFN intermediate/output dense matmuls float in layers 6-11",
    ),
    (
        "ffn_layers_4_11_float",
        "Keep FFN intermediate/output dense matmuls float in layers 4-11",
    ),
    (
        "attention_output_layers_8_11_float",
        "Keep attention output dense matmuls float in layers 8-11",
    ),
    (
        "layer_11_block_float",
        "Keep the full quantizable block float in layer 11",
    ),
    (
        "layers_10_11_block_float",
        "Keep the full quantizable block float in layers 10-11",
    ),
    (
        "attention_only",
        "Quantize attention-side weight matmuls only; leave FFN dense matmuls float",
    ),
    (
        "ffn_only",
        "Quantize FFN and output dense matmuls only; leave attention-side weight matmuls float",
    ),
    (
        "attention_proj_only",
        "Quantize only attention projection matmuls; leave attention output and all FFN dense matmuls float",
    ),
    (
        "attention_only_layer_11_float",
        "Attention-only baseline, but keep all quantizable attention matmuls in layer 11 float",
    ),
    (
        "attention_only_layers_10_11_float",
        "Attention-only baseline, but keep all quantizable attention matmuls in layers 10-11 float",
    ),
    (
        "attention_only_attention_output_layers_8_11_float",
        "Attention-only baseline, but keep upper-layer attention output dense matmuls float",
    ),
    (
        "attention_proj_only_layer_11_float",
        "Projection-only attention quantization, with layer 11 attention projections also left float",
    ),
    (
        "attention_proj_only_layers_10_11_float",
        "Projection-only attention quantization, with layers 10-11 attention projections also left float",
    ),
]


@dataclass
class CandidateSpec:
    name: str
    description: str
    output_path: pathlib.Path
    nodes_to_exclude: list[str] = field(default_factory=list)
    generate: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and benchmark structured quantization exclusion-family candidates "
            "for the mDeBERTa ONNX model."
        )
    )
    parser.add_argument(
        "--float-model",
        default=str(DEFAULT_FLOAT_MODEL),
        help=f"Reference float ONNX model (default: {DEFAULT_FLOAT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for generated family-search candidates (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--eval-binary",
        default=str(DEFAULT_EVAL_BINARY),
        help=f"nli-eval binary used for benchmarking (default: {DEFAULT_EVAL_BINARY})",
    )
    parser.add_argument(
        "--quantize-script",
        default=str(DEFAULT_QUANTIZE_SCRIPT),
        help=f"Quantization driver script (default: {DEFAULT_QUANTIZE_SCRIPT})",
    )
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON),
        help=f"Python interpreter used to run tooling (default: {DEFAULT_PYTHON})",
    )
    parser.add_argument(
        "--benchmark",
        dest="benchmarks",
        action="append",
        default=[],
        help="Benchmark TSV to include. Repeat to add more. Defaults to the non-overlapping suite.",
    )
    parser.add_argument(
        "--candidate",
        dest="candidate_names",
        action="append",
        default=[],
        help="Candidate name to include. Repeat to filter the sweep.",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="List available candidate families and exit.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate requested candidates but do not benchmark them.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip candidate generation and benchmark existing files only.",
    )
    parser.add_argument(
        "--backend",
        default="cpu",
        help="Backend passed to nli-eval (default: cpu)",
    )
    parser.add_argument(
        "--max-disagreements",
        type=int,
        default=0,
        help="Maximum disagreements to print per benchmark run (default: 0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing candidate files and rerun benchmarking for the selected candidates.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume benchmark work from an existing --summary-json file and checkpoint after each candidate.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue benchmarking other candidates if one candidate fails.",
    )
    parser.add_argument(
        "--show-slices",
        action="store_true",
        help="Print per-slice benchmark details in the final summary.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional path to write the full summary as JSON.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional path to write the aggregate summary as CSV.",
    )
    return parser.parse_args()
def make_candidate_specs(
    output_dir: pathlib.Path,
    families: dict[str, object],
) -> list[CandidateSpec]:
    layer_all = families["layer_all"]

    return [
        CandidateSpec(
            name="current_best_reference",
            description="Existing best candidate from the single-example exclusion search",
            output_path=DEFAULT_EXISTING_BEST,
            nodes_to_exclude=[],
            generate=False,
        ),
        CandidateSpec(
            name="baseline_dynamic_matmul",
            description="Baseline dynamic MatMul quantization with no exclusions",
            output_path=DEFAULT_BASELINE_DYNAMIC,
            nodes_to_exclude=[],
            generate=True,
        ),
        CandidateSpec(
            name="ffn_layers_8_11_float",
            description="Keep FFN intermediate/output dense matmuls float in layers 8-11",
            output_path=output_dir / "dynamic_qint8_matmul_ffn_layers_8_11_float.onnx",
            nodes_to_exclude=layer_subset(families["ffn_all"], 8, 11),
        ),
        CandidateSpec(
            name="ffn_layers_6_11_float",
            description="Keep FFN intermediate/output dense matmuls float in layers 6-11",
            output_path=output_dir / "dynamic_qint8_matmul_ffn_layers_6_11_float.onnx",
            nodes_to_exclude=layer_subset(families["ffn_all"], 6, 11),
        ),
        CandidateSpec(
            name="ffn_layers_4_11_float",
            description="Keep FFN intermediate/output dense matmuls float in layers 4-11",
            output_path=output_dir / "dynamic_qint8_matmul_ffn_layers_4_11_float.onnx",
            nodes_to_exclude=layer_subset(families["ffn_all"], 4, 11),
        ),
        CandidateSpec(
            name="attention_output_layers_8_11_float",
            description="Keep attention output dense matmuls float in layers 8-11",
            output_path=output_dir / (
                "dynamic_qint8_matmul_attention_output_layers_8_11_float.onnx"
            ),
            nodes_to_exclude=layer_subset(families["attention_output"], 8, 11),
        ),
        CandidateSpec(
            name="layer_11_block_float",
            description="Keep the full quantizable block float in layer 11",
            output_path=output_dir / "dynamic_qint8_matmul_layer_11_block_float.onnx",
            nodes_to_exclude=layer_all.get(11, []),
        ),
        CandidateSpec(
            name="layers_10_11_block_float",
            description="Keep the full quantizable block float in layers 10-11",
            output_path=output_dir / "dynamic_qint8_matmul_layers_10_11_block_float.onnx",
            nodes_to_exclude=sorted(layer_all.get(10, []) + layer_all.get(11, [])),
        ),
        CandidateSpec(
            name="attention_only",
            description="Quantize attention-side weight matmuls only; leave FFN dense matmuls float",
            output_path=output_dir / "dynamic_qint8_matmul_attention_only.onnx",
            nodes_to_exclude=families["ffn_all"],
        ),
        CandidateSpec(
            name="ffn_only",
            description="Quantize FFN and output dense matmuls only; leave attention-side weight matmuls float",
            output_path=output_dir / "dynamic_qint8_matmul_ffn_only.onnx",
            nodes_to_exclude=families["attention_all"],
        ),
        CandidateSpec(
            name="attention_proj_only",
            description="Quantize only attention projection matmuls; leave attention output and all FFN dense matmuls float",
            output_path=output_dir / "dynamic_qint8_matmul_attention_proj_only.onnx",
            nodes_to_exclude=merge_nodes(
                families["ffn_all"],
                families["attention_output"],
            ),
        ),
        CandidateSpec(
            name="attention_only_layer_11_float",
            description="Attention-only baseline, but keep all quantizable attention matmuls in layer 11 float",
            output_path=output_dir / "dynamic_qint8_matmul_attention_only_layer_11_float.onnx",
            nodes_to_exclude=merge_nodes(
                families["ffn_all"],
                layer_subset(families["attention_all"], 11, 11),
            ),
        ),
        CandidateSpec(
            name="attention_only_layers_10_11_float",
            description="Attention-only baseline, but keep all quantizable attention matmuls in layers 10-11 float",
            output_path=output_dir / "dynamic_qint8_matmul_attention_only_layers_10_11_float.onnx",
            nodes_to_exclude=merge_nodes(
                families["ffn_all"],
                layer_subset(families["attention_all"], 10, 11),
            ),
        ),
        CandidateSpec(
            name="attention_only_attention_output_layers_8_11_float",
            description="Attention-only baseline, but keep upper-layer attention output dense matmuls float",
            output_path=output_dir / (
                "dynamic_qint8_matmul_attention_only_attention_output_layers_8_11_float.onnx"
            ),
            nodes_to_exclude=merge_nodes(
                families["ffn_all"],
                layer_subset(families["attention_output"], 8, 11),
            ),
        ),
        CandidateSpec(
            name="attention_proj_only_layer_11_float",
            description="Projection-only attention quantization, with layer 11 attention projections also left float",
            output_path=output_dir / (
                "dynamic_qint8_matmul_attention_proj_only_layer_11_float.onnx"
            ),
            nodes_to_exclude=merge_nodes(
                families["ffn_all"],
                families["attention_output"],
                layer_subset(families["attention_proj"], 11, 11),
            ),
        ),
        CandidateSpec(
            name="attention_proj_only_layers_10_11_float",
            description="Projection-only attention quantization, with layers 10-11 attention projections also left float",
            output_path=output_dir / (
                "dynamic_qint8_matmul_attention_proj_only_layers_10_11_float.onnx"
            ),
            nodes_to_exclude=merge_nodes(
                families["ffn_all"],
                families["attention_output"],
                layer_subset(families["attention_proj"], 10, 11),
            ),
        ),
    ]


def validate_paths(args: argparse.Namespace, benchmarks: list[pathlib.Path]) -> None:
    float_model = pathlib.Path(args.float_model)
    eval_binary = pathlib.Path(args.eval_binary)
    quantize_script = pathlib.Path(args.quantize_script)
    python_executable = pathlib.Path(args.python)

    if not float_model.is_file():
        raise RuntimeError(f"Float model not found: {float_model}")
    if not eval_binary.exists():
        raise RuntimeError(f"nli-eval binary not found: {eval_binary}")
    if not quantize_script.is_file():
        raise RuntimeError(f"Quantization script not found: {quantize_script}")
    if not python_executable.exists():
        raise RuntimeError(f"Python interpreter not found: {python_executable}")
    for benchmark in benchmarks:
        if not benchmark.is_file():
            raise RuntimeError(f"Benchmark TSV not found: {benchmark}")


def parse_eval_metrics(output: str) -> dict[str, object]:
    metrics = {}
    for line in output.splitlines():
        if line.startswith("examples: "):
            metrics["examples"] = int(line.split(": ", 1)[1])
        elif line.startswith("labeled_examples: "):
            metrics["labeled_examples"] = int(line.split(": ", 1)[1])
        elif line.startswith("primary_accuracy: "):
            match = re.search(r"\((\d+)/(\d+)\)$", line)
            if not match:
                raise RuntimeError(f"Could not parse primary_accuracy line: {line}")
            metrics["primary_correct"] = int(match.group(1))
            metrics["primary_total"] = int(match.group(2))
            metrics["primary_accuracy"] = metrics["primary_correct"] / max(
                metrics["primary_total"], 1
            )
        elif line.startswith("compare_accuracy: "):
            match = re.search(r"\((\d+)/(\d+)\)$", line)
            if not match:
                raise RuntimeError(f"Could not parse compare_accuracy line: {line}")
            metrics["compare_correct"] = int(match.group(1))
            metrics["compare_total"] = int(match.group(2))
            metrics["compare_accuracy"] = metrics["compare_correct"] / max(
                metrics["compare_total"], 1
            )
        elif line.startswith("model_agreement: "):
            match = re.search(r"\((\d+)/(\d+)\)$", line)
            if not match:
                raise RuntimeError(f"Could not parse model_agreement line: {line}")
            metrics["model_agreements"] = int(match.group(1))
            metrics["model_agreement_total"] = int(match.group(2))
            metrics["model_agreement"] = metrics["model_agreements"] / max(
                metrics["model_agreement_total"], 1
            )

    required = (
        "examples",
        "labeled_examples",
        "primary_correct",
        "compare_correct",
        "model_agreements",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(
            "Could not parse nli-eval output. Missing fields: "
            + ", ".join(missing)
            + "\nOutput was:\n"
            + output
        )
    return metrics


def validate_candidate_output(model_path: pathlib.Path) -> tuple[bool, str]:
    if not model_path.exists():
        return False, "missing"
    if model_path.stat().st_size <= 0:
        return False, "zero-byte file"

    try:
        import onnx

        model = onnx.load(model_path, load_external_data=False)
    except Exception as exc:
        return False, f"failed to load ONNX: {exc}"

    if len(model.graph.node) == 0 or len(model.graph.output) == 0:
        return False, "empty graph"
    return True, "ok"


def run_quantization(
    args: argparse.Namespace,
    candidate: CandidateSpec,
) -> None:
    candidate.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_valid, output_reason = validate_candidate_output(candidate.output_path)
    if output_valid and not args.force:
        print(f"reuse: {candidate.name} -> {candidate.output_path}")
        return

    if candidate.output_path.exists() and not output_valid:
        print(
            f"regenerate-invalid: {candidate.name} -> {candidate.output_path} "
            f"({output_reason})"
        )

    command = [
        str(pathlib.Path(args.python)),
        str(pathlib.Path(args.quantize_script)),
        "--preset",
        "single",
        "--input",
        str(pathlib.Path(args.float_model)),
        "--output",
        str(candidate.output_path),
        "--op-type",
        "MatMul",
    ]
    if candidate.nodes_to_exclude:
        command.extend(["--nodes-to-exclude", ",".join(candidate.nodes_to_exclude)])
    if args.force or candidate.output_path.exists():
        command.append("--force")

    print(f"generate: {candidate.name}")
    subprocess.run(command, check=True)


def run_benchmark(
    args: argparse.Namespace,
    benchmark_path: pathlib.Path,
    candidate: CandidateSpec,
) -> dict[str, object]:
    command = [
        str(pathlib.Path(args.eval_binary)),
        "-b",
        args.backend,
        "--model",
        str(pathlib.Path(args.float_model)),
        "--compare-model",
        str(candidate.output_path),
        "--max-disagreements",
        str(args.max_disagreements),
        str(benchmark_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    metrics = parse_eval_metrics(completed.stdout)
    metrics["benchmark"] = str(benchmark_path)
    return metrics


def aggregate_results(
    results: list[dict[str, object]],
) -> dict[str, object]:
    examples = sum(item["examples"] for item in results)
    labeled_examples = sum(item["labeled_examples"] for item in results)
    primary_correct = sum(item["primary_correct"] for item in results)
    compare_correct = sum(item["compare_correct"] for item in results)
    model_agreements = sum(item["model_agreements"] for item in results)

    primary_accuracy = primary_correct / max(labeled_examples, 1)
    compare_accuracy = compare_correct / max(labeled_examples, 1)
    agreement = model_agreements / max(examples, 1)
    return {
        "examples": examples,
        "labeled_examples": labeled_examples,
        "primary_correct": primary_correct,
        "compare_correct": compare_correct,
        "model_agreements": model_agreements,
        "primary_accuracy": primary_accuracy,
        "compare_accuracy": compare_accuracy,
        "agreement": agreement,
        "delta_vs_float": compare_accuracy - primary_accuracy,
    }


def load_resume_payload(summary_path: pathlib.Path) -> dict[str, object]:
    if not summary_path.exists():
        return {"results": [], "failures": []}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def remove_summary(summaries: list[dict[str, object]], candidate_name: str) -> None:
    summaries[:] = [
        item for item in summaries if item.get("candidate") != candidate_name
    ]


def upsert_summary(summaries: list[dict[str, object]], summary: dict[str, object]) -> None:
    remove_summary(summaries, summary["candidate"])
    summaries.append(summary)


def upsert_failure(
    failures: list[dict[str, object]],
    candidate_name: str,
    error: str,
) -> None:
    failures[:] = [
        item for item in failures if item.get("candidate") != candidate_name
    ]
    failures.append({"candidate": candidate_name, "error": error})


def remove_failure(failures: list[dict[str, object]], candidate_name: str) -> None:
    failures[:] = [
        item for item in failures if item.get("candidate") != candidate_name
    ]


def sort_summaries(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        summaries,
        key=lambda item: (
            item["aggregate"]["compare_accuracy"],
            item["aggregate"]["agreement"],
            -item["exclude_count"],
        ),
        reverse=True,
    )


def build_table_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for index, summary in enumerate(summaries, start=1):
        rows.append(
            {
                "rank": index,
                "candidate": summary["candidate"],
                "description": summary["description"],
                "path": summary["path"],
                "generated": summary["generated"],
                "exclude_count": summary["exclude_count"],
                "primary_accuracy": summary["aggregate"]["primary_accuracy"],
                "compare_accuracy": summary["aggregate"]["compare_accuracy"],
                "delta_vs_float": summary["aggregate"]["delta_vs_float"],
                "agreement": summary["aggregate"]["agreement"],
                "per_benchmark": summary["per_benchmark"],
            }
        )
    return rows


def write_summary_json(summary_path: pathlib.Path, payload: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(summary_path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "candidate",
        "description",
        "path",
        "generated",
        "exclude_count",
        "primary_accuracy",
        "compare_accuracy",
        "delta_vs_float",
        "agreement",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def print_candidate_table(rows: list[dict[str, object]], show_slices: bool) -> None:
    if not rows:
        print("No candidate results.")
        return

    header = (
        f"{'rank':>4}  {'candidate':<34} {'acc':>8} {'delta':>8} "
        f"{'agree':>8} {'exclude':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['rank']:>4}  "
            f"{row['candidate']:<34} "
            f"{row['compare_accuracy'] * 100:>7.2f}% "
            f"{row['delta_vs_float'] * 100:>+7.2f} "
            f"{row['agreement'] * 100:>7.2f}% "
            f"{row['exclude_count']:>8}"
        )
        if show_slices:
            for item in row["per_benchmark"]:
                benchmark_name = pathlib.Path(item["benchmark"]).name
                print(
                    f"      {benchmark_name:<42} "
                    f"cmp={item['compare_accuracy'] * 100:>6.2f}% "
                    f"float={item['primary_accuracy'] * 100:>6.2f}% "
                    f"agree={item['model_agreement'] * 100:>6.2f}%"
                )


def write_checkpoint(
    args: argparse.Namespace,
    benchmarks: list[pathlib.Path],
    summaries: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sorted_summaries = sort_summaries(list(summaries))
    table_rows = build_table_rows(sorted_summaries)
    payload = {
        "float_model": str(pathlib.Path(args.float_model)),
        "benchmarks": [str(path) for path in benchmarks],
        "results": sorted_summaries,
        "failures": failures,
    }
    if args.summary_json:
        write_summary_json(pathlib.Path(args.summary_json), payload)
    if args.summary_csv:
        write_summary_csv(pathlib.Path(args.summary_csv), table_rows)
    return sorted_summaries, table_rows


def main() -> int:
    default_python = pathlib.Path(DEFAULT_PYTHON)
    if default_python.exists() and pathlib.Path(sys.executable).resolve() != default_python.resolve():
        os.execv(str(default_python), [str(default_python), __file__, *sys.argv[1:]])

    args = parse_args()
    if args.list_candidates:
        for name, description in STATIC_CANDIDATE_DESCRIPTIONS:
            print(f"{name}: {description}")
        return 0
    if args.resume and not args.summary_json:
        raise RuntimeError("--resume requires --summary-json")

    benchmarks = [pathlib.Path(path) for path in (args.benchmarks or DEFAULT_BENCHMARKS)]
    validate_paths(args, benchmarks)

    families = load_matmul_families(pathlib.Path(args.float_model))
    all_candidates = make_candidate_specs(pathlib.Path(args.output_dir), families)

    if args.generate_only and args.benchmark_only:
        raise RuntimeError("--generate-only and --benchmark-only cannot be combined")

    selected_names = set(args.candidate_names)
    candidates = [
        candidate
        for candidate in all_candidates
        if not selected_names or candidate.name in selected_names
    ]
    if not candidates:
        raise RuntimeError("No candidates selected")

    unknown_names = selected_names - {candidate.name for candidate in all_candidates}
    if unknown_names:
        raise RuntimeError(
            "Unknown candidate names: " + ", ".join(sorted(unknown_names))
        )

    print("benchmarks:")
    for benchmark in benchmarks:
        print(f"  - {benchmark}")

    if not args.benchmark_only:
        for candidate in candidates:
            if not candidate.generate:
                print(f"reference: {candidate.name} -> {candidate.output_path}")
                continue
            run_quantization(args, candidate)

    if args.generate_only:
        return 0

    summaries = []
    failures = []
    if args.resume:
        resume_payload = load_resume_payload(pathlib.Path(args.summary_json))
        expected_benchmarks = [str(path) for path in benchmarks]
        resume_benchmarks = resume_payload.get("benchmarks", expected_benchmarks)
        if resume_benchmarks != expected_benchmarks:
            raise RuntimeError(
                "Resume summary benchmarks do not match the current run"
            )
        resume_float_model = resume_payload.get(
            "float_model",
            str(pathlib.Path(args.float_model)),
        )
        if resume_float_model != str(pathlib.Path(args.float_model)):
            raise RuntimeError("Resume summary float_model does not match the current run")
        summaries = list(resume_payload.get("results", []))
        failures = list(resume_payload.get("failures", []))

    for candidate in candidates:
        cached_summary = next(
            (
                item
                for item in summaries
                if item.get("candidate") == candidate.name
            ),
            None,
        )
        output_valid, output_reason = validate_candidate_output(candidate.output_path)
        if cached_summary and output_valid and not args.force:
            print(f"resume: {candidate.name}")
            continue
        if cached_summary and not output_valid:
            print(
                f"resume-stale: {candidate.name} -> {candidate.output_path} "
                f"({output_reason})"
            )
            remove_summary(summaries, candidate.name)

        if not output_valid:
            if candidate.generate and not args.benchmark_only:
                run_quantization(args, candidate)
                output_valid, output_reason = validate_candidate_output(candidate.output_path)
            if not output_valid:
                message = (
                    f"candidate file is invalid: {candidate.output_path} ({output_reason})"
                )
                if args.keep_going:
                    upsert_failure(failures, candidate.name, message)
                    print(f"skip: {candidate.name}: {message}")
                    if args.summary_json or args.summary_csv:
                        write_checkpoint(args, benchmarks, summaries, failures)
                    continue
                raise RuntimeError(message)

        if not candidate.output_path.exists():
            message = f"candidate file not found: {candidate.output_path}"
            if args.keep_going:
                upsert_failure(failures, candidate.name, message)
                print(f"skip: {candidate.name}: {message}")
                if args.summary_json or args.summary_csv:
                    write_checkpoint(args, benchmarks, summaries, failures)
                continue
            raise RuntimeError(message)

        print(f"benchmark: {candidate.name}")
        try:
            per_benchmark = [
                run_benchmark(args, benchmark, candidate) for benchmark in benchmarks
            ]
            aggregate = aggregate_results(per_benchmark)
        except subprocess.CalledProcessError as exc:
            message = exc.stdout or exc.stderr or str(exc)
            if args.keep_going:
                upsert_failure(failures, candidate.name, message)
                print(f"failed: {candidate.name}")
                if args.summary_json or args.summary_csv:
                    write_checkpoint(args, benchmarks, summaries, failures)
                continue
            raise

        upsert_summary(
            summaries,
            {
                "candidate": candidate.name,
                "description": candidate.description,
                "path": str(candidate.output_path),
                "generated": candidate.generate,
                "exclude_count": len(candidate.nodes_to_exclude),
                "exclude_nodes": candidate.nodes_to_exclude,
                "aggregate": aggregate,
                "per_benchmark": per_benchmark,
            },
        )
        remove_failure(failures, candidate.name)
        if args.summary_json or args.summary_csv:
            write_checkpoint(args, benchmarks, summaries, failures)

    summaries, table_rows = write_checkpoint(args, benchmarks, summaries, failures)

    print()
    print_candidate_table(table_rows, args.show_slices)

    if failures:
        print()
        print("failures:")
        for failure in failures:
            print(f"  - {failure['candidate']}: {failure['error']}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stderr.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
