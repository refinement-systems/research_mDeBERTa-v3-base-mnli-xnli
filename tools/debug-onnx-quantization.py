#!/usr/bin/env python3

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import venv


DEFAULT_FLOAT_MODEL = "models/mdeberta/onnx/model.onnx"
DEFAULT_QUANTIZED_MODEL = "models/mdeberta/onnx/model_quantized.onnx"
DEFAULT_TOKENIZER_SOURCE = "models/mdeberta"
DEFAULT_PREMISE = (
    "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
)
DEFAULT_HYPOTHESIS = "Emmanuel Macron is the President of France"
DEFAULT_VENV = ".venv" if os.path.exists(".venv") else os.path.join(
    tempfile.gettempdir(), "nli-onnx-tools-venv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug quantization drift by collecting intermediate activations from the "
            "float and quantized ONNX graphs and ranking shared tensors by error."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              .venv/bin/python tools/debug-onnx-quantization.py
              .venv/bin/python tools/debug-onnx-quantization.py --op-type MatMul --op-type Add
              .venv/bin/python tools/debug-onnx-quantization.py --tsv tests/data/nli_eval_fixture.tsv --max-examples-per-source 3
              .venv/bin/python tools/debug-onnx-quantization.py --input-dir benchmarks/nli --pattern 'xnli-*.tsv' --max-examples-per-source 6
            """
        ),
    )
    parser.add_argument(
        "--float-model",
        default=DEFAULT_FLOAT_MODEL,
        help=f"Reference float ONNX model (default: {DEFAULT_FLOAT_MODEL})",
    )
    parser.add_argument(
        "--quantized-model",
        default=DEFAULT_QUANTIZED_MODEL,
        help=f"Quantized ONNX model to debug (default: {DEFAULT_QUANTIZED_MODEL})",
    )
    parser.add_argument(
        "--tokenizer-source",
        default=DEFAULT_TOKENIZER_SOURCE,
        help=f"Local tokenizer/model directory used by transformers (default: {DEFAULT_TOKENIZER_SOURCE})",
    )
    parser.add_argument("--premise", default=DEFAULT_PREMISE, help="Single-example premise")
    parser.add_argument(
        "--hypothesis",
        default=DEFAULT_HYPOTHESIS,
        help="Single-example hypothesis",
    )
    parser.add_argument(
        "--tsv",
        dest="tsv_paths",
        action="append",
        default=[],
        help="TSV with premise/hypothesis columns. Repeat to debug multiple files.",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Optional directory containing TSV files to sample for debugging.",
    )
    parser.add_argument(
        "--pattern",
        default="*.tsv",
        help="Glob pattern used with --input-dir (default: *.tsv)",
    )
    parser.add_argument(
        "--max-examples-per-source",
        type=int,
        default=1,
        help="Maximum number of examples to load from each TSV source (default: 1)",
    )
    parser.add_argument(
        "--max-total-examples",
        type=int,
        default=0,
        help="Optional cap across all loaded examples after per-source limits (default: unlimited)",
    )
    parser.add_argument(
        "--op-type",
        dest="op_types",
        action="append",
        default=[],
        help="Operator type to save from the augmented models. Repeat to add more. Defaults to MatMul.",
    )
    parser.add_argument(
        "--all-ops",
        action="store_true",
        help="Save all intermediate tensors instead of only selected op types.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many top-drifting tensors to print (default: 20)",
    )
    parser.add_argument(
        "--suggest-exclusions",
        type=int,
        default=8,
        help=(
            "How many actual quantized MatMul node names to suggest for nodes_to_exclude "
            "(default: 8)"
        ),
    )
    parser.add_argument(
        "--suggest-by",
        choices=["max_abs", "mean_abs", "rmse", "max_rel", "mean_rel", "rmse_rel"],
        default="rmse_rel",
        help="Metric used to rank suggested exclusions (default: rmse_rel)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of the human-readable report.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Create a virtualenv and install onnx+onnxruntime+transformers+sentencepiece if needed",
    )
    parser.add_argument(
        "--venv",
        default=DEFAULT_VENV,
        help=f"Virtualenv path used with --install-deps (default: {DEFAULT_VENV})",
    )
    return parser.parse_args()


def module_available(module_name: str) -> bool:
    code = (
        "import importlib.util, sys;"
        f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    )
    return subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def ensure_python(args: argparse.Namespace) -> str:
    needed = ("onnx", "onnxruntime", "transformers", "sentencepiece", "numpy")
    if all(module_available(module) for module in needed):
        return sys.executable

    if not args.install_deps:
        raise RuntimeError(
            "Python packages 'onnx', 'onnxruntime', 'transformers', 'sentencepiece', and "
            "'numpy' are required. Install them in the active environment or re-run with "
            "--install-deps."
        )

    venv_dir = pathlib.Path(args.venv)
    python_path = venv_dir / "bin" / "python3"
    if not python_path.exists():
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)

    subprocess.run(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "onnx",
            "onnxruntime",
            "transformers",
            "sentencepiece",
            "numpy",
        ],
        check=True,
    )
    return str(python_path)


def main() -> int:
    args = parse_args()
    python_executable = ensure_python(args)

    op_types = [] if args.all_ops else (args.op_types or ["MatMul"])

    helper = r"""
import csv
import json
import math
import pathlib
import sys
import tempfile
from dataclasses import dataclass

import numpy
import onnx
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations,
    create_weight_matching,
    modify_model_output_intermediate_tensors,
)
from transformers import AutoTokenizer


@dataclass
class Example:
    premise: str
    hypothesis: str
    example_id: str


class FeedListDataReader(CalibrationDataReader):
    def __init__(self, feeds):
        self._feeds = feeds
        self._index = 0

    def get_next(self):
        if self._index >= len(self._feeds):
            return None
        item = self._feeds[self._index]
        self._index += 1
        return item

    def rewind(self):
        self._index = 0


def discover_tsv_paths(explicit_paths, input_dir, pattern):
    paths = []
    for path in explicit_paths:
        paths.append(pathlib.Path(path))
    if input_dir:
        paths.extend(sorted(pathlib.Path(input_dir).glob(pattern)))
    return [path for path in paths if path.is_file()]


def read_examples(tsv_paths, input_dir, pattern, premise, hypothesis, max_examples_per_source, max_total_examples):
    source_counts = {}
    discovered_paths = discover_tsv_paths(tsv_paths, input_dir, pattern)
    if not discovered_paths:
        return [Example(premise=premise, hypothesis=hypothesis, example_id="example-1")], source_counts

    examples = []
    for tsv_path in discovered_paths:
        loaded_for_source = 0
        with open(tsv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "premise" not in reader.fieldnames or "hypothesis" not in reader.fieldnames:
                raise RuntimeError(f"TSV must include premise and hypothesis columns: {tsv_path}")
            for index, row in enumerate(reader):
                if max_examples_per_source > 0 and loaded_for_source >= max_examples_per_source:
                    break
                if max_total_examples > 0 and len(examples) >= max_total_examples:
                    break
                examples.append(
                    Example(
                        premise=row["premise"],
                        hypothesis=row["hypothesis"],
                        example_id=row.get("id") or f"{tsv_path.stem}-row-{index + 1}",
                    )
                )
                loaded_for_source += 1
        source_counts[str(tsv_path)] = loaded_for_source
        if max_total_examples > 0 and len(examples) >= max_total_examples:
            break

    if not examples:
        raise RuntimeError("No examples were loaded for debugging")
    return examples, source_counts


def model_input_names(model_path):
    model = onnx.load(model_path, load_external_data=False)
    return [value_info.name for value_info in model.graph.input]


def output_name_metadata(model_path):
    model = onnx.load(model_path, load_external_data=False)
    mapping = {}
    for node in model.graph.node:
        for output_name in node.output:
            if output_name:
                mapping[output_name] = {
                    "op_type": node.op_type,
                    "node_name": node.name,
                }
    return mapping


def quantized_source_nodes(model_path):
    model = onnx.load(model_path, load_external_data=False)
    quantized = set()
    for node in model.graph.node:
        if node.op_type in ("MatMulInteger", "QLinearMatMul"):
            node_name = node.name
            if node_name.endswith("_quant"):
                quantized.add(node_name[:-len("_quant")])
            else:
                quantized.add(node_name)
            continue
        if node.op_type == "Mul" and node.name.endswith("_quant_scales_mul"):
            quantized.add(node.name[:-len("_quant_scales_mul")])
    return quantized


def build_feeds(tokenizer_source, model_inputs, examples):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=pathlib.Path(tokenizer_source).exists(),
        use_fast=True,
    )
    feeds = []
    for example in examples:
        encoded = tokenizer(
            example.premise,
            example.hypothesis,
            truncation=True,
            return_tensors="np",
        )
        feed = {}
        for name in model_inputs:
            if name not in encoded:
                continue
            value = encoded[name]
            if value.dtype != numpy.int64:
                value = value.astype(numpy.int64, copy=False)
            feed[name] = value
        feeds.append(feed)
    return feeds


def flatten_example_arrays(arrays):
    if not arrays:
        return numpy.array([], dtype=numpy.float32)
    flattened = []
    for array in arrays:
        if array is None:
            continue
        np_array = numpy.asarray(array)
        flattened.append(np_array.astype(numpy.float32, copy=False).reshape(-1))
    if not flattened:
        return numpy.array([], dtype=numpy.float32)
    return numpy.concatenate(flattened)


def compare_activations(float_activations, quant_activations, float_meta, quant_meta):
    shared_names = sorted(set(float_activations) & set(quant_activations))
    comparisons = []
    for name in shared_names:
        float_values = flatten_example_arrays(float_activations[name])
        quant_values = flatten_example_arrays(quant_activations[name])
        if float_values.size == 0 or quant_values.size == 0:
            continue
        if float_values.shape != quant_values.shape:
            continue
        delta = quant_values - float_values
        abs_delta = numpy.abs(delta)
        comparisons.append(
            {
                "name": name,
                "float_op_type": float_meta.get(name, {}).get("op_type", ""),
                "quant_op_type": quant_meta.get(name, {}).get("op_type", ""),
                "float_node_name": float_meta.get(name, {}).get("node_name", ""),
                "quant_node_name": quant_meta.get(name, {}).get("node_name", ""),
                "element_count": int(float_values.size),
                "mean_abs": float(abs_delta.mean()),
                "max_abs": float(abs_delta.max()),
                "rmse": float(numpy.sqrt(numpy.mean(delta * delta))),
                "float_mean_abs": float(numpy.abs(float_values).mean()),
                "quant_mean_abs": float(numpy.abs(quant_values).mean()),
            }
        )
    return comparisons


def enrich_comparisons(comparisons):
    for item in comparisons:
        scale = max(item["float_mean_abs"], 1e-12)
        item["mean_rel"] = float(item["mean_abs"] / scale)
        item["max_rel"] = float(item["max_abs"] / scale)
        item["rmse_rel"] = float(item["rmse"] / scale)
    return comparisons


def build_node_ranking(comparisons, suggest_metric, quantized_sources):
    nodes = {}
    for item in comparisons:
        node_name = item["float_node_name"]
        if not node_name:
            continue
        entry = nodes.setdefault(
            node_name,
            {
                "node_name": node_name,
                "float_op_type": item["float_op_type"],
                "quant_op_type": item["quant_op_type"],
                "is_quantized_source": node_name in quantized_sources,
                "output_count": 0,
                "element_count": 0,
                "mean_abs_sum": 0.0,
                "rmse_sum": 0.0,
                "float_mean_abs_sum": 0.0,
                "max_abs": 0.0,
                "max_rel": 0.0,
                "mean_rel": 0.0,
                "rmse_rel": 0.0,
            },
        )
        entry["output_count"] += 1
        entry["element_count"] += item["element_count"]
        entry["mean_abs_sum"] += item["mean_abs"]
        entry["rmse_sum"] += item["rmse"]
        entry["float_mean_abs_sum"] += item["float_mean_abs"]
        entry["max_abs"] = max(entry["max_abs"], item["max_abs"])
        entry["max_rel"] = max(entry["max_rel"], item["max_rel"])

    ranking = []
    for entry in nodes.values():
        output_count = max(entry["output_count"], 1)
        scale = max(entry["float_mean_abs_sum"] / output_count, 1e-12)
        mean_abs = entry["mean_abs_sum"] / output_count
        rmse = entry["rmse_sum"] / output_count
        node_summary = {
            "node_name": entry["node_name"],
            "float_op_type": entry["float_op_type"],
            "quant_op_type": entry["quant_op_type"],
            "is_quantized_source": entry["is_quantized_source"],
            "output_count": entry["output_count"],
            "element_count": entry["element_count"],
            "mean_abs": float(mean_abs),
            "max_abs": float(entry["max_abs"]),
            "rmse": float(rmse),
            "mean_rel": float(mean_abs / scale),
            "max_rel": float(entry["max_rel"]),
            "rmse_rel": float(rmse / scale),
        }
        node_summary["suggest_metric"] = float(node_summary[suggest_metric])
        ranking.append(node_summary)

    ranking.sort(key=lambda item: item["suggest_metric"], reverse=True)
    return ranking


def shared_initializer_summary(float_model_path, quant_model_path):
    float_model = onnx.load(float_model_path, load_external_data=False)
    quant_model = onnx.load(quant_model_path, load_external_data=False)
    float_inits = {init.name for init in float_model.graph.initializer}
    quant_inits = {init.name for init in quant_model.graph.initializer}
    shared = float_inits & quant_inits
    quant_only = quant_inits - float_inits
    return {
        "float_initializer_count": len(float_inits),
        "quant_initializer_count": len(quant_inits),
        "shared_initializer_count": len(shared),
        "quant_only_initializer_count": len(quant_only),
    }


payload = json.loads(sys.argv[1])
examples, source_counts = read_examples(
    payload["tsv_paths"],
    payload["input_dir"],
    payload["pattern"],
    payload["premise"],
    payload["hypothesis"],
    payload["max_examples_per_source"],
    payload["max_total_examples"],
)
float_inputs = model_input_names(payload["float_model"])
quant_inputs = model_input_names(payload["quantized_model"])
if float_inputs != quant_inputs:
    raise RuntimeError(f"Model inputs differ: float={float_inputs} quantized={quant_inputs}")
feeds = build_feeds(payload["tokenizer_source"], float_inputs, examples)

with tempfile.TemporaryDirectory(prefix="nli-qdebug-") as tmp_dir:
    tmp_dir = pathlib.Path(tmp_dir)
    float_augmented = tmp_dir / "float_augmented.onnx"
    quant_augmented = tmp_dir / "quantized_augmented.onnx"

    modify_model_output_intermediate_tensors(
        payload["float_model"],
        float_augmented,
        op_types_for_saving=payload["op_types"] or None,
        save_as_external_data=True,
    )
    modify_model_output_intermediate_tensors(
        payload["quantized_model"],
        quant_augmented,
        op_types_for_saving=payload["op_types"] or None,
        save_as_external_data=True,
    )

    float_activations = collect_activations(
        str(float_augmented),
        FeedListDataReader(list(feeds)),
        execution_providers=["CPUExecutionProvider"],
    )
    quant_activations = collect_activations(
        str(quant_augmented),
        FeedListDataReader(list(feeds)),
        execution_providers=["CPUExecutionProvider"],
    )

comparison = compare_activations(
    float_activations,
    quant_activations,
    output_name_metadata(payload["float_model"]),
    output_name_metadata(payload["quantized_model"]),
)
comparison = enrich_comparisons(comparison)
quantized_sources = quantized_source_nodes(payload["quantized_model"])
for item in comparison:
    item["is_quantized_source"] = item["float_node_name"] in quantized_sources
comparison.sort(key=lambda item: item[payload["suggest_by"]], reverse=True)
node_ranking = build_node_ranking(comparison, payload["suggest_by"], quantized_sources)

weight_matching_count = 0
weight_matching_error = ""
try:
    weight_matching_count = len(create_weight_matching(payload["float_model"], payload["quantized_model"]))
except Exception as exc:
    weight_matching_error = str(exc)

result = {
    "examples": [
        {
            "id": example.example_id,
            "premise": example.premise,
            "hypothesis": example.hypothesis,
        }
        for example in examples
    ],
    "source_counts": source_counts,
    "float_model": payload["float_model"],
    "quantized_model": payload["quantized_model"],
    "tokenizer_source": payload["tokenizer_source"],
    "op_types": payload["op_types"],
    "suggest_by": payload["suggest_by"],
    "float_activation_count": len(float_activations),
    "quantized_activation_count": len(quant_activations),
    "shared_activation_count": len(comparison),
    "quantized_source_node_count": len(quantized_sources),
    "shared_initializer_summary": shared_initializer_summary(
        payload["float_model"], payload["quantized_model"]
    ),
    "weight_matching_count": weight_matching_count,
    "weight_matching_error": weight_matching_error,
    "top_drift": comparison[: payload["top"]],
    "top_nodes": node_ranking[: payload["top"]],
    "top_quantized_nodes": [item for item in node_ranking if item["is_quantized_source"]][: payload["top"]],
    "suggested_nodes_to_exclude": [],
}

suggested = []
for item in node_ranking:
    if item["float_op_type"] != "MatMul":
        continue
    if not item["is_quantized_source"]:
        continue
    node_name = item["node_name"]
    if node_name in suggested:
        continue
    suggested.append(node_name)
    if len(suggested) >= payload["suggest_exclusions"]:
        break
result["suggested_nodes_to_exclude"] = suggested
print(json.dumps(result))
"""

    payload = {
        "float_model": args.float_model,
        "quantized_model": args.quantized_model,
        "tokenizer_source": args.tokenizer_source,
        "premise": args.premise,
        "hypothesis": args.hypothesis,
        "tsv_paths": args.tsv_paths,
        "input_dir": args.input_dir,
        "pattern": args.pattern,
        "max_examples_per_source": args.max_examples_per_source,
        "max_total_examples": args.max_total_examples,
        "op_types": op_types,
        "top": args.top,
        "suggest_exclusions": args.suggest_exclusions,
        "suggest_by": args.suggest_by,
    }
    result = subprocess.run(
        [python_executable, "-c", helper, json.dumps(payload)],
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"float_model: {report['float_model']}")
    print(f"quantized_model: {report['quantized_model']}")
    print(f"tokenizer_source: {report['tokenizer_source']}")
    print(
        f"examples: {len(report['examples'])}  "
        f"saved_op_types: {report['op_types'] or 'ALL'}  "
        f"suggest_by: {report['suggest_by']}"
    )
    if report["source_counts"]:
        print("source_counts:")
        for source, count in report["source_counts"].items():
            print(f"  - {source}: {count}")
    for example in report["examples"]:
        print(f"  - {example['id']}: {example['premise']} || {example['hypothesis']}")
    print(
        f"activation_counts: float={report['float_activation_count']} "
        f"quantized={report['quantized_activation_count']} "
        f"shared={report['shared_activation_count']}"
    )
    print(f"quantized_source_node_count: {report['quantized_source_node_count']}")
    print(f"initializer_summary: {report['shared_initializer_summary']}")
    print(
        f"weight_matching_count: {report['weight_matching_count']}"
        + (
            f"  error={report['weight_matching_error']}"
            if report["weight_matching_error"]
            else ""
        )
    )
    print("top_drift:")
    for item in report["top_drift"]:
        print(
            f"  - {item['name']}: max_abs={item['max_abs']:.6f} "
            f"mean_abs={item['mean_abs']:.6f} rmse={item['rmse']:.6f} "
            f"max_rel={item['max_rel']:.6f} mean_rel={item['mean_rel']:.6f} "
            f"rmse_rel={item['rmse_rel']:.6f} "
            f"quantized_source={item['is_quantized_source']} "
            f"elements={item['element_count']} "
            f"float_op={item['float_op_type'] or '?'} "
            f"quant_op={item['quant_op_type'] or '?'} "
            f"float_node={item['float_node_name'] or '?'} "
            f"quant_node={item['quant_node_name'] or '?'}"
        )
    print("top_nodes:")
    for item in report["top_nodes"]:
        print(
            f"  - {item['node_name']}: max_abs={item['max_abs']:.6f} "
            f"mean_abs={item['mean_abs']:.6f} rmse={item['rmse']:.6f} "
            f"max_rel={item['max_rel']:.6f} mean_rel={item['mean_rel']:.6f} "
            f"rmse_rel={item['rmse_rel']:.6f} "
            f"quantized_source={item['is_quantized_source']} "
            f"outputs={item['output_count']} elements={item['element_count']} "
            f"float_op={item['float_op_type'] or '?'} "
            f"quant_op={item['quant_op_type'] or '?'}"
        )
    print("top_quantized_nodes:")
    for item in report["top_quantized_nodes"]:
        print(
            f"  - {item['node_name']}: max_abs={item['max_abs']:.6f} "
            f"mean_abs={item['mean_abs']:.6f} rmse={item['rmse']:.6f} "
            f"max_rel={item['max_rel']:.6f} mean_rel={item['mean_rel']:.6f} "
            f"rmse_rel={item['rmse_rel']:.6f} "
            f"outputs={item['output_count']} elements={item['element_count']} "
            f"float_op={item['float_op_type'] or '?'} "
            f"quant_op={item['quant_op_type'] or '?'}"
        )
    if report["suggested_nodes_to_exclude"]:
        print("suggested_nodes_to_exclude:")
        for node_name in report["suggested_nodes_to_exclude"]:
            print(f"  - {node_name}")
        print(
            "suggested_nodes_to_exclude_csv: "
            + ",".join(report["suggested_nodes_to_exclude"])
        )

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
