#!/usr/bin/env python3

from __future__ import annotations

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
from dataclasses import dataclass

from mdeberta_onnx_quantization import (
    SUPPORTED_IGNORE_FAMILIES,
    ignored_nodes_for_family,
)


DEFAULT_INPUT = "models/mdeberta/onnx/model.onnx"
DEFAULT_OUTPUT = "models/mdeberta/onnx/candidates/nncf/dynamic_mixed_attention_only.onnx"
DEFAULT_TOKENIZER_SOURCE = "models/mdeberta"
DEFAULT_HF_SOURCE = "models/mdeberta"
DEFAULT_VENV = ".venv" if os.path.exists(".venv") else os.path.join(
    tempfile.gettempdir(), "nli-onnx-tools-venv"
)


@dataclass(frozen=True)
class Example:
    example_id: str
    premise: str
    hypothesis: str
    gold_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an ONNX candidate using NNCF PTQ or accuracy-controlled PTQ with "
            "mixed-precision-friendly transformer settings."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              .venv/bin/python tools/quantize-onnx-nncf.py \\
                --mode ptq \\
                --ignored-scope-family attention_only \\
                --calibration-tsv benchmarks/nli/mnli-train-calibration-64-per-label.tsv \\
                --calibration-tsv benchmarks/nli/xnli-en-validation-calibration-32-per-label.tsv \\
                --output models/mdeberta/onnx/candidates/nncf/ptq_attention_only.onnx

              .venv/bin/python tools/quantize-onnx-nncf.py \\
                --mode accuracy-control \\
                --metric hf_agreement \\
                --ignored-scope-family attention_proj_only \\
                --validation-tsv benchmarks/nli/mnli-train-search-validation-skip64-64-per-label.tsv \\
                --validation-tsv benchmarks/nli/xnli-en-validation-search-validation-skip32-32-per-label.tsv
            """
        ),
    )
    parser.add_argument("--input", "--src", default=DEFAULT_INPUT, dest="input")
    parser.add_argument("--output", "--dest", default=DEFAULT_OUTPUT, dest="output")
    parser.add_argument("--tokenizer-source", default=DEFAULT_TOKENIZER_SOURCE)
    parser.add_argument("--hf-source", default=DEFAULT_HF_SOURCE)
    parser.add_argument(
        "--mode",
        choices=["ptq", "accuracy-control"],
        default="ptq",
        help="Quantization mode (default: ptq).",
    )
    parser.add_argument(
        "--metric",
        choices=["gold_accuracy", "hf_agreement"],
        default="gold_accuracy",
        help="Validation metric used for accuracy-control mode (default: gold_accuracy).",
    )
    parser.add_argument(
        "--preset",
        choices=["performance", "mixed"],
        default="mixed",
        help="NNCF quantization preset (default: mixed).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=300,
        help="Calibration subset size passed to NNCF (default: 300).",
    )
    parser.add_argument(
        "--ignored-scope-family",
        choices=SUPPORTED_IGNORE_FAMILIES,
        default="none",
        help=(
            "Structured exclusion family reused from the ONNX family search "
            f"(default: none; choices: {', '.join(SUPPORTED_IGNORE_FAMILIES)})."
        ),
    )
    parser.add_argument(
        "--max-drop",
        type=float,
        default=0.01,
        help="Maximum absolute validation-metric drop for accuracy-control mode (default: 0.01).",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run ORT quant_pre_process on the input model before NNCF quantization.",
    )
    parser.add_argument(
        "--skip-preprocess-optimization",
        action="store_true",
        help="Disable graph optimization during quant_pre_process.",
    )
    parser.add_argument(
        "--disable-smooth-quant",
        action="store_true",
        help="Disable SmoothQuant inside NNCF advanced parameters.",
    )
    bias_group = parser.add_mutually_exclusive_group()
    bias_group.add_argument(
        "--fast-bias-correction",
        dest="fast_bias_correction",
        action="store_true",
        help="Use fast bias correction (default).",
    )
    bias_group.add_argument(
        "--accurate-bias-correction",
        dest="fast_bias_correction",
        action="store_false",
        help="Disable fast bias correction for a slower, more accurate pass.",
    )
    parser.set_defaults(fast_bias_correction=True)
    parser.add_argument(
        "--calibration-tsv",
        dest="calibration_tsvs",
        action="append",
        default=[],
        help="Calibration TSV with premise/hypothesis columns. Repeat to add more.",
    )
    parser.add_argument(
        "--validation-tsv",
        dest="validation_tsvs",
        action="append",
        default=[],
        help="Validation TSV with premise/hypothesis/label columns. Repeat to add more.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Create a virtualenv and install the required Python packages if needed.",
    )
    parser.add_argument("--venv", default=DEFAULT_VENV)
    return parser.parse_args()


def module_available(module_name: str) -> bool:
    code = (
        "import importlib.util, sys;"
        f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    )
    return subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def ensure_python(args: argparse.Namespace) -> str:
    needed = ("onnx", "onnxruntime", "transformers", "sentencepiece", "numpy", "nncf")
    if all(module_available(module) for module in needed):
        return sys.executable

    if not args.install_deps:
        raise RuntimeError(
            "Python packages 'onnx', 'onnxruntime', 'transformers', 'sentencepiece', "
            "'numpy', and 'nncf' are required. Install them in the active environment or "
            "re-run with --install-deps."
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
            "openvino",
            "nncf",
        ],
        check=True,
    )
    return str(python_path)


def read_examples(tsv_paths: list[pathlib.Path]) -> list[Example]:
    examples: list[Example] = []
    for tsv_path in tsv_paths:
        with tsv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "premise" not in reader.fieldnames or "hypothesis" not in reader.fieldnames:
                raise RuntimeError(f"TSV must include premise and hypothesis columns: {tsv_path}")
            for index, row in enumerate(reader):
                examples.append(
                    Example(
                        example_id=row.get("id") or f"{tsv_path.stem}-{index + 1}",
                        premise=row["premise"],
                        hypothesis=row["hypothesis"],
                        gold_label=(row.get("label") or row.get("gold_label") or "").strip(),
                    )
                )
    if not examples:
        raise RuntimeError("No examples were loaded")
    return examples


def parse_last_json_line(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Quantization helper did not emit a JSON summary")


def main() -> int:
    args = parse_args()
    python_executable = ensure_python(args)

    input_model = pathlib.Path(args.input)
    if not input_model.exists():
        raise RuntimeError(f"Input model not found: {input_model}")

    if not args.calibration_tsvs:
        raise RuntimeError("At least one --calibration-tsv is required")
    if args.mode == "accuracy-control" and not args.validation_tsvs:
        raise RuntimeError("Accuracy-control mode requires at least one --validation-tsv")

    calibration_paths = [pathlib.Path(path) for path in args.calibration_tsvs]
    validation_paths = [pathlib.Path(path) for path in args.validation_tsvs]
    for path in calibration_paths + validation_paths:
        if not path.is_file():
            raise RuntimeError(f"TSV not found: {path}")

    output_path = pathlib.Path(args.output)
    if output_path.exists() and not args.force:
        raise RuntimeError(
            f"Output already exists: {output_path}. Re-run with --force or choose a new path."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    helper = r"""
import csv
import json
import pathlib
import sys

import nncf
import numpy
import onnx
import onnxruntime
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters


LABELS = ["entailment", "neutral", "contradiction"]


def predicted_label(logits):
    return LABELS[max(range(len(logits)), key=lambda index: logits[index])]


def read_examples(paths):
    examples = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for index, row in enumerate(reader):
                examples.append(
                    {
                        "id": row.get("id") or f"{path}-{index + 1}",
                        "premise": row["premise"],
                        "hypothesis": row["hypothesis"],
                        "gold_label": (row.get("label") or row.get("gold_label") or "").strip(),
                    }
                )
    if not examples:
        raise RuntimeError("No examples were loaded")
    return examples


def build_model_inputs(tokenizer, input_names, example):
    encoded = tokenizer(
        example["premise"],
        example["hypothesis"],
        truncation=True,
        return_tensors="np",
    )
    feed = {}
    for name in input_names:
        if name not in encoded:
            continue
        value = encoded[name]
        if value.dtype != numpy.int64:
            value = value.astype(numpy.int64, copy=False)
        feed[name] = value
    return feed


def iter_validation_rows(validation_dataset):
    if hasattr(validation_dataset, "get_data"):
        return validation_dataset.get_data()
    return validation_dataset


payload = json.loads(pathlib.Path("PAYLOAD.json").read_text(encoding="utf-8"))
calibration_examples = read_examples(payload["calibration_tsvs"])
validation_examples = read_examples(payload["validation_tsvs"]) if payload["validation_tsvs"] else []

tokenizer = AutoTokenizer.from_pretrained(
    payload["tokenizer_source"],
    local_files_only=pathlib.Path(payload["tokenizer_source"]).exists(),
    use_fast=True,
)

resolved_input_model = pathlib.Path(payload["input"])
preprocess_dir = None
if payload["preprocess"]:
    import tempfile

    preprocess_dir = tempfile.TemporaryDirectory(prefix="nli-nncf-preprocess-")
    preprocessed_model = pathlib.Path(preprocess_dir.name) / "preprocessed.onnx"
    quant_pre_process(
        str(resolved_input_model),
        str(preprocessed_model),
        skip_optimization=payload["skip_preprocess_optimization"],
        save_as_external_data=False,
    )
    resolved_input_model = preprocessed_model

input_names = [
    value_info.name
    for value_info in onnx.load(resolved_input_model, load_external_data=False).graph.input
]

calibration_dataset = nncf.Dataset(
    calibration_examples,
    lambda item: build_model_inputs(tokenizer, input_names, item),
)

ignored_scope = None
if payload["ignored_nodes"]:
    ignored_scope = nncf.IgnoredScope(names=payload["ignored_nodes"])

preset = (
    nncf.QuantizationPreset.MIXED
    if payload["preset"] == "mixed"
    else nncf.QuantizationPreset.PERFORMANCE
)

validation_dataset = None
validation_fn = None
reference_labels = {}

if payload["mode"] == "accuracy-control":
    validation_dataset = nncf.Dataset(validation_examples)
    if payload["metric"] == "hf_agreement":
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            payload["hf_source"],
            local_files_only=pathlib.Path(payload["hf_source"]).exists(),
        )
        hf_model.eval()
        import torch

        for item in validation_examples:
            encoded = tokenizer(
                item["premise"],
                item["hypothesis"],
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = hf_model(**encoded).logits[0].tolist()
            reference_labels[item["id"]] = predicted_label([float(value) for value in logits])

    def validation_fn(model, validation_dataset):
        session = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        session_input_names = [meta.name for meta in session.get_inputs()]
        total = 0
        hits = 0
        for item in iter_validation_rows(validation_dataset):
            feed = build_model_inputs(tokenizer, session_input_names, item)
            outputs = session.run(None, feed)
            logits = [float(value) for value in outputs[0][0].tolist()]
            label = predicted_label(logits)
            if payload["metric"] == "gold_accuracy":
                reference_label = item["gold_label"]
            else:
                reference_label = reference_labels[item["id"]]
            if reference_label:
                hits += int(label == reference_label)
                total += 1
        if total == 0:
            raise RuntimeError("Validation metric requires at least one labeled validation example")
        return hits / total, None

common_kwargs = {
    "preset": preset,
    "subset_size": payload["subset_size"],
    "fast_bias_correction": payload["fast_bias_correction"],
    "model_type": nncf.ModelType.TRANSFORMER,
    "ignored_scope": ignored_scope,
}

retry_reason = ""
smooth_quant_disabled = payload["disable_smooth_quant"]


def make_advanced_parameters(disable_smooth_quant):
    if not disable_smooth_quant:
        return None
    return AdvancedQuantizationParameters(smooth_quant_alpha=-1.0)


def run_quantization(disable_smooth_quant):
    model = onnx.load(resolved_input_model, load_external_data=False)
    advanced_parameters = make_advanced_parameters(disable_smooth_quant)
    if payload["mode"] == "ptq":
        return nncf.quantize(
            model,
            calibration_dataset,
            advanced_parameters=advanced_parameters,
            **common_kwargs,
        )
    return nncf.quantize_with_accuracy_control(
        model,
        calibration_dataset,
        validation_dataset,
        validation_fn,
        max_drop=payload["max_drop"],
        drop_type=nncf.DropType.ABSOLUTE,
        advanced_quantization_parameters=advanced_parameters,
        **common_kwargs,
    )

try:
    quantized_model = run_quantization(payload["disable_smooth_quant"])
except Exception as exc:
    if payload["disable_smooth_quant"]:
        raise
    exc_text = str(exc)
    if (
        "Smooth Quant" not in exc_text
        and "smooth_quant" not in exc_text
        and "weight_value" not in exc_text
        and "'NoneType' object has no attribute 'shape'" not in exc_text
    ):
        raise
    retry_reason = exc_text
    smooth_quant_disabled = True
    print(
        "warning: retrying NNCF quantization with SmoothQuant disabled after backend failure",
        file=sys.stderr,
    )
    quantized_model = run_quantization(True)

onnx.save_model(quantized_model, payload["output"])
if preprocess_dir is not None:
    preprocess_dir.cleanup()

result = {
    "input": payload["input"],
    "output": payload["output"],
    "mode": payload["mode"],
    "metric": payload["metric"],
    "preset": payload["preset"],
    "subset_size": payload["subset_size"],
    "ignored_scope_family": payload["ignored_scope_family"],
    "ignored_nodes": payload["ignored_nodes"],
    "max_drop": payload["max_drop"],
    "fast_bias_correction": payload["fast_bias_correction"],
    "preprocess": payload["preprocess"],
    "skip_preprocess_optimization": payload["skip_preprocess_optimization"],
    "smooth_quant_disabled": smooth_quant_disabled,
    "retry_reason": retry_reason,
    "calibration_examples": len(calibration_examples),
    "validation_examples": len(validation_examples),
}
print(json.dumps(result))
"""

    payload = {
        "input": str(input_model),
        "output": str(output_path),
        "tokenizer_source": args.tokenizer_source,
        "hf_source": args.hf_source,
        "mode": args.mode,
        "metric": args.metric,
        "preset": args.preset,
        "subset_size": args.subset_size,
        "ignored_scope_family": args.ignored_scope_family,
        "ignored_nodes": ignored_nodes_for_family(input_model, args.ignored_scope_family),
        "max_drop": args.max_drop,
        "preprocess": args.preprocess,
        "skip_preprocess_optimization": args.skip_preprocess_optimization,
        "disable_smooth_quant": args.disable_smooth_quant,
        "fast_bias_correction": args.fast_bias_correction,
        "calibration_tsvs": [str(path) for path in calibration_paths],
        "validation_tsvs": [str(path) for path in validation_paths],
    }

    with tempfile.TemporaryDirectory(prefix="nli-nncf-quantize-") as tmp_dir:
        tmp_dir_path = pathlib.Path(tmp_dir)
        payload_path = tmp_dir_path / "PAYLOAD.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        helper_code = helper.replace("PAYLOAD.json", str(payload_path))
        result = subprocess.run(
            [python_executable, "-c", helper_code],
            text=True,
            capture_output=True,
            check=True,
        )

    report = parse_last_json_line(result.stdout)
    print(f"input_model: {report['input']}")
    print(f"generated: {report['output']}")
    print(
        f"  config: mode={report['mode']} metric={report['metric']} "
        f"preset={report['preset']} subset_size={report['subset_size']} "
        f"ignored_scope_family={report['ignored_scope_family']} "
        f"ignored_nodes={report['ignored_nodes']} "
        f"max_drop={report['max_drop']} "
        f"preprocess={report['preprocess']} "
        f"skip_preprocess_optimization={report['skip_preprocess_optimization']} "
        f"fast_bias_correction={report['fast_bias_correction']} "
        f"smooth_quant_disabled={report['smooth_quant_disabled']}"
    )
    print(
        f"  datasets: calibration_examples={report['calibration_examples']} "
        f"validation_examples={report['validation_examples']}"
    )
    if report["retry_reason"]:
        print(f"  retry_reason: {report['retry_reason']}")
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
