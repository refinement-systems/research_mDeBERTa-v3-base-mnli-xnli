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


DEFAULT_INPUT = "models/mdeberta/onnx/model.onnx"
DEFAULT_OUTPUT = "models/mdeberta/onnx/candidates/static_qdq_qint8_matmul.onnx"
DEFAULT_TOKENIZER_SOURCE = "models/mdeberta"
DEFAULT_VENV = ".venv" if os.path.exists(".venv") else os.path.join(
    tempfile.gettempdir(), "nli-onnx-tools-venv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a static-quantized ONNX candidate from the float export using "
            "TSV benchmark slices as calibration data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              .venv/bin/python tools/quantize-onnx-static.py \\
                --calibration-tsv benchmarks/nli/mnli-validation_matched-100-per-label.tsv \\
                --calibration-tsv benchmarks/nli/xnli-en-test-50-per-label.tsv

              .venv/bin/python tools/quantize-onnx-static.py \\
                --input-dir benchmarks/nli --pattern 'xnli-*.tsv' \\
                --max-examples-per-source 8 --op-type MatMul --force
            """
        ),
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Float ONNX model used as the source (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path for the static-quantized model (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--tokenizer-source",
        default=DEFAULT_TOKENIZER_SOURCE,
        help=f"Local tokenizer/model directory used for calibration encoding (default: {DEFAULT_TOKENIZER_SOURCE})",
    )
    parser.add_argument(
        "--calibration-tsv",
        dest="calibration_tsvs",
        action="append",
        default=[],
        help="Calibration TSV with premise/hypothesis columns. Repeat to add more.",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Optional directory containing calibration TSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.tsv",
        help="Glob pattern used with --input-dir (default: *.tsv)",
    )
    parser.add_argument(
        "--max-examples-per-source",
        type=int,
        default=16,
        help="Maximum calibration examples to load from each TSV source (default: 16)",
    )
    parser.add_argument(
        "--max-total-examples",
        type=int,
        default=0,
        help="Optional cap across all calibration examples after per-source limits (default: unlimited)",
    )
    parser.add_argument(
        "--op-type",
        dest="op_types",
        action="append",
        default=[],
        help="Operator type to quantize. Repeat to add more. Defaults to MatMul.",
    )
    parser.add_argument(
        "--quant-format",
        choices=["qdq", "qoperator"],
        default="qdq",
        help="Static quantization graph format (default: qdq)",
    )
    parser.add_argument(
        "--activation-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Activation quantization type (default: qint8)",
    )
    parser.add_argument(
        "--weight-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Weight quantization type (default: qint8)",
    )
    parser.add_argument(
        "--calibrate-method",
        choices=["minmax", "entropy", "percentile"],
        default="minmax",
        help="Calibration method used by quantize_static (default: minmax)",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel weight quantization",
    )
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="Enable reduced-range quantization",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run ORT quant_pre_process on the float model before static quantization",
    )
    parser.add_argument(
        "--skip-preprocess-optimization",
        action="store_true",
        help="Disable graph optimization during quant_pre_process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
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

    input_model = pathlib.Path(args.input)
    if not input_model.exists():
        raise RuntimeError(f"Input model not found: {input_model}")

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
import tempfile

import numpy
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoTokenizer


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


def read_examples(tsv_paths, input_dir, pattern, max_examples_per_source, max_total_examples):
    examples = []
    source_counts = {}
    discovered_paths = discover_tsv_paths(tsv_paths, input_dir, pattern)
    if not discovered_paths:
        raise RuntimeError("Static quantization requires calibration TSV input")

    for tsv_path in discovered_paths:
        loaded_for_source = 0
        with open(tsv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "premise" not in reader.fieldnames or "hypothesis" not in reader.fieldnames:
                raise RuntimeError(f"TSV must include premise and hypothesis columns: {tsv_path}")
            for row in reader:
                if max_examples_per_source > 0 and loaded_for_source >= max_examples_per_source:
                    break
                if max_total_examples > 0 and len(examples) >= max_total_examples:
                    break
                examples.append((row["premise"], row["hypothesis"]))
                loaded_for_source += 1
        source_counts[str(tsv_path)] = loaded_for_source
        if max_total_examples > 0 and len(examples) >= max_total_examples:
            break

    if not examples:
        raise RuntimeError("No calibration examples were loaded")
    return examples, source_counts


def model_input_names(model_path):
    import onnx
    model = onnx.load(model_path, load_external_data=False)
    return [value_info.name for value_info in model.graph.input]


def build_feeds(tokenizer_source, model_inputs, examples):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=pathlib.Path(tokenizer_source).exists(),
        use_fast=True,
    )
    feeds = []
    for premise, hypothesis in examples:
        encoded = tokenizer(
            premise,
            hypothesis,
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


def quant_type(name):
    if name == "qint8":
        return QuantType.QInt8
    if name == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"unsupported quant type: {name}")


def quant_format(name):
    if name == "qdq":
        return QuantFormat.QDQ
    if name == "qoperator":
        return QuantFormat.QOperator
    raise ValueError(f"unsupported quant format: {name}")


def calibrate_method(name):
    if name == "minmax":
        return CalibrationMethod.MinMax
    if name == "entropy":
        return CalibrationMethod.Entropy
    if name == "percentile":
        return CalibrationMethod.Percentile
    raise ValueError(f"unsupported calibrate method: {name}")


payload = json.loads(pathlib.Path("PAYLOAD.json").read_text(encoding="utf-8"))
examples, source_counts = read_examples(
    payload["calibration_tsvs"],
    payload["input_dir"],
    payload["pattern"],
    payload["max_examples_per_source"],
    payload["max_total_examples"],
)

with tempfile.TemporaryDirectory(prefix="nli-static-quantize-") as tmp_dir:
    tmp_dir = pathlib.Path(tmp_dir)
    quant_input = pathlib.Path(payload["input"])
    if payload["preprocess"]:
        preprocessed_input = tmp_dir / "preprocessed.onnx"
        quant_pre_process(
            str(quant_input),
            str(preprocessed_input),
            skip_optimization=payload["skip_preprocess_optimization"],
            save_as_external_data=False,
        )
        quant_input = preprocessed_input

    feeds = build_feeds(
        payload["tokenizer_source"],
        model_input_names(str(quant_input)),
        examples,
    )
    reader = FeedListDataReader(feeds)
    quantize_static(
        str(quant_input),
        payload["output"],
        reader,
        quant_format=quant_format(payload["quant_format"]),
        op_types_to_quantize=payload["op_types"] or None,
        per_channel=payload["per_channel"],
        reduce_range=payload["reduce_range"],
        activation_type=quant_type(payload["activation_type"]),
        weight_type=quant_type(payload["weight_type"]),
        use_external_data_format=False,
        calibrate_method=calibrate_method(payload["calibrate_method"]),
        extra_options={"MatMulConstBOnly": True},
    )

result = {
    "input": payload["input"],
    "output": payload["output"],
    "tokenizer_source": payload["tokenizer_source"],
    "calibration_example_count": len(examples),
    "source_counts": source_counts,
    "op_types": payload["op_types"] or ["MatMul"],
    "quant_format": payload["quant_format"],
    "activation_type": payload["activation_type"],
    "weight_type": payload["weight_type"],
    "calibrate_method": payload["calibrate_method"],
    "per_channel": payload["per_channel"],
    "reduce_range": payload["reduce_range"],
    "preprocess": payload["preprocess"],
    "skip_preprocess_optimization": payload["skip_preprocess_optimization"],
}
print(json.dumps(result))
"""

    payload = {
        "input": str(input_model),
        "output": str(output_path),
        "tokenizer_source": args.tokenizer_source,
        "calibration_tsvs": args.calibration_tsvs,
        "input_dir": args.input_dir,
        "pattern": args.pattern,
        "max_examples_per_source": args.max_examples_per_source,
        "max_total_examples": args.max_total_examples,
        "op_types": args.op_types or ["MatMul"],
        "quant_format": args.quant_format,
        "activation_type": args.activation_type,
        "weight_type": args.weight_type,
        "calibrate_method": args.calibrate_method,
        "per_channel": args.per_channel,
        "reduce_range": args.reduce_range,
        "preprocess": args.preprocess,
        "skip_preprocess_optimization": args.skip_preprocess_optimization,
    }

    with tempfile.TemporaryDirectory(prefix="nli-static-quantize-payload-") as tmp_dir:
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

    report = json.loads(result.stdout)
    print(f"input_model: {report['input']}")
    print(f"generated: {report['output']}")
    print(f"  calibration_example_count: {report['calibration_example_count']}")
    print(f"  source_counts: {report['source_counts']}")
    print(
        f"  config: op_types={report['op_types']} "
        f"quant_format={report['quant_format']} "
        f"activation_type={report['activation_type']} "
        f"weight_type={report['weight_type']} "
        f"calibrate_method={report['calibrate_method']} "
        f"per_channel={report['per_channel']} "
        f"reduce_range={report['reduce_range']} "
        f"preprocess={report['preprocess']} "
        f"skip_preprocess_optimization={report['skip_preprocess_optimization']}"
    )
    print(
        "  compare: "
        f".venv/bin/python tools/compare-hf-onnx-logits.py --quantized-model {report['output']}"
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
