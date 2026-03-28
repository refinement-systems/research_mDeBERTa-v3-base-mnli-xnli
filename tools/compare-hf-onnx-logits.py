#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import venv


LABELS = ["entailment", "neutral", "contradiction"]
DEFAULT_PREMISE = (
    "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
)
DEFAULT_HYPOTHESIS = "Emmanuel Macron is the President of France"
DEFAULT_CPP_BINARY = "builddir/nli"
DEFAULT_FLOAT_MODEL = "models/mdeberta/onnx/model.onnx"
DEFAULT_QUANTIZED_MODEL = "models/mdeberta/onnx/model_quantized.onnx"
DEFAULT_LOCAL_HF_SOURCE = "models/mdeberta"
DEFAULT_REMOTE_HF_SOURCE = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DEFAULT_VENV = ".venv" if os.path.exists(".venv") else os.path.join(
    tempfile.gettempdir(), "nli-hf-logits-venv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Hugging Face/PyTorch reference logits against the float and "
            "quantized ONNX exports used by the local C++ runtime."
        )
    )
    parser.add_argument("--premise", default=DEFAULT_PREMISE, help="Premise text")
    parser.add_argument("--hypothesis", default=DEFAULT_HYPOTHESIS, help="Hypothesis text")
    parser.add_argument(
        "--cpp-binary",
        default=DEFAULT_CPP_BINARY,
        help=f"Path to the rebuilt C++ CLI (default: {DEFAULT_CPP_BINARY})",
    )
    parser.add_argument(
        "--float-model",
        default=DEFAULT_FLOAT_MODEL,
        help=f"Path to the float ONNX export (default: {DEFAULT_FLOAT_MODEL})",
    )
    parser.add_argument(
        "--quantized-model",
        default=DEFAULT_QUANTIZED_MODEL,
        help=f"Path to the quantized ONNX export (default: {DEFAULT_QUANTIZED_MODEL})",
    )
    parser.add_argument(
        "--hf-source",
        default="",
        help=(
            "Hugging Face model source used for the PyTorch reference. "
            "Defaults to a local models/mdeberta directory when it contains "
            "reference weights, otherwise the remote model id."
        ),
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Create a virtualenv and install torch+transformers+sentencepiece if needed",
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


def ensure_reference_python(args: argparse.Namespace) -> str:
    needed = ("torch", "transformers", "sentencepiece")
    if all(module_available(module) for module in needed):
        return sys.executable

    if not args.install_deps:
        raise RuntimeError(
            "Python packages 'torch', 'transformers', and 'sentencepiece' are required. "
            "Install them in the active environment or re-run with --install-deps."
        )

    venv_dir = pathlib.Path(args.venv)
    python_path = venv_dir / "bin" / "python3"
    if not python_path.exists():
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)

    install_command = [
        str(python_path),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "torch",
        "transformers",
        "sentencepiece",
    ]
    subprocess.run(install_command, check=True)
    return str(python_path)


def has_local_reference_weights(path: pathlib.Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def resolve_hf_source(args: argparse.Namespace) -> str:
    if args.hf_source:
        return args.hf_source

    local_source = pathlib.Path(DEFAULT_LOCAL_HF_SOURCE)
    if has_local_reference_weights(local_source):
        return str(local_source)
    return DEFAULT_REMOTE_HF_SOURCE


def shell_quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_cpp_output(output: str) -> dict[str, object]:
    logits_match = re.search(
        r"^logits:\s+entailment=(?P<entailment>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+"
        r"neutral=(?P<neutral>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+"
        r"contradiction=(?P<contradiction>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$",
        output,
        re.MULTILINE | re.IGNORECASE,
    )
    if not logits_match:
        raise RuntimeError("Failed to parse logits from C++ output")

    score_matches = {}
    for label in LABELS:
        match = re.search(
            rf"^{label}:\s+(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$",
            output,
            re.MULTILINE | re.IGNORECASE,
        )
        if not match:
            raise RuntimeError(f"Failed to parse {label} score from C++ output")
        score_matches[label] = float(match.group(1))

    label_match = re.search(r"^predicted_label:\s+(\w+)$", output, re.MULTILINE)
    if not label_match:
        raise RuntimeError("Failed to parse predicted_label from C++ output")

    logits = [float(logits_match.group(label)) for label in LABELS]
    scores = [score_matches[label] for label in LABELS]
    return {
        "logits": logits,
        "scores": scores,
        "predicted_label": label_match.group(1),
    }


def run_cpp_model(
    cpp_binary: str,
    model_path: str,
    premise: str,
    hypothesis: str,
) -> dict[str, object]:
    command = [
        cpp_binary,
        "-b",
        "cpu",
        "--dump-logits",
        "--model",
        model_path,
        "--premise",
        premise,
        "--hypothesis",
        hypothesis,
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = parse_cpp_output(result.stdout)
    payload["command"] = command
    return payload


def run_hf_reference(
    python_executable: str,
    hf_source: str,
    premise: str,
    hypothesis: str,
) -> dict[str, object]:
    local_files_only = pathlib.Path(hf_source).exists()
    helper = r"""
import json
import math
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

labels = ["entailment", "neutral", "contradiction"]
source, premise, hypothesis, local_files_only = sys.argv[1:5]
tokenizer = AutoTokenizer.from_pretrained(
    source,
    local_files_only=(local_files_only == "1"),
    use_fast=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    source,
    local_files_only=(local_files_only == "1"),
)
model.eval()
encoded = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
with torch.no_grad():
    logits_tensor = model(**encoded).logits[0]
logits = [float(value) for value in logits_tensor.tolist()]
max_value = max(logits)
exps = [math.exp(value - max_value) for value in logits]
total = sum(exps)
scores = [value / total for value in exps]
best_index = max(range(len(logits)), key=lambda index: logits[index])
payload = {
    "logits": logits,
    "scores": scores,
    "predicted_label": labels[best_index],
    "tokenizer_class": tokenizer.__class__.__name__,
    "model_class": model.__class__.__name__,
}
print(json.dumps(payload))
"""

    result = subprocess.run(
        [
            python_executable,
            "-c",
            helper,
            hf_source,
            premise,
            hypothesis,
            "1" if local_files_only else "0",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def vector_delta(left: list[float], right: list[float]) -> list[float]:
    return [left_value - right_value for left_value, right_value in zip(left, right)]


def max_abs(values: list[float]) -> float:
    return max(abs(value) for value in values)


def format_values(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def print_model_result(name: str, payload: dict[str, object]) -> None:
    print(f"{name}:")
    print(f"  predicted_label: {payload['predicted_label']}")
    print(f"  logits: {format_values(payload['logits'])}")
    print(f"  scores: {format_values(payload['scores'])}")
    if "command" in payload:
        print(f"  command: {shell_quote(payload['command'])}")
    if "model_class" in payload:
        print(f"  model_class: {payload['model_class']}")
    if "tokenizer_class" in payload:
        print(f"  tokenizer_class: {payload['tokenizer_class']}")


def print_delta(name: str, left: dict[str, object], right: dict[str, object]) -> None:
    logit_delta = vector_delta(left["logits"], right["logits"])
    score_delta = vector_delta(left["scores"], right["scores"])
    print(f"{name}:")
    print(f"  logit_delta: {format_values(logit_delta)}")
    print(f"  max_abs_logit_delta: {max_abs(logit_delta):.6f}")
    print(f"  score_delta: {format_values(score_delta)}")
    print(f"  max_abs_score_delta: {max_abs(score_delta):.6f}")


def main() -> int:
    args = parse_args()
    python_executable = ensure_reference_python(args)
    hf_source = resolve_hf_source(args)

    cpp_binary = pathlib.Path(args.cpp_binary)
    if not cpp_binary.exists():
        raise RuntimeError(
            f"C++ binary not found at {cpp_binary}. Build it first with tools/build.sh --target nli"
        )

    hf_reference = run_hf_reference(
        python_executable,
        hf_source,
        args.premise,
        args.hypothesis,
    )
    onnx_float = run_cpp_model(
        args.cpp_binary,
        args.float_model,
        args.premise,
        args.hypothesis,
    )
    onnx_quantized = run_cpp_model(
        args.cpp_binary,
        args.quantized_model,
        args.premise,
        args.hypothesis,
    )

    print(f"premise: {args.premise}")
    print(f"hypothesis: {args.hypothesis}")
    print(f"label_order: {', '.join(LABELS)}")
    print(f"hf_source: {hf_source}")
    print_model_result("hf_reference", hf_reference)
    print_model_result("onnx_float", onnx_float)
    print_model_result("onnx_quantized", onnx_quantized)
    print_delta("float_minus_hf", onnx_float, hf_reference)
    print_delta("quantized_minus_hf", onnx_quantized, hf_reference)
    print_delta("quantized_minus_float", onnx_quantized, onnx_float)
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
