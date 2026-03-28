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


DEFAULT_PREMISE = (
    "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
)
DEFAULT_HYPOTHESIS = "Emmanuel Macron is the President of France"
DEFAULT_CPP_BINARY = "builddir/nli"
DEFAULT_CPP_MODEL = "models/mdeberta/onnx/model_quantized.onnx"
DEFAULT_HF_SOURCE = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DEFAULT_VENV = os.path.join(tempfile.gettempdir(), "nli-hf-tokenizer-venv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the local C++ DeBERTa encoding path against a Hugging Face "
            "tokenizer reference."
        )
    )
    parser.add_argument("--premise", default=DEFAULT_PREMISE, help="Premise text")
    parser.add_argument(
        "--hypothesis", default=DEFAULT_HYPOTHESIS, help="Hypothesis text"
    )
    parser.add_argument(
        "--cpp-binary",
        default=DEFAULT_CPP_BINARY,
        help=f"Path to the rebuilt C++ CLI (default: {DEFAULT_CPP_BINARY})",
    )
    parser.add_argument(
        "--cpp-model",
        default=DEFAULT_CPP_MODEL,
        help=(
            "Model path passed to the C++ CLI while dumping encodings "
            f"(default: {DEFAULT_CPP_MODEL})"
        ),
    )
    parser.add_argument(
        "--hf-source",
        default=DEFAULT_HF_SOURCE,
        help=(
            "Tokenizer source for AutoTokenizer.from_pretrained(). "
            "Can be a model id or a local directory."
        ),
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Create a temporary venv and install transformers+sentencepiece if needed",
    )
    parser.add_argument(
        "--venv",
        default=DEFAULT_VENV,
        help=f"Virtualenv path used with --install-deps (default: {DEFAULT_VENV})",
    )
    parser.add_argument(
        "--show-arrays",
        action="store_true",
        help="Print the full encoded arrays from both implementations",
    )
    return parser.parse_args()


def parse_cpp_dump(output: str) -> dict[str, object]:
    scalar_patterns = {
        "normalized_premise": re.compile(r"^normalized_premise:\s?(.*)$", re.MULTILINE),
        "normalized_hypothesis": re.compile(
            r"^normalized_hypothesis:\s?(.*)$", re.MULTILINE
        ),
    }

    vector_patterns = {
        "input_ids": re.compile(r"^input_ids:(.*)$", re.MULTILINE),
        "attention_mask": re.compile(r"^attention_mask:(.*)$", re.MULTILINE),
        "token_type_ids": re.compile(r"^token_type_ids:(.*)$", re.MULTILINE),
    }

    parsed: dict[str, object] = {}
    for key, pattern in scalar_patterns.items():
        match = pattern.search(output)
        if not match:
            raise RuntimeError(f"Failed to parse {key} from C++ output")
        parsed[key] = match.group(1)

    for key, pattern in vector_patterns.items():
        match = pattern.search(output)
        if not match:
            raise RuntimeError(f"Failed to parse {key} from C++ output")
        text = match.group(1).strip()
        parsed[key] = [] if not text else [int(value) for value in text.split()]

    return parsed


def run_cpp_dump(args: argparse.Namespace) -> dict[str, object]:
    cpp_binary = pathlib.Path(args.cpp_binary)
    if not cpp_binary.exists():
        raise RuntimeError(
            f"C++ binary not found at {cpp_binary}. Build it first with tools/build.sh --target nli"
        )

    command = [
        str(cpp_binary),
        "-b",
        "cpu",
        "--dump-encoding",
        "--model",
        args.cpp_model,
        "--premise",
        args.premise,
        "--hypothesis",
        args.hypothesis,
    ]

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=True,
    )
    return parse_cpp_dump(result.stdout)


def module_available(module_name: str) -> bool:
    code = (
        "import importlib.util, sys;"
        f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    )
    return subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def ensure_reference_python(args: argparse.Namespace) -> str:
    if module_available("transformers") and module_available("sentencepiece"):
        return sys.executable

    if not args.install_deps:
        raise RuntimeError(
            "Python packages 'transformers' and 'sentencepiece' are not installed. "
            "Re-run with --install-deps or install them yourself."
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
        "transformers",
        "sentencepiece",
    ]
    subprocess.run(install_command, check=True)
    return str(python_path)


def run_hf_reference(args: argparse.Namespace, python_executable: str) -> dict[str, object]:
    source_path = pathlib.Path(args.hf_source)
    local_files_only = source_path.exists()

    helper = r"""
import json
import sys
from transformers import AutoTokenizer

source, premise, hypothesis, local_files_only = sys.argv[1:5]
tokenizer = AutoTokenizer.from_pretrained(
    source,
    local_files_only=(local_files_only == "1"),
    use_fast=True,
)
encoded = tokenizer(premise, hypothesis, truncation=True, return_tensors=None)
payload = {
    "tokenizer_class": tokenizer.__class__.__name__,
    "input_ids": encoded["input_ids"],
    "attention_mask": encoded.get("attention_mask", []),
    "token_type_ids": encoded.get("token_type_ids", []),
    "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
}
print(json.dumps(payload))
"""

    result = subprocess.run(
        [
            python_executable,
            "-c",
            helper,
            args.hf_source,
            args.premise,
            args.hypothesis,
            "1" if local_files_only else "0",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def first_difference(left: list[int], right: list[int]) -> tuple[int, int | None, int | None] | None:
    max_common = min(len(left), len(right))
    for index in range(max_common):
        if left[index] != right[index]:
            return (index, left[index], right[index])
    if len(left) != len(right):
        left_value = left[max_common] if max_common < len(left) else None
        right_value = right[max_common] if max_common < len(right) else None
        return (max_common, left_value, right_value)
    return None


def print_sequence_comparison(name: str, cpp_values: list[int], hf_values: list[int], show_arrays: bool) -> None:
    difference = first_difference(cpp_values, hf_values)
    if difference is None:
        print(f"{name}: MATCH (length={len(cpp_values)})")
    else:
        index, cpp_value, hf_value = difference
        print(
            f"{name}: DIFFER at index {index} "
            f"(cpp={cpp_value}, hf={hf_value}, cpp_len={len(cpp_values)}, hf_len={len(hf_values)})"
        )

    if show_arrays:
        print(f"  cpp_{name}: {cpp_values}")
        print(f"  hf_{name}:  {hf_values}")


def shell_quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> int:
    args = parse_args()

    cpp_command = [
        args.cpp_binary,
        "-b",
        "cpu",
        "--dump-encoding",
        "--model",
        args.cpp_model,
        "--premise",
        args.premise,
        "--hypothesis",
        args.hypothesis,
    ]
    print(f"cpp_command: {shell_quote(cpp_command)}")
    cpp_dump = run_cpp_dump(args)

    python_executable = ensure_reference_python(args)
    hf_reference = run_hf_reference(args, python_executable)

    print(f"hf_source: {args.hf_source}")
    print(f"hf_tokenizer_class: {hf_reference['tokenizer_class']}")
    print(f"normalized_premise: {cpp_dump['normalized_premise']}")
    print(f"normalized_hypothesis: {cpp_dump['normalized_hypothesis']}")

    print_sequence_comparison(
        "input_ids",
        cpp_dump["input_ids"],
        hf_reference["input_ids"],
        args.show_arrays,
    )
    print_sequence_comparison(
        "attention_mask",
        cpp_dump["attention_mask"],
        hf_reference["attention_mask"],
        args.show_arrays,
    )
    print_sequence_comparison(
        "token_type_ids",
        cpp_dump["token_type_ids"],
        hf_reference["token_type_ids"],
        args.show_arrays,
    )

    if args.show_arrays:
        print(f"hf_tokens: {hf_reference['tokens']}")

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
