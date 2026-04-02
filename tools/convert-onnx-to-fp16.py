#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import venv


DEFAULT_VENV = ".venv" if os.path.exists(".venv") else os.path.join(
    tempfile.gettempdir(), "nli-onnx-tools-venv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an ONNX model to float16 while preserving the current text-model "
            "interface by default. This is intended for the CoreML attempt, where the "
            "repo still expects integer token inputs and float32 logits."
        )
    )
    parser.add_argument(
        "--src",
        "--input",
        dest="src",
        required=True,
        help="Source ONNX model path.",
    )
    parser.add_argument(
        "--dest",
        "--output",
        dest="dest",
        required=True,
        help="Destination ONNX model path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Create a virtualenv and install onnx + onnxconverter-common if needed.",
    )
    parser.add_argument(
        "--venv",
        default=DEFAULT_VENV,
        help=f"Virtualenv path used with --install-deps (default: {DEFAULT_VENV})",
    )
    parser.add_argument(
        "--keep-io-types",
        dest="keep_io_types",
        action="store_true",
        default=True,
        help="Preserve model input and output tensor types (default: on).",
    )
    parser.add_argument(
        "--no-keep-io-types",
        dest="keep_io_types",
        action="store_false",
        help="Allow float16 conversion to change model input and output tensor types.",
    )
    return parser.parse_args()


def module_available(module_name: str, python_executable: str | None = None) -> bool:
    code = (
        "import importlib.util, sys;"
        f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    )
    executable = python_executable or sys.executable
    return subprocess.run([executable, "-c", code], check=False).returncode == 0


def ensure_python(args: argparse.Namespace) -> str:
    needed = ("onnx", "onnxruntime")
    if all(module_available(module) for module in needed):
        return sys.executable

    if not args.install_deps:
        raise RuntimeError(
            "Converting ONNX to fp16 requires Python packages 'onnx' and "
            "'onnxruntime'. Install them in the active environment or re-run "
            "with --install-deps."
        )

    venv_dir = pathlib.Path(args.venv)
    python_path = venv_dir / "bin" / "python3"
    if not python_path.exists():
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)

    if not all(module_available(module, str(python_path)) for module in needed):
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
            ],
            check=True,
        )
    return str(python_path)


def validate_generated_model(path: pathlib.Path) -> None:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    ort.InferenceSession(
        str(path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def main() -> int:
    args = parse_args()
    python_executable = ensure_python(args)
    if python_executable != sys.executable:
        return subprocess.run(
            [python_executable, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        ).returncode

    src_path = pathlib.Path(args.src).resolve()
    dest_path = pathlib.Path(args.dest).resolve()

    if not src_path.is_file():
        raise FileNotFoundError(f"Source ONNX model not found: {src_path}")
    if dest_path.exists() and not args.force:
        try:
            validate_generated_model(dest_path)
            print(f"exists: {dest_path}")
            return 0
        except Exception as exc:
            print(
                f"invalid existing fp16 artifact, regenerating: {dest_path} ({exc})",
                file=sys.stderr,
            )

    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    converted = convert_float_to_float16(
        str(src_path),
        keep_io_types=args.keep_io_types,
    )
    onnx.save_model(converted, str(dest_path), save_as_external_data=False)
    validate_generated_model(dest_path)

    print(f"src: {src_path}")
    print(f"dest: {dest_path}")
    print(f"keep_io_types: {args.keep_io_types}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
