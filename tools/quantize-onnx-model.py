#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import venv


DEFAULT_INPUT = "models/mdeberta/onnx/model.onnx"
DEFAULT_OUTPUT_DIR = "models/mdeberta/onnx/candidates"
DEFAULT_VENV = ".venv" if os.path.exists(".venv") else os.path.join(
    tempfile.gettempdir(), "nli-onnx-tools-venv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dynamic-quantized ONNX candidates from the known-good float export. "
            "Use the resulting files with tools/compare-hf-onnx-logits.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Presets:
              mdeberta-study  Generate a small candidate sweep for transformer/NLI debugging.
              single          Generate exactly one output using the explicit flags below.

            Examples:
              .venv/bin/python tools/quantize-onnx-model.py
              .venv/bin/python tools/quantize-onnx-model.py --preset single \\
                --output models/mdeberta/onnx/candidates/matmul_pc.onnx \\
                --per-channel --op-type MatMul
            """
        ),
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Float ONNX model used as the source (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--preset",
        choices=["mdeberta-study", "single"],
        default="mdeberta-study",
        help="Quantization preset to generate (default: mdeberta-study)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory used by sweep presets (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path for --preset single",
    )
    parser.add_argument(
        "--op-type",
        dest="op_types",
        action="append",
        default=[],
        help="Operator type to quantize in --preset single. Repeat to add more.",
    )
    parser.add_argument(
        "--nodes-to-exclude",
        default="",
        help="Comma-separated node names to exclude from quantization in --preset single",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel weight quantization in --preset single",
    )
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="Enable reduced-range quantization in --preset single",
    )
    parser.add_argument(
        "--weight-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Weight quantization type for --preset single (default: qint8)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs that already exist",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Create a virtualenv and install onnx+onnxruntime if needed",
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
    needed = ("onnx", "onnxruntime")
    if all(module_available(module) for module in needed):
        return sys.executable

    if not args.install_deps:
        raise RuntimeError(
            "Python packages 'onnx' and 'onnxruntime' are required. Install them in the "
            "active environment or re-run with --install-deps."
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
        ],
        check=True,
    )
    return str(python_path)


def candidate_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.preset == "single":
        if not args.output:
            raise RuntimeError("--output is required with --preset single")
        return [
            {
                "name": pathlib.Path(args.output).stem,
                "output": args.output,
                "weight_type": args.weight_type,
                "per_channel": args.per_channel,
                "reduce_range": args.reduce_range,
                "op_types": args.op_types,
                "nodes_to_exclude": [
                    node.strip() for node in args.nodes_to_exclude.split(",") if node.strip()
                ],
            }
        ]

    output_dir = pathlib.Path(args.output_dir)
    return [
        {
            "name": "dynamic_qint8_default",
            "output": str(output_dir / "dynamic_qint8_default.onnx"),
            "weight_type": "qint8",
            "per_channel": False,
            "reduce_range": False,
            "op_types": [],
            "nodes_to_exclude": [],
        },
        {
            "name": "dynamic_qint8_per_channel",
            "output": str(output_dir / "dynamic_qint8_per_channel.onnx"),
            "weight_type": "qint8",
            "per_channel": True,
            "reduce_range": False,
            "op_types": [],
            "nodes_to_exclude": [],
        },
        {
            "name": "dynamic_qint8_matmul",
            "output": str(output_dir / "dynamic_qint8_matmul.onnx"),
            "weight_type": "qint8",
            "per_channel": False,
            "reduce_range": False,
            "op_types": ["MatMul"],
            "nodes_to_exclude": [],
        },
        {
            "name": "dynamic_qint8_matmul_per_channel",
            "output": str(output_dir / "dynamic_qint8_matmul_per_channel.onnx"),
            "weight_type": "qint8",
            "per_channel": True,
            "reduce_range": False,
            "op_types": ["MatMul"],
            "nodes_to_exclude": [],
        },
        {
            "name": "dynamic_quint8_matmul",
            "output": str(output_dir / "dynamic_quint8_matmul.onnx"),
            "weight_type": "quint8",
            "per_channel": False,
            "reduce_range": False,
            "op_types": ["MatMul"],
            "nodes_to_exclude": [],
        },
    ]


def main() -> int:
    args = parse_args()
    python_executable = ensure_python(args)
    input_model = pathlib.Path(args.input)
    if not input_model.exists():
        raise RuntimeError(f"Input model not found: {input_model}")

    specs = candidate_specs(args)
    for spec in specs:
        output_path = pathlib.Path(spec["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.force:
            raise RuntimeError(
                f"Output already exists: {output_path}. Re-run with --force or choose a new path."
            )

    helper = r"""
import json
import pathlib
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic

input_model = sys.argv[1]
specs = json.loads(sys.argv[2])


def quant_type(name):
    if name == "qint8":
        return QuantType.QInt8
    if name == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"unsupported weight type: {name}")


for spec in specs:
    output_path = pathlib.Path(spec["output"])
    quantize_dynamic(
        input_model,
        str(output_path),
        per_channel=spec["per_channel"],
        reduce_range=spec["reduce_range"],
        weight_type=quant_type(spec["weight_type"]),
        op_types_to_quantize=spec["op_types"] or None,
        nodes_to_exclude=spec["nodes_to_exclude"] or None,
        extra_options={"MatMulConstBOnly": True},
    )
    print(json.dumps(spec))
"""

    result = subprocess.run(
        [
            python_executable,
            "-c",
            helper,
            str(input_model),
            json.dumps(specs),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    print(f"input_model: {input_model}")
    for line in result.stdout.splitlines():
        spec = json.loads(line)
        print(f"generated: {spec['output']}")
        print(
            f"  config: weight_type={spec['weight_type']} "
            f"per_channel={spec['per_channel']} reduce_range={spec['reduce_range']} "
            f"op_types={spec['op_types'] or 'default'} "
            f"nodes_to_exclude={spec['nodes_to_exclude'] or []}"
        )
        print(
            "  compare: "
            f".venv/bin/python tools/compare-hf-onnx-logits.py --quantized-model {spec['output']}"
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
