#!/usr/bin/env python3

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import venv


DEFAULT_MODELS = [
    "models/mdeberta/onnx/model.onnx",
    "models/mdeberta/onnx/model_quantized.onnx",
]
DEFAULT_VENV = ".venv" if os.path.exists(".venv") else os.path.join(
    tempfile.gettempdir(), "nli-onnx-tools-venv"
)
QUANTIZATION_OPS = {
    "DynamicQuantizeLinear",
    "QuantizeLinear",
    "DequantizeLinear",
    "MatMulInteger",
    "QLinearMatMul",
    "ConvInteger",
    "QLinearConv",
    "GatherBlockQuantized",
    "MatMulNBits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ONNX model metadata, operator usage, and quantization structure."
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="ONNX model paths to inspect",
    )
    parser.add_argument(
        "--top-ops",
        type=int,
        default=20,
        help="How many of the most common operators to print (default: 20)",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Create a virtualenv and install onnx if needed",
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
    if module_available("onnx"):
        return sys.executable

    if not args.install_deps:
        raise RuntimeError(
            "Python package 'onnx' is required. Install it in the active environment or "
            "re-run with --install-deps."
        )

    venv_dir = pathlib.Path(args.venv)
    python_path = venv_dir / "bin" / "python3"
    if not python_path.exists():
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)

    subprocess.run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip", "onnx"],
        check=True,
    )
    return str(python_path)


def main() -> int:
    args = parse_args()
    python_executable = ensure_python(args)
    helper = r"""
import collections
import json
import os
import sys

import onnx
from onnx import TensorProto

QUANTIZATION_OPS = {
    "DynamicQuantizeLinear",
    "QuantizeLinear",
    "DequantizeLinear",
    "MatMulInteger",
    "QLinearMatMul",
    "ConvInteger",
    "QLinearConv",
    "GatherBlockQuantized",
    "MatMulNBits",
}


def dims_for(value_info):
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def tensor_info(value_info):
    tensor_type = value_info.type.tensor_type
    elem_type = tensor_type.elem_type
    return {
        "name": value_info.name,
        "elem_type": TensorProto.DataType.Name(elem_type),
        "shape": dims_for(value_info),
    }


def quantization_style(op_counts):
    styles = []
    if any(op_counts.get(op, 0) for op in ("DynamicQuantizeLinear", "MatMulInteger", "ConvInteger")):
        styles.append("dynamic_integer")
    if any(op_counts.get(op, 0) for op in ("QuantizeLinear", "DequantizeLinear")):
        styles.append("qdq")
    if any(op_counts.get(op, 0) for op in ("QLinearMatMul", "QLinearConv")):
        styles.append("qoperator")
    if not styles:
        styles.append("float_or_unknown")
    return styles


for path in sys.argv[1:]:
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    op_counts = collections.Counter(node.op_type for node in graph.node)
    quant_counts = {op: op_counts[op] for op in sorted(QUANTIZATION_OPS) if op_counts.get(op, 0)}
    initializer_types = collections.Counter(
        TensorProto.DataType.Name(initializer.data_type)
        for initializer in graph.initializer
    )
    payload = {
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "domain": model.domain,
        "model_version": model.model_version,
        "ir_version": model.ir_version,
        "graph_name": graph.name,
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "sparse_initializer_count": len(graph.sparse_initializer),
        "value_info_count": len(graph.value_info),
        "opset_imports": {
            (opset.domain or "ai.onnx"): opset.version for opset in model.opset_import
        },
        "metadata_props": {item.key: item.value for item in model.metadata_props},
        "inputs": [tensor_info(value_info) for value_info in graph.input],
        "outputs": [tensor_info(value_info) for value_info in graph.output],
        "top_ops": op_counts.most_common(),
        "quantization_ops": quant_counts,
        "quantization_style": quantization_style(op_counts),
        "initializer_data_types": dict(initializer_types),
    }
    print(json.dumps(payload))
"""

    result = subprocess.run(
        [python_executable, "-c", helper, *args.models],
        text=True,
        capture_output=True,
        check=True,
    )

    for line in result.stdout.splitlines():
        payload = json.loads(line)
        print(f"model: {payload['path']}")
        print(f"  file_size_bytes: {payload['file_size_bytes']}")
        print(
            f"  producer: {payload['producer_name'] or '<none>'}"
            f" {payload['producer_version'] or ''}".rstrip()
        )
        print(
            f"  domain/model_version/ir_version: "
            f"{payload['domain'] or '<none>'}/{payload['model_version']}/{payload['ir_version']}"
        )
        print(f"  graph_name: {payload['graph_name'] or '<none>'}")
        print(
            f"  node_count: {payload['node_count']}  "
            f"initializer_count: {payload['initializer_count']}  "
            f"value_info_count: {payload['value_info_count']}"
        )
        print(f"  opset_imports: {payload['opset_imports']}")
        print(f"  quantization_style: {', '.join(payload['quantization_style'])}")
        print(f"  quantization_ops: {payload['quantization_ops'] or '{}'}")
        if payload["metadata_props"]:
            print(f"  metadata_props: {payload['metadata_props']}")
        print("  inputs:")
        for item in payload["inputs"]:
            print(
                f"    - {item['name']}: {item['elem_type']} shape={item['shape']}"
            )
        print("  outputs:")
        for item in payload["outputs"]:
            print(
                f"    - {item['name']}: {item['elem_type']} shape={item['shape']}"
            )
        print(f"  initializer_data_types: {payload['initializer_data_types']}")
        print("  top_ops:")
        for op_type, count in payload["top_ops"][: args.top_ops]:
            print(f"    - {op_type}: {count}")
        print()

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
