#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys


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


def main() -> int:
    args = parse_args()

    src_path = pathlib.Path(args.src).resolve()
    dest_path = pathlib.Path(args.dest).resolve()

    if not src_path.is_file():
        raise FileNotFoundError(f"Source ONNX model not found: {src_path}")
    if dest_path.exists() and not args.force:
        print(f"exists: {dest_path}")
        return 0

    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise RuntimeError(
            "Converting ONNX to fp16 requires Python packages 'onnx' and "
            "'onnxconverter-common'. Install them in the environment that runs this tool."
        ) from exc

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load_model(str(src_path), load_external_data=True)
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=args.keep_io_types,
    )
    onnx.save_model(converted, str(dest_path))

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
