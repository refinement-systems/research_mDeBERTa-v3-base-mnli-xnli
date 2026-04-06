#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing artifact into a study scratchpad and emit a small JSON report "
            "that can preserve upstream generation metadata such as SmoothQuant retry status."
        )
    )
    parser.add_argument("--src", required=True, help="Source artifact path.")
    parser.add_argument("--dest", required=True, help="Destination artifact path.")
    parser.add_argument(
        "--source-stdout-log",
        default="",
        help="Optional stdout log from the original generator run.",
    )
    return parser.parse_args()


def parse_generation_metadata(stdout_log_path: pathlib.Path) -> tuple[bool | None, str]:
    if not stdout_log_path.is_file():
        return None, ""

    text = stdout_log_path.read_text(encoding="utf-8", errors="replace")
    smooth_quant_disabled: bool | None = None
    retry_reason = ""

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if "smooth_quant_disabled" in payload:
                value = payload["smooth_quant_disabled"]
                smooth_quant_disabled = None if value is None else bool(value)
            if "retry_reason" in payload and payload["retry_reason"] is not None:
                retry_reason = str(payload["retry_reason"])
            return smooth_quant_disabled, retry_reason

    match = re.search(r"smooth_quant_disabled=(True|False)", text)
    if match:
        smooth_quant_disabled = match.group(1) == "True"

    for line in text.splitlines():
        if line.startswith("  retry_reason: "):
            retry_reason = line.split(": ", 1)[1].strip()
            break

    return smooth_quant_disabled, retry_reason


def main() -> int:
    args = parse_args()

    source_path = pathlib.Path(args.src).resolve()
    dest_path = pathlib.Path(args.dest).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source artifact not found: {source_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dest_path)

    smooth_quant_disabled = None
    retry_reason = ""
    if args.source_stdout_log:
        smooth_quant_disabled, retry_reason = parse_generation_metadata(
            pathlib.Path(args.source_stdout_log).resolve()
        )

    payload = {
        "source": str(source_path),
        "dest": str(dest_path),
        "size_bytes": dest_path.stat().st_size,
        "smooth_quant_disabled": smooth_quant_disabled,
        "retry_reason": retry_reason,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
