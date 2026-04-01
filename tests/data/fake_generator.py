#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import sys


def read_option(argv: list[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def repeated_options(argv: list[str], name: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif value.startswith(name + "="):
            values.append(value.split("=", 1)[1])
    return values


def main() -> int:
    argv = sys.argv[1:]
    dest = read_option(argv, "--dest")
    if not dest:
        print("missing --dest", file=sys.stderr)
        return 2

    src = read_option(argv, "--src")
    capture = read_option(argv, "--capture")

    dest_path = pathlib.Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(
        "generated\n"
        f"src={src}\n",
        encoding="utf-8",
    )

    if capture:
        capture_path = pathlib.Path(capture)
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_path.write_text(
            json.dumps(
                {
                    "argv": argv,
                    "src": src,
                    "dest": dest,
                    "calibration_tsvs": repeated_options(argv, "--calibration-tsv"),
                    "validation_tsvs": repeated_options(argv, "--validation-tsv"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

