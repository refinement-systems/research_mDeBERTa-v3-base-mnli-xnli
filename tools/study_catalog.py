#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = (
    REPO_ROOT / "research/attempt2_course-correction/study_quantization_catalog.json"
)


REQUIRED_KEYS = {
    "name",
    "generator_program",
    "generator_args_json",
    "source_artifact_name",
    "output_relpath",
    "calibration_role",
    "validation_role",
    "allowed_backends",
    "notes",
}


def load_catalog(path: pathlib.Path | None = None) -> list[dict[str, object]]:
    catalog_path = pathlib.Path(path) if path else DEFAULT_CATALOG_PATH
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Catalog must be a JSON array: {catalog_path}")

    seen_names: set[str] = set()
    validated: list[dict[str, object]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(f"Catalog entry {index} must be a JSON object")

        missing = REQUIRED_KEYS - set(item.keys())
        if missing:
            raise RuntimeError(
                f"Catalog entry {index} is missing keys: {', '.join(sorted(missing))}"
            )

        name = item["name"]
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"Catalog entry {index} must define a non-empty string name")
        if name in seen_names:
            raise RuntimeError(f"Catalog entry names must be unique: {name}")
        seen_names.add(name)

        args_json = item["generator_args_json"]
        if not isinstance(args_json, list) or not all(
            isinstance(entry, str) for entry in args_json
        ):
            raise RuntimeError(f"Catalog entry {name} must use a string array for generator_args_json")

        allowed_backends = item["allowed_backends"]
        if not isinstance(allowed_backends, list) or not all(
            isinstance(entry, str) for entry in allowed_backends
        ):
            raise RuntimeError(f"Catalog entry {name} must use a string array for allowed_backends")

        validated.append(item)

    return validated

