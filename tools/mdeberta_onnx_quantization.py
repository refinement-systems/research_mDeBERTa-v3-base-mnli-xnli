#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import re
from typing import Any


LAYER_PATTERN = re.compile(r"^/deberta/encoder/layer\.(\d+)/")
SUPPORTED_IGNORE_FAMILIES = ("none", "attention_only", "attention_proj_only")


def load_matmul_families(model_path: pathlib.Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(model_path, load_external_data=False)
    families: dict[str, Any] = {
        "all_quantizable": [],
        "attention_proj": [],
        "attention_output": [],
        "ffn_intermediate": [],
        "ffn_output": [],
        "ffn_all": [],
        "attention_all": [],
        "layer_all": {},
    }

    for node in model.graph.node:
        if node.op_type != "MatMul" or not node.name.startswith("/deberta/encoder/layer."):
            continue

        layer_match = LAYER_PATTERN.match(node.name)
        if not layer_match:
            continue
        layer_index = int(layer_match.group(1))

        family = None
        if any(
            token in node.name
            for token in (
                "/attention/self/query_proj/MatMul",
                "/attention/self/key_proj/MatMul",
                "/attention/self/value_proj/MatMul",
                "/attention/self/query_proj_1/MatMul",
                "/attention/self/key_proj_1/MatMul",
            )
        ):
            family = "attention_proj"
        elif "/attention/output/dense/MatMul" in node.name:
            family = "attention_output"
        elif "/intermediate/dense/MatMul" in node.name:
            family = "ffn_intermediate"
        elif "/output/dense/MatMul" in node.name:
            family = "ffn_output"

        if not family:
            continue

        families["all_quantizable"].append(node.name)
        families[family].append(node.name)
        families["layer_all"].setdefault(layer_index, []).append(node.name)

    families["attention_all"] = sorted(
        families["attention_proj"] + families["attention_output"]
    )
    families["ffn_all"] = sorted(families["ffn_intermediate"] + families["ffn_output"])
    families["all_quantizable"] = sorted(families["all_quantizable"])
    for key in ("attention_proj", "attention_output", "ffn_intermediate", "ffn_output"):
        families[key] = sorted(families[key])
    for layer_index, names in list(families["layer_all"].items()):
        families["layer_all"][layer_index] = sorted(names)

    return families


def layer_subset(names: list[str], start: int, end: int) -> list[str]:
    selected = []
    for name in names:
        layer_match = LAYER_PATTERN.match(name)
        if not layer_match:
            continue
        layer_index = int(layer_match.group(1))
        if start <= layer_index <= end:
            selected.append(name)
    return sorted(selected)


def merge_nodes(*node_lists: list[str]) -> list[str]:
    merged = set()
    for node_list in node_lists:
        merged.update(node_list)
    return sorted(merged)


def ignored_nodes_for_family(
    model_path: pathlib.Path,
    family_name: str,
) -> list[str]:
    if family_name not in SUPPORTED_IGNORE_FAMILIES:
        raise RuntimeError(
            f"Unsupported ignore family: {family_name}. "
            f"Choose one of: {', '.join(SUPPORTED_IGNORE_FAMILIES)}"
        )

    families = load_matmul_families(model_path)
    if family_name == "none":
        return []
    if family_name == "attention_only":
        return list(families["ffn_all"])
    if family_name == "attention_proj_only":
        return merge_nodes(
            families["ffn_all"],
            families["attention_output"],
        )
    raise AssertionError(f"Unhandled ignore family: {family_name}")


def parse_nodes_csv(nodes_csv: str) -> list[str]:
    return [node.strip() for node in nodes_csv.split(",") if node.strip()]

