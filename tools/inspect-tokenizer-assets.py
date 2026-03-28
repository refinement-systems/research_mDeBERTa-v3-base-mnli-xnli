#!/usr/bin/env python3

import argparse
import json
import pathlib
import re
import subprocess
import sys


DEFAULT_ASSET_DIR = "models/mdeberta"
DEFAULT_CPP_BINARY = "builddir/nli"
DEFAULT_CPP_MODEL = "models/mdeberta/onnx/model_quantized.onnx"
DEFAULT_PROBE_PREMISE = "  Hello\tworld \n from\r\nCodex  "
DEFAULT_PROBE_HYPOTHESIS = "  Another\tinput \nline  "
DEFAULT_SPECIAL_TOKEN_PROBE_PREMISE = "Literal [MASK] token"
DEFAULT_SPECIAL_TOKEN_PROBE_HYPOTHESIS = "Control example"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the local Hugging Face tokenizer assets and compare them "
            "against the current C++ SentencePiece-based runtime assumptions."
        )
    )
    parser.add_argument(
        "--asset-dir",
        default=DEFAULT_ASSET_DIR,
        help=f"Directory containing tokenizer.json and related files (default: {DEFAULT_ASSET_DIR})",
    )
    parser.add_argument(
        "--cpp-binary",
        default=DEFAULT_CPP_BINARY,
        help=f"C++ CLI used for runtime diagnostics (default: {DEFAULT_CPP_BINARY})",
    )
    parser.add_argument(
        "--cpp-model",
        default=DEFAULT_CPP_MODEL,
        help=f"Model path passed to the C++ CLI (default: {DEFAULT_CPP_MODEL})",
    )
    parser.add_argument(
        "--premise",
        default=DEFAULT_PROBE_PREMISE,
        help="Probe premise text for normalization diagnostics",
    )
    parser.add_argument(
        "--hypothesis",
        default=DEFAULT_PROBE_HYPOTHESIS,
        help="Probe hypothesis text for normalization diagnostics",
    )
    return parser.parse_args()


def require_json(path: pathlib.Path) -> object:
    if not path.exists():
        raise RuntimeError(
            f"Required asset missing: {path}. "
            "Fetch it with tools/download-mdeberta-v3-base.sh --tokenizer-assets"
        )
    return json.loads(path.read_text())


def load_assets(asset_dir: pathlib.Path) -> dict[str, object]:
    return {
        "tokenizer_json": require_json(asset_dir / "tokenizer.json"),
        "tokenizer_config": require_json(asset_dir / "tokenizer_config.json"),
        "special_tokens_map": require_json(asset_dir / "special_tokens_map.json"),
        "added_tokens": require_json(asset_dir / "added_tokens.json"),
    }


def parse_cpp_output(output: str) -> dict[str, object]:
    special_match = re.search(
        r"^special_token_ids:\s+pad=(?P<pad>-?\d+)\s+cls=(?P<cls>-?\d+)\s+sep=(?P<sep>-?\d+)\s+unk=(?P<unk>-?\d+)\s+mask=(?P<mask>-?\d+)$",
        output,
        re.MULTILINE,
    )
    if not special_match:
        raise RuntimeError("Failed to parse special_token_ids from C++ output")

    premise_match = re.search(r"^normalized_premise:\s?(.*)$", output, re.MULTILINE)
    hypothesis_match = re.search(r"^normalized_hypothesis:\s?(.*)$", output, re.MULTILINE)
    input_ids_match = re.search(r"^input_ids:(?P<input_ids>(?:\s+-?\d+)*)$", output, re.MULTILINE)
    if not premise_match or not hypothesis_match or not input_ids_match:
        raise RuntimeError("Failed to parse encoding details from C++ output")

    return {
        "special_token_ids": {
            key: int(value) for key, value in special_match.groupdict().items()
        },
        "normalized_premise": premise_match.group(1),
        "normalized_hypothesis": hypothesis_match.group(1),
        "input_ids": [
            int(value) for value in input_ids_match.group("input_ids").split()
        ],
    }


def run_cpp_probe(args: argparse.Namespace) -> dict[str, object]:
    return run_cpp_probe_for_text(args, args.premise, args.hypothesis)


def run_cpp_probe_for_text(
    args: argparse.Namespace,
    premise: str,
    hypothesis: str,
) -> dict[str, object]:
    command = [
        args.cpp_binary,
        "-b",
        "cpu",
        "--model",
        args.cpp_model,
        "--dump-special-token-ids",
        "--dump-encoding",
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
    return parse_cpp_output(result.stdout)


def hf_special_token_ids(tokenizer_json: dict[str, object]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for token in tokenizer_json.get("added_tokens", []):
        content = token.get("content")
        token_id = token.get("id")
        if content == "[PAD]":
            ids["pad"] = token_id
        elif content == "[CLS]":
            ids["cls"] = token_id
        elif content == "[SEP]":
            ids["sep"] = token_id
        elif content == "[UNK]":
            ids["unk"] = token_id
        elif content == "[MASK]":
            ids["mask"] = token_id
    return ids


def concise_normalizer(tokenizer_json: dict[str, object]) -> object:
    normalizer = tokenizer_json.get("normalizer")
    if not normalizer:
        return None
    if normalizer.get("type") != "Sequence":
        return normalizer

    concise = []
    for item in normalizer.get("normalizers", []):
        summary = {key: value for key, value in item.items() if key != "precompiled_charsmap"}
        if "precompiled_charsmap" in item:
            summary["precompiled_charsmap_len"] = len(item["precompiled_charsmap"])
        concise.append(summary)
    return concise


def compare_token_ids(hf_ids: dict[str, int], cpp_ids: dict[str, int]) -> None:
    print("special_token_id_comparison:")
    for key in ["pad", "cls", "sep", "unk", "mask"]:
        hf_value = hf_ids.get(key)
        cpp_value = cpp_ids.get(key)
        status = "MATCH" if hf_value == cpp_value else "DIFF"
        print(f"  {key}: hf={hf_value} cpp={cpp_value} -> {status}")


def print_normalization_diagnostics(
    tokenizer_json: dict[str, object],
    cpp_probe: dict[str, object],
    args: argparse.Namespace,
) -> None:
    pre_tokenizer = tokenizer_json.get("pre_tokenizer", {})
    print("normalization_probe:")
    print(f"  premise_input={args.premise!r}")
    print(f"  premise_cpp_normalized={cpp_probe['normalized_premise']!r}")
    print(f"  hypothesis_input={args.hypothesis!r}")
    print(f"  hypothesis_cpp_normalized={cpp_probe['normalized_hypothesis']!r}")
    print(f"  hf_normalizer={json.dumps(concise_normalizer(tokenizer_json))}")
    print(f"  hf_pre_tokenizer={json.dumps(pre_tokenizer)}")


def print_inference_notes(tokenizer_json: dict[str, object], cpp_probe: dict[str, object]) -> None:
    hf_ids = hf_special_token_ids(tokenizer_json)
    if hf_ids.get("mask") != cpp_probe["special_token_ids"].get("mask"):
        print("notes:")
        print(
            "  [MASK] differs between tokenizer.json and bare SentencePiece. "
            "That is expected because this checkpoint uses added-token metadata "
            "outside spm.model."
        )
        print(
            "  This mismatch is usually low-risk for NLI inference because [MASK] "
            "is not part of normal premise/hypothesis inputs."
        )


def print_special_token_text_probe(
    tokenizer_json: dict[str, object],
    cpp_probe: dict[str, object],
) -> None:
    mask_id = hf_special_token_ids(tokenizer_json).get("mask")
    print("special_token_text_probe:")
    print(f"  premise_input={DEFAULT_SPECIAL_TOKEN_PROBE_PREMISE!r}")
    print(f"  hypothesis_input={DEFAULT_SPECIAL_TOKEN_PROBE_HYPOTHESIS!r}")
    print(f"  cpp_input_ids={cpp_probe['input_ids']}")
    print(f"  expected_mask_id={mask_id}")
    print(f"  mask_id_present={mask_id in cpp_probe['input_ids']}")


def main() -> int:
    args = parse_args()
    asset_dir = pathlib.Path(args.asset_dir)
    assets = load_assets(asset_dir)
    cpp_probe = run_cpp_probe(args)
    special_token_probe = run_cpp_probe_for_text(
        args,
        DEFAULT_SPECIAL_TOKEN_PROBE_PREMISE,
        DEFAULT_SPECIAL_TOKEN_PROBE_HYPOTHESIS,
    )

    tokenizer_json = assets["tokenizer_json"]
    hf_ids = hf_special_token_ids(tokenizer_json)

    print(f"asset_dir: {asset_dir}")
    print(
        f"tokenizer_class: {assets['tokenizer_config'].get('tokenizer_class')} "
        f"vocab_type: {assets['tokenizer_config'].get('vocab_type')}"
    )
    compare_token_ids(hf_ids, cpp_probe["special_token_ids"])
    print_normalization_diagnostics(tokenizer_json, cpp_probe, args)
    print_special_token_text_probe(tokenizer_json, special_token_probe)
    print_inference_notes(tokenizer_json, cpp_probe)
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
