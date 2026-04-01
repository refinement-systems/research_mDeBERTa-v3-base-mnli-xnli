#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import venv
from dataclasses import dataclass


DEFAULT_HF_SOURCE = "models/mdeberta"
DEFAULT_OUTPUT = "models/mdeberta/onnx/candidates/qat/attention_only_qat_pilot.onnx"
DEFAULT_VENV = ".venv" if os.path.exists(".venv") else os.path.join(
    tempfile.gettempdir(), "nli-onnx-tools-venv"
)
LABEL_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class Example:
    example_id: str
    premise: str
    hypothesis: str
    gold_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a gated NNCF QAT recovery pass for the local mDeBERTa model and export the "
            "result back to ONNX for the existing benchmark harness."
        )
    )
    parser.add_argument("--hf-source", "--src", default=DEFAULT_HF_SOURCE, dest="hf_source")
    parser.add_argument("--output", "--dest", default=DEFAULT_OUTPUT, dest="output")
    parser.add_argument(
        "--ignored-scope-family",
        choices=["none", "attention_only", "attention_proj_only"],
        default="attention_only",
        help="PTQ/QAT seed family to start from (default: attention_only).",
    )
    parser.add_argument(
        "--train-tsv",
        dest="train_tsvs",
        action="append",
        default=[],
        help="Training TSV with premise/hypothesis/label columns. Repeat to add more.",
    )
    parser.add_argument(
        "--validation-tsv",
        dest="validation_tsvs",
        action="append",
        default=[],
        help="Validation TSV with premise/hypothesis/label columns. Repeat to add more.",
    )
    parser.add_argument(
        "--metric",
        choices=["gold_accuracy", "hf_agreement"],
        default="gold_accuracy",
        help="Validation metric used to report recovery quality (default: gold_accuracy).",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=0,
        help="Optional cap on total training examples after loading (default: unlimited).",
    )
    parser.add_argument(
        "--allow-cpu-epochs",
        type=int,
        default=1,
        help="Maximum epochs allowed on CPU-only runs before requiring an accelerator (default: 1).",
    )
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--venv", default=DEFAULT_VENV)
    return parser.parse_args()


def module_available(module_name: str) -> bool:
    code = (
        "import importlib.util, sys;"
        f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    )
    return subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def ensure_python(args: argparse.Namespace) -> str:
    needed = ("torch", "transformers", "sentencepiece", "numpy", "nncf")
    if all(module_available(module) for module in needed):
        return sys.executable

    if not args.install_deps:
        raise RuntimeError(
            "Python packages 'torch', 'transformers', 'sentencepiece', 'numpy', and 'nncf' "
            "are required. Install them in the active environment or re-run with "
            "--install-deps."
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
            "torch",
            "transformers",
            "sentencepiece",
            "numpy",
            "openvino",
            "nncf",
        ],
        check=True,
    )
    return str(python_path)


def read_examples(tsv_paths: list[pathlib.Path], max_examples: int) -> list[Example]:
    examples: list[Example] = []
    for tsv_path in tsv_paths:
        with tsv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "premise" not in reader.fieldnames or "hypothesis" not in reader.fieldnames:
                raise RuntimeError(f"TSV must include premise and hypothesis columns: {tsv_path}")
            for index, row in enumerate(reader):
                label = (row.get("label") or row.get("gold_label") or "").strip()
                if label not in LABEL_TO_ID:
                    continue
                examples.append(
                    Example(
                        example_id=row.get("id") or f"{tsv_path.stem}-{index + 1}",
                        premise=row["premise"],
                        hypothesis=row["hypothesis"],
                        gold_label=label,
                    )
                )
                if max_examples > 0 and len(examples) >= max_examples:
                    return examples
    if not examples:
        raise RuntimeError("No labeled examples were loaded")
    return examples


def parse_last_json_line(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("QAT helper did not emit a JSON summary")


def main() -> int:
    args = parse_args()
    python_executable = ensure_python(args)

    if not args.train_tsvs:
        raise RuntimeError("At least one --train-tsv is required")
    if not args.validation_tsvs:
        raise RuntimeError("At least one --validation-tsv is required")
    if args.epochs <= 0:
        raise RuntimeError("--epochs must be positive")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be positive")

    train_paths = [pathlib.Path(path) for path in args.train_tsvs]
    validation_paths = [pathlib.Path(path) for path in args.validation_tsvs]
    for path in train_paths + validation_paths:
        if not path.is_file():
            raise RuntimeError(f"TSV not found: {path}")

    output_path = pathlib.Path(args.output)
    if output_path.exists() and not args.force:
        raise RuntimeError(
            f"Output already exists: {output_path}. Re-run with --force or choose a new path."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    helper = r"""
import csv
import json
import pathlib
import random

import nncf
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def read_examples(paths, max_examples):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for index, row in enumerate(reader):
                label = (row.get("label") or row.get("gold_label") or "").strip()
                if label not in LABEL_TO_ID:
                    continue
                rows.append(
                    {
                        "id": row.get("id") or f"{path}-{index + 1}",
                        "premise": row["premise"],
                        "hypothesis": row["hypothesis"],
                        "gold_label": label,
                    }
                )
                if max_examples > 0 and len(rows) >= max_examples:
                    return rows
    if not rows:
        raise RuntimeError("No labeled examples were loaded")
    return rows


def ignored_module_names(family_name, num_layers):
    names = []
    for layer_index in range(num_layers):
        if family_name in ("attention_only", "attention_proj_only"):
            names.append(f"deberta.encoder.layer.{layer_index}.intermediate.dense")
            names.append(f"deberta.encoder.layer.{layer_index}.output.dense")
        if family_name == "attention_proj_only":
            names.append(f"deberta.encoder.layer.{layer_index}.attention.output.dense")
    return names


def device_name():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_inputs(tokenizer, batch):
    encoded = tokenizer(
        [item["premise"] for item in batch],
        [item["hypothesis"] for item in batch],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(
        [LABEL_TO_ID[item["gold_label"]] for item in batch],
        dtype=torch.long,
    )
    return encoded


def single_example_inputs(tokenizer, item):
    encoded = tokenizer(
        [item["premise"]],
        [item["hypothesis"]],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return {key: value for key, value in encoded.items()}


def validate_model(model, tokenizer, rows, metric, reference_labels, batch_size, device):
    model.eval()
    total = 0
    hits = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = model_inputs(tokenizer, batch)
            labels = encoded.pop("labels")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            predicted = logits.argmax(dim=-1).cpu().tolist()
            for item, gold_id, pred_id in zip(batch, labels.tolist(), predicted):
                if metric == "gold_accuracy":
                    reference = ID_TO_LABEL[gold_id]
                else:
                    reference = reference_labels[item["id"]]
                hits += int(ID_TO_LABEL[pred_id] == reference)
                total += 1
    model.train()
    return hits / max(total, 1)


payload = json.loads(pathlib.Path("PAYLOAD.json").read_text(encoding="utf-8"))
train_rows = read_examples(payload["train_tsvs"], payload["max_train_examples"])
validation_rows = read_examples(payload["validation_tsvs"], 0)

device = device_name()
if device == "cpu" and payload["epochs"] > payload["allow_cpu_epochs"]:
    raise RuntimeError(
        f"CPU-only environment detected; refusing {payload['epochs']} epochs because "
        f"--allow-cpu-epochs is {payload['allow_cpu_epochs']}"
    )

tokenizer = AutoTokenizer.from_pretrained(
    payload["hf_source"],
    local_files_only=pathlib.Path(payload["hf_source"]).exists(),
    use_fast=True,
)
float_model = AutoModelForSequenceClassification.from_pretrained(
    payload["hf_source"],
    local_files_only=pathlib.Path(payload["hf_source"]).exists(),
)
float_model.eval()

reference_labels = {}
if payload["metric"] == "hf_agreement":
    with torch.no_grad():
        for start in range(0, len(validation_rows), payload["batch_size"]):
            batch = validation_rows[start : start + payload["batch_size"]]
            encoded = model_inputs(tokenizer, batch)
            encoded.pop("labels")
            logits = float_model(**encoded).logits
            predicted = logits.argmax(dim=-1).tolist()
            for item, pred_id in zip(batch, predicted):
                reference_labels[item["id"]] = ID_TO_LABEL[pred_id]

quantization_dataset = nncf.Dataset(
    train_rows,
    lambda item: single_example_inputs(tokenizer, item),
)

ignored_scope = None
ignored_names = ignored_module_names(
    payload["ignored_scope_family"],
    int(float_model.config.num_hidden_layers),
)
if ignored_names:
    ignored_scope = nncf.IgnoredScope(names=ignored_names, validate=False)

quantized_model = nncf.quantize(
    float_model,
    quantization_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    subset_size=min(payload["ptq_subset_size"], len(train_rows)),
    model_type=nncf.ModelType.TRANSFORMER,
    ignored_scope=ignored_scope,
)
quantized_model.to(device)
quantized_model.train()

optimizer = AdamW(quantized_model.parameters(), lr=payload["learning_rate"])
random.Random(0).shuffle(train_rows)

baseline_metric = validate_model(
    quantized_model,
    tokenizer,
    validation_rows,
    payload["metric"],
    reference_labels,
    payload["batch_size"],
    device,
)

for epoch in range(payload["epochs"]):
    for start in range(0, len(train_rows), payload["batch_size"]):
        batch = train_rows[start : start + payload["batch_size"]]
        encoded = model_inputs(tokenizer, batch)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = quantized_model(**encoded).loss
        loss.backward()
        optimizer.step()

final_metric = validate_model(
    quantized_model,
    tokenizer,
    validation_rows,
    payload["metric"],
    reference_labels,
    payload["batch_size"],
    device,
)

quantized_model.eval()
example_input = model_inputs(tokenizer, validation_rows[:1])
example_labels = example_input.pop("labels")
example_input = {key: value.to(device) for key, value in example_input.items()}

try:
    export_model = nncf.strip(
        quantized_model,
        strip_format=nncf.StripFormat.DQ,
        example_input=example_input,
    )
except Exception:
    export_model = quantized_model

input_names = list(example_input.keys())
dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
dynamic_axes["logits"] = {0: "batch"}

torch.onnx.export(
    export_model,
    tuple(example_input[name] for name in input_names),
    payload["output"],
    input_names=input_names,
    output_names=["logits"],
    dynamic_axes=dynamic_axes,
    opset_version=payload["onnx_opset"],
)

result = {
    "output": payload["output"],
    "device": device,
    "ignored_scope_family": payload["ignored_scope_family"],
    "ignored_module_names": ignored_names,
    "train_examples": len(train_rows),
    "validation_examples": len(validation_rows),
    "baseline_metric": baseline_metric,
    "final_metric": final_metric,
    "epochs": payload["epochs"],
}
print(json.dumps(result))
"""

    payload = {
        "hf_source": args.hf_source,
        "output": str(output_path),
        "ignored_scope_family": args.ignored_scope_family,
        "train_tsvs": [str(path) for path in train_paths],
        "validation_tsvs": [str(path) for path in validation_paths],
        "metric": args.metric,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_train_examples": args.max_train_examples,
        "allow_cpu_epochs": args.allow_cpu_epochs,
        "onnx_opset": args.onnx_opset,
        "ptq_subset_size": min(args.batch_size * 32, args.max_train_examples or 128),
    }

    with tempfile.TemporaryDirectory(prefix="nli-qat-recovery-") as tmp_dir:
        payload_path = pathlib.Path(tmp_dir) / "PAYLOAD.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        helper_code = helper.replace("PAYLOAD.json", str(payload_path))
        result = subprocess.run(
            [python_executable, "-c", helper_code],
            text=True,
            capture_output=True,
            check=True,
        )

    report = parse_last_json_line(result.stdout)
    print(f"generated: {report['output']}")
    print(
        f"  config: device={report['device']} epochs={report['epochs']} "
        f"ignored_scope_family={report['ignored_scope_family']} "
        f"ignored_module_names={report['ignored_module_names']}"
    )
    print(
        f"  metrics: baseline={report['baseline_metric']:.6f} "
        f"final={report['final_metric']:.6f}"
    )
    print(
        f"  datasets: train_examples={report['train_examples']} "
        f"validation_examples={report['validation_examples']}"
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
