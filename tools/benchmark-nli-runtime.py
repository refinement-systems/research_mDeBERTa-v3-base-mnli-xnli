#!/usr/bin/env python3

import argparse
import csv
import json
import math
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EXECUTABLE = REPO_ROOT / "builddir/nli"
DEFAULT_PERSISTENT_EXECUTABLE = REPO_ROOT / "builddir/nli-runtime-bench"
DEFAULT_TSVS = [REPO_ROOT / "benchmarks/nli/hf-core-probe.tsv"]
DEFAULT_MODELS = [
    (
        "float",
        REPO_ROOT / "models/mdeberta/onnx/model.onnx",
    ),
    (
        "attention_only",
        REPO_ROOT / "models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx",
    ),
    (
        "attention_proj_only",
        REPO_ROOT / "models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx",
    ),
]


@dataclass
class Example:
    benchmark: str
    example_id: str
    premise: str
    hypothesis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark runtime for builddir/nli across one or more TSV probes, summarizing "
            "cold model load and warm inference latency for multiple models/backends."
        )
    )
    parser.add_argument(
        "--executable",
        default=str(DEFAULT_EXECUTABLE),
        help=f"Path to nli executable (default: {DEFAULT_EXECUTABLE})",
    )
    parser.add_argument(
        "--persistent-executable",
        default=str(DEFAULT_PERSISTENT_EXECUTABLE),
        help=f"Path to persistent runtime benchmark executable (default: {DEFAULT_PERSISTENT_EXECUTABLE})",
    )
    parser.add_argument(
        "--mode",
        choices=["coldstart", "persistent"],
        default="coldstart",
        help="Runtime benchmark mode (default: coldstart)",
    )
    parser.add_argument(
        "--tsv",
        dest="tsv_paths",
        action="append",
        default=[],
        help="Probe TSV path. Repeat to add more. Defaults to hf-core-probe.tsv.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=[],
        help="Model in NAME=PATH form. Repeat to add more.",
    )
    parser.add_argument(
        "--backend",
        dest="backends",
        action="append",
        default=[],
        help="Backend to benchmark. Repeat to add more. Defaults to cpu and coreml on macOS, otherwise cpu.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Timed runs per example (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per example (default: 1)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum total examples after sampling (default: 5; use 0 for all)",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="random",
        help="How to sample examples from the provided TSVs (default: random)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --sample-mode=random (default: 0)",
    )
    parser.add_argument(
        "--show-slices",
        action="store_true",
        help="Print per-benchmark runtime summaries.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> list[tuple[str, pathlib.Path]]:
    if not args.models:
        return [(name, pathlib.Path(path)) for name, path in DEFAULT_MODELS]

    resolved = []
    for item in args.models:
        if "=" not in item:
            raise RuntimeError(f"--model must be NAME=PATH: {item}")
        name, path = item.split("=", 1)
        resolved.append((name.strip(), pathlib.Path(path.strip())))
    return resolved


def resolve_backends(args: argparse.Namespace) -> list[str]:
    if args.backends:
        return args.backends
    return ["cpu", "coreml"] if platform.system() == "Darwin" else ["cpu"]


def read_examples(
    tsv_paths: list[pathlib.Path],
    sample_mode: str,
    max_examples: int,
    seed: int,
) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for tsv_path in tsv_paths:
        with tsv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "premise" not in reader.fieldnames or "hypothesis" not in reader.fieldnames:
                raise RuntimeError(f"TSV must include premise and hypothesis columns: {tsv_path}")
            source_rows = [
                Example(
                    benchmark=(row.get("benchmark") or "").strip() or tsv_path.name,
                    example_id=row.get("id") or f"{tsv_path.stem}-{index + 1}",
                    premise=row["premise"],
                    hypothesis=row["hypothesis"],
                )
                for index, row in enumerate(reader)
            ]
        if sample_mode == "random":
            rng.shuffle(source_rows)
        examples.extend(source_rows)

    if sample_mode == "random":
        rng.shuffle(examples)
    if max_examples > 0:
        examples = examples[:max_examples]
    if not examples:
        raise RuntimeError("No runtime benchmark examples loaded")
    return examples


def parse_key_value_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_structured_line(stdout: str, prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        marker = f"{prefix}: "
        if not line.startswith(marker):
            continue
        payload = line[len(marker):]
        row: dict[str, str] = {}
        for field in payload.split("\t"):
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            row[key] = value
        rows.append(row)
    return rows


def parse_float(value: str) -> float:
    return float(value)


def parse_float_list(value: str) -> list[float]:
    if not value:
        return []
    return [float(item) for item in value.split()]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, math.ceil(pct * len(sorted_values)) - 1)
    return sorted_values[min(index, len(sorted_values) - 1)]


def summarize_numeric(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def benchmark_model_backend_coldstart(
    executable: pathlib.Path,
    model_path: pathlib.Path,
    backend: str,
    repeat: int,
    warmup: int,
    examples: list[Example],
) -> dict[str, object]:
    load_values: list[float] = []
    warm_run_values: list[float] = []
    per_example: list[dict[str, object]] = []

    for example in examples:
        command = [
            str(executable),
            "-b",
            backend,
            "--model",
            str(model_path),
            "--premise",
            example.premise,
            "--hypothesis",
            example.hypothesis,
            "--repeat",
            str(repeat),
            "--warmup",
            str(warmup),
            "--timing",
            "--dump-timing-runs",
            "--quiet",
        ]
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Runtime benchmark command failed:\n"
                f"command={' '.join(command)}\n"
                f"stdout={completed.stdout}\n"
                f"stderr={completed.stderr}"
            )

        parsed = parse_key_value_output(completed.stdout)
        load_ms = parse_float(parsed["load_ms"])
        runs_ms = parse_float_list(parsed.get("timing_runs_ms", ""))
        load_values.append(load_ms)
        warm_run_values.extend(runs_ms)
        per_example.append(
            {
                "benchmark": example.benchmark,
                "id": example.example_id,
                "load_ms": load_ms,
                "timing_mean_ms": parse_float(parsed["timing_mean_ms"]),
                "timing_median_ms": parse_float(parsed["timing_median_ms"]),
                "timing_p95_ms": parse_float(parsed["timing_p95_ms"]),
                "timing_runs_ms": runs_ms,
            }
        )

    per_benchmark: dict[str, dict[str, object]] = {}
    for benchmark_name in sorted({row["benchmark"] for row in per_example}):
        benchmark_rows = [row for row in per_example if row["benchmark"] == benchmark_name]
        benchmark_loads = [row["load_ms"] for row in benchmark_rows]
        benchmark_runs = [
            run_ms
            for row in benchmark_rows
            for run_ms in row["timing_runs_ms"]
        ]
        per_benchmark[benchmark_name] = {
            "examples": len(benchmark_rows),
            "load_ms": summarize_numeric(benchmark_loads),
            "warm_latency_ms": summarize_numeric(benchmark_runs),
        }

    return {
        "examples": len(examples),
        "file_size_bytes": model_path.stat().st_size,
        "load_ms": summarize_numeric(load_values),
        "warm_latency_ms": summarize_numeric(warm_run_values),
        "per_benchmark": per_benchmark,
        "per_example": per_example,
    }


def write_temp_examples_tsv(examples: list[Example]) -> pathlib.Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix="nli-runtime-bench-",
        suffix=".tsv",
        dir="/tmp",
        delete=False,
    )
    path = pathlib.Path(handle.name)
    with handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["benchmark", "id", "premise", "hypothesis"],
            delimiter="\t",
        )
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "benchmark": example.benchmark,
                    "id": example.example_id,
                    "premise": example.premise,
                    "hypothesis": example.hypothesis,
                }
            )
    return path


def benchmark_model_backend_persistent(
    executable: pathlib.Path,
    model_path: pathlib.Path,
    backend: str,
    repeat: int,
    warmup: int,
    examples: list[Example],
) -> dict[str, object]:
    temp_tsv = write_temp_examples_tsv(examples)
    try:
        command = [
            str(executable),
            "-b",
            backend,
            "--model",
            str(model_path),
            "--repeat",
            str(repeat),
            "--warmup",
            str(warmup),
            "--dump-example-timings",
            str(temp_tsv),
        ]
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Persistent runtime benchmark command failed:\n"
                f"command={' '.join(command)}\n"
                f"stdout={completed.stdout}\n"
                f"stderr={completed.stderr}"
            )

        parsed = parse_key_value_output(completed.stdout)
        benchmark_rows = parse_structured_line(completed.stdout, "benchmark_timing")
        example_rows = parse_structured_line(completed.stdout, "example_timing")

        per_benchmark = {}
        for row in benchmark_rows:
            benchmark_name = row["benchmark"]
            per_benchmark[benchmark_name] = {
                "examples": sum(1 for example in examples if example.benchmark == benchmark_name),
                "warm_latency_ms": {
                    "mean": parse_float(row["mean_ms"]),
                    "median": parse_float(row["median_ms"]),
                    "p95": parse_float(row["p95_ms"]),
                    "min": parse_float(row["min_ms"]),
                    "max": parse_float(row["max_ms"]),
                },
            }

        per_example = [
            {
                "benchmark": row["benchmark"],
                "id": row["id"],
                "timing_mean_ms": parse_float(row["mean_ms"]),
                "timing_median_ms": parse_float(row["median_ms"]),
                "timing_p95_ms": parse_float(row["p95_ms"]),
                "timing_min_ms": parse_float(row["min_ms"]),
                "timing_max_ms": parse_float(row["max_ms"]),
            }
            for row in example_rows
        ]

        load_ms = parse_float(parsed["load_ms"])
        return {
            "examples": len(examples),
            "file_size_bytes": model_path.stat().st_size,
            "load_ms": summarize_numeric([load_ms]),
            "warm_latency_ms": {
                "mean": parse_float(parsed["timing_mean_ms"]),
                "median": parse_float(parsed["timing_median_ms"]),
                "p95": parse_float(parsed["timing_p95_ms"]),
                "min": parse_float(parsed["timing_min_ms"]),
                "max": parse_float(parsed["timing_max_ms"]),
            },
            "per_benchmark": per_benchmark,
            "per_example": per_example,
        }
    finally:
        temp_tsv.unlink(missing_ok=True)


def print_summary(rows: list[dict[str, object]]) -> None:
    header = (
        f"{'candidate':<22} {'backend':<8} {'mode':<11} {'size_mb':>8} "
        f"{'load_med':>10} {'warm_med':>10} {'warm_p95':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['candidate']:<22} "
            f"{row['backend']:<8} "
            f"{row['mode']:<11} "
            f"{row['file_size_bytes'] / (1024.0 * 1024.0):>7.2f} "
            f"{row['load_ms']['median']:>10.3f} "
            f"{row['warm_latency_ms']['median']:>10.3f} "
            f"{row['warm_latency_ms']['p95']:>10.3f}"
        )


def print_per_benchmark(rows: list[dict[str, object]]) -> None:
    for row in rows:
        print()
        print(f"{row['candidate']} [{row['backend']}, {row['mode']}]")
        for benchmark_name, benchmark_summary in row["per_benchmark"].items():
            parts = [f"  {benchmark_name:<42}"]
            if "load_ms" in benchmark_summary:
                parts.append(f"load_med={benchmark_summary['load_ms']['median']:.3f}")
            parts.append(f"warm_med={benchmark_summary['warm_latency_ms']['median']:.3f}")
            parts.append(f"warm_p95={benchmark_summary['warm_latency_ms']['p95']:.3f}")
            print(" ".join(parts))


def write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate",
        "backend",
        "mode",
        "examples",
        "file_size_bytes",
        "load_mean_ms",
        "load_median_ms",
        "load_p95_ms",
        "warm_mean_ms",
        "warm_median_ms",
        "warm_p95_ms",
        "warm_min_ms",
        "warm_max_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate": row["candidate"],
                    "backend": row["backend"],
                    "mode": row["mode"],
                    "examples": row["examples"],
                    "file_size_bytes": row["file_size_bytes"],
                    "load_mean_ms": row["load_ms"]["mean"],
                    "load_median_ms": row["load_ms"]["median"],
                    "load_p95_ms": row["load_ms"]["p95"],
                    "warm_mean_ms": row["warm_latency_ms"]["mean"],
                    "warm_median_ms": row["warm_latency_ms"]["median"],
                    "warm_p95_ms": row["warm_latency_ms"]["p95"],
                    "warm_min_ms": row["warm_latency_ms"]["min"],
                    "warm_max_ms": row["warm_latency_ms"]["max"],
                }
            )


def main() -> int:
    args = parse_args()
    executable = pathlib.Path(args.executable)
    persistent_executable = pathlib.Path(args.persistent_executable)
    if args.mode == "coldstart":
        if not executable.is_file():
            raise RuntimeError(f"nli executable not found: {executable}")
    else:
        if not persistent_executable.is_file():
            raise RuntimeError(f"persistent runtime benchmark executable not found: {persistent_executable}")

    models = resolve_models(args)
    for _, model_path in models:
        if not model_path.is_file():
            raise RuntimeError(f"Model file not found: {model_path}")

    tsv_paths = [pathlib.Path(path) for path in (args.tsv_paths or DEFAULT_TSVS)]
    for tsv_path in tsv_paths:
        if not tsv_path.is_file():
            raise RuntimeError(f"Probe TSV not found: {tsv_path}")

    if args.repeat <= 0:
        raise RuntimeError("--repeat must be positive")
    if args.warmup < 0:
        raise RuntimeError("--warmup must be non-negative")

    examples = read_examples(tsv_paths, args.sample_mode, args.max_examples, args.seed)
    backends = resolve_backends(args)

    rows: list[dict[str, object]] = []
    for candidate_name, model_path in models:
        for backend in backends:
            if args.mode == "coldstart":
                summary = benchmark_model_backend_coldstart(
                    executable,
                    model_path,
                    backend,
                    args.repeat,
                    args.warmup,
                    examples,
                )
            else:
                summary = benchmark_model_backend_persistent(
                    persistent_executable,
                    model_path,
                    backend,
                    args.repeat,
                    args.warmup,
                    examples,
                )
            rows.append(
                {
                    "candidate": candidate_name,
                    "backend": backend,
                    "mode": args.mode,
                    **summary,
                }
            )

    rows.sort(
        key=lambda row: (
            row["backend"],
            row["warm_latency_ms"]["median"],
            row["load_ms"]["median"],
        )
    )

    print(f"examples: {len(examples)}")
    print(f"sources: {', '.join(sorted({example.benchmark for example in examples}))}")
    print_summary(rows)
    if args.show_slices:
        print_per_benchmark(rows)

    payload = {
        "examples": len(examples),
        "sources": sorted({example.benchmark for example in examples}),
        "mode": args.mode,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "sample_mode": args.sample_mode,
        "max_examples": args.max_examples,
        "results": rows,
    }
    if args.summary_json:
        write_json(pathlib.Path(args.summary_json), payload)
    if args.summary_csv:
        write_csv(pathlib.Path(args.summary_csv), rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
