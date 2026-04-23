#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from replication_cpu_final import build_attempt4_cpu_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the attempt4 CPU deployment-study report by joining study summaries, "
            "development gates, and CPU runtime/RSS benchmarks."
        )
    )
    parser.add_argument("--datasets-manifest", required=True)
    parser.add_argument("--validation-summary-json", required=True)
    parser.add_argument("--validation-runtime-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--test-summary-json", default="")
    parser.add_argument("--stress-summary-json", default="")
    parser.add_argument("--cold-benchmark-csv", default="")
    parser.add_argument("--report-markdown", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_attempt4_cpu_report(
        pathlib.Path(args.datasets_manifest).resolve(),
        pathlib.Path(args.validation_summary_json).resolve(),
        pathlib.Path(args.validation_runtime_csv).resolve(),
        pathlib.Path(args.output_prefix).resolve(),
        test_summary_path=pathlib.Path(args.test_summary_json).resolve() if args.test_summary_json else None,
        stress_summary_path=pathlib.Path(args.stress_summary_json).resolve() if args.stress_summary_json else None,
        cold_benchmark_path=pathlib.Path(args.cold_benchmark_csv).resolve() if args.cold_benchmark_csv else None,
        report_markdown_path=pathlib.Path(args.report_markdown).resolve() if args.report_markdown else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
