from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replication_cpu_final = load_module(
    REPO_ROOT / "tools" / "replication_cpu_final.py",
    "replication_cpu_final_report",
)


class BuildAttempt4CpuReportTest(unittest.TestCase):
    def summary_row(
        self,
        *,
        dataset: str,
        quantization: str,
        role: str,
        gold_accuracy: float,
        float_label_agreement: float,
        size_bytes: int,
        mean_abs_logit_delta: float = 0.1,
        max_abs_logit_delta: float = 0.2,
        example_count: int = 10,
        disagreement_count: int = 0,
    ) -> dict[str, object]:
        return {
            "dataset": dataset,
            "role": role,
            "backend": "cpu",
            "quantization": quantization,
            "artifact_path": f"/tmp/{quantization}.onnx",
            "stdout_log_path": "",
            "size_bytes": size_bytes,
            "example_count": example_count,
            "labeled_example_count": example_count,
            "correct_prediction_count": int(round(example_count * gold_accuracy)),
            "gold_accuracy": gold_accuracy,
            "float_label_agreement": float_label_agreement,
            "mean_abs_logit_delta": mean_abs_logit_delta,
            "max_abs_logit_delta": max_abs_logit_delta,
            "disagreement_count": disagreement_count,
            "smooth_quant_disabled": None,
            "retry_reason": "",
            "pareto_frontier": True,
        }

    def write_json(self, path: pathlib.Path, payload: dict[str, object]) -> pathlib.Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def write_benchmark_csv(self, path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=replication_cpu_final.benchmark_csv_fieldnames())
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def runtime_row(
        self,
        *,
        candidate: str,
        load_ms: float = 10.0,
        warm_ms: float = 8.0,
        resident_after_warmup_bytes: int = 90,
        peak_rss_after_timed_runs_bytes: int = 100,
    ) -> dict[str, object]:
        return {
            "candidate": candidate,
            "backend": "cpu",
            "mode": "persistent",
            "examples": 2,
            "file_size_bytes": 1234,
            "load_mean_ms": load_ms,
            "load_median_ms": load_ms,
            "load_p95_ms": load_ms,
            "warm_mean_ms": warm_ms,
            "warm_median_ms": warm_ms,
            "warm_p95_ms": warm_ms,
            "warm_min_ms": warm_ms,
            "warm_max_ms": warm_ms,
            "resident_after_load_mean_bytes": resident_after_warmup_bytes,
            "resident_after_load_median_bytes": resident_after_warmup_bytes,
            "resident_after_load_p95_bytes": resident_after_warmup_bytes,
            "resident_after_warmup_mean_bytes": resident_after_warmup_bytes,
            "resident_after_warmup_median_bytes": resident_after_warmup_bytes,
            "resident_after_warmup_p95_bytes": resident_after_warmup_bytes,
            "resident_after_timed_runs_mean_bytes": resident_after_warmup_bytes,
            "resident_after_timed_runs_median_bytes": resident_after_warmup_bytes,
            "resident_after_timed_runs_p95_bytes": resident_after_warmup_bytes,
            "peak_rss_after_load_mean_bytes": peak_rss_after_timed_runs_bytes,
            "peak_rss_after_load_median_bytes": peak_rss_after_timed_runs_bytes,
            "peak_rss_after_load_p95_bytes": peak_rss_after_timed_runs_bytes,
            "peak_rss_after_warmup_mean_bytes": peak_rss_after_timed_runs_bytes,
            "peak_rss_after_warmup_median_bytes": peak_rss_after_timed_runs_bytes,
            "peak_rss_after_warmup_p95_bytes": peak_rss_after_timed_runs_bytes,
            "peak_rss_after_timed_runs_mean_bytes": peak_rss_after_timed_runs_bytes,
            "peak_rss_after_timed_runs_median_bytes": peak_rss_after_timed_runs_bytes,
            "peak_rss_after_timed_runs_p95_bytes": peak_rss_after_timed_runs_bytes,
            "time_l_peak_rss_mean_bytes": "",
            "time_l_peak_rss_median_bytes": "",
            "time_l_peak_rss_p95_bytes": "",
        }

    def cold_row(self, *, candidate: str, load_ms: float, warm_ms: float) -> dict[str, object]:
        row = self.runtime_row(
            candidate=candidate,
            load_ms=load_ms,
            warm_ms=warm_ms,
            resident_after_warmup_bytes=0,
            peak_rss_after_timed_runs_bytes=0,
        )
        row["mode"] = "coldstart"
        return row

    def test_build_attempt4_cpu_report_applies_gate_frontier_and_recommendation_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            manifest_path = self.write_json(
                root / "attempt4-datasets-manifest.json",
                {
                    "validation_datasets": [
                        "mnli-validation_matched-attempt4-dev.tsv",
                        "anli-r1-dev-attempt4-dev.tsv",
                    ],
                    "test_datasets": ["mnli-validation_matched-attempt4-test.tsv"],
                    "stress_datasets": ["wanli-test-attempt4-stress-test.tsv"],
                },
            )

            validation_rows = [
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-dev.tsv",
                    quantization="reference",
                    role="fidelity_validation",
                    gold_accuracy=0.90,
                    float_label_agreement=1.0,
                    size_bytes=1000,
                ),
                self.summary_row(
                    dataset="anli-r1-dev-attempt4-dev.tsv",
                    quantization="reference",
                    role="fidelity_validation",
                    gold_accuracy=0.80,
                    float_label_agreement=1.0,
                    size_bytes=1000,
                ),
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-dev.tsv",
                    quantization="candidate_a",
                    role="fidelity_validation",
                    gold_accuracy=0.898,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
                self.summary_row(
                    dataset="anli-r1-dev-attempt4-dev.tsv",
                    quantization="candidate_a",
                    role="fidelity_validation",
                    gold_accuracy=0.795,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-dev.tsv",
                    quantization="candidate_b",
                    role="fidelity_validation",
                    gold_accuracy=0.898,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
                self.summary_row(
                    dataset="anli-r1-dev-attempt4-dev.tsv",
                    quantization="candidate_b",
                    role="fidelity_validation",
                    gold_accuracy=0.795,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-dev.tsv",
                    quantization="candidate_c",
                    role="fidelity_validation",
                    gold_accuracy=0.897,
                    float_label_agreement=0.99,
                    size_bytes=500,
                ),
                self.summary_row(
                    dataset="anli-r1-dev-attempt4-dev.tsv",
                    quantization="candidate_c",
                    role="fidelity_validation",
                    gold_accuracy=0.794,
                    float_label_agreement=0.99,
                    size_bytes=500,
                ),
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-dev.tsv",
                    quantization="candidate_fail",
                    role="fidelity_validation",
                    gold_accuracy=0.899,
                    float_label_agreement=0.97,
                    size_bytes=450,
                    disagreement_count=1,
                ),
                self.summary_row(
                    dataset="anli-r1-dev-attempt4-dev.tsv",
                    quantization="candidate_fail",
                    role="fidelity_validation",
                    gold_accuracy=0.799,
                    float_label_agreement=0.97,
                    size_bytes=450,
                    disagreement_count=1,
                ),
            ]
            validation_summary_path = self.write_json(root / "validation-summary.json", {"rows": validation_rows})

            test_rows = [
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-test.tsv",
                    quantization="reference",
                    role="fidelity_test",
                    gold_accuracy=0.90,
                    float_label_agreement=1.0,
                    size_bytes=1000,
                ),
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-test.tsv",
                    quantization="candidate_a",
                    role="fidelity_test",
                    gold_accuracy=0.88,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-test.tsv",
                    quantization="candidate_b",
                    role="fidelity_test",
                    gold_accuracy=0.87,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-test.tsv",
                    quantization="candidate_c",
                    role="fidelity_test",
                    gold_accuracy=0.86,
                    float_label_agreement=0.99,
                    size_bytes=500,
                ),
            ]
            test_summary_path = self.write_json(root / "test-summary.json", {"rows": test_rows})

            stress_rows = [
                self.summary_row(
                    dataset="wanli-test-attempt4-stress-test.tsv",
                    quantization="reference",
                    role="stress_test",
                    gold_accuracy=0.70,
                    float_label_agreement=1.0,
                    size_bytes=1000,
                ),
                self.summary_row(
                    dataset="wanli-test-attempt4-stress-test.tsv",
                    quantization="candidate_a",
                    role="stress_test",
                    gold_accuracy=0.69,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
                self.summary_row(
                    dataset="wanli-test-attempt4-stress-test.tsv",
                    quantization="candidate_b",
                    role="stress_test",
                    gold_accuracy=0.68,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
                self.summary_row(
                    dataset="wanli-test-attempt4-stress-test.tsv",
                    quantization="candidate_c",
                    role="stress_test",
                    gold_accuracy=0.67,
                    float_label_agreement=0.99,
                    size_bytes=500,
                ),
            ]
            stress_summary_path = self.write_json(root / "stress-summary.json", {"rows": stress_rows})

            validation_runtime_path = self.write_benchmark_csv(
                root / "validation-runtime.csv",
                [
                    self.runtime_row(
                        candidate="reference",
                        warm_ms=12.0,
                        resident_after_warmup_bytes=120,
                        peak_rss_after_timed_runs_bytes=100,
                    ),
                    self.runtime_row(
                        candidate="candidate_a",
                        warm_ms=8.0,
                        resident_after_warmup_bytes=90,
                        peak_rss_after_timed_runs_bytes=110,
                    ),
                    self.runtime_row(
                        candidate="candidate_b",
                        warm_ms=8.0,
                        resident_after_warmup_bytes=90,
                        peak_rss_after_timed_runs_bytes=110,
                    ),
                    self.runtime_row(
                        candidate="candidate_c",
                        warm_ms=9.0,
                        resident_after_warmup_bytes=100,
                        peak_rss_after_timed_runs_bytes=115,
                    ),
                    self.runtime_row(
                        candidate="candidate_fail",
                        warm_ms=7.5,
                        resident_after_warmup_bytes=95,
                        peak_rss_after_timed_runs_bytes=110,
                    ),
                ],
            )
            cold_benchmark_path = self.write_benchmark_csv(
                root / "cold-runtime.csv",
                [
                    self.cold_row(candidate="reference", load_ms=15.0, warm_ms=12.0),
                    self.cold_row(candidate="candidate_a", load_ms=10.0, warm_ms=8.0),
                    self.cold_row(candidate="candidate_b", load_ms=10.5, warm_ms=8.0),
                    self.cold_row(candidate="candidate_c", load_ms=11.0, warm_ms=9.0),
                ],
            )

            result = replication_cpu_final.build_attempt4_cpu_report(
                manifest_path,
                validation_summary_path,
                validation_runtime_path,
                root / "attempt4-cpu-summary",
                test_summary_path=test_summary_path,
                stress_summary_path=stress_summary_path,
                cold_benchmark_path=cold_benchmark_path,
            )

            self.assertFalse(result["partial"])
            self.assertEqual(
                result["locked_quantizations"],
                ["candidate_a", "candidate_b", "candidate_c", "reference"],
            )
            self.assertEqual(result["final_frontier"], ["candidate_a", "candidate_b"])
            self.assertEqual(result["recommendation"]["quantization"], "candidate_a")

            payload = json.loads(result["paths"]["candidate_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["locked_quantizations"], ["candidate_a", "candidate_b", "candidate_c", "reference"])
            self.assertEqual(payload["final_frontier"], ["candidate_a", "candidate_b"])
            self.assertEqual(payload["recommendation"]["quantization"], "candidate_a")

            candidate_fail = next(row for row in payload["candidates"] if row["quantization"] == "candidate_fail")
            self.assertFalse(candidate_fail["validation_gate_pass"])
            self.assertIn("aggregate float-label agreement is below 98.0%", candidate_fail["validation_gate_reason_text"])

    def test_incomplete_development_inputs_write_partial_artifacts_and_later_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            manifest_path = self.write_json(
                root / "attempt4-datasets-manifest.json",
                {
                    "validation_datasets": [
                        "mnli-validation_matched-attempt4-dev.tsv",
                        "anli-r1-dev-attempt4-dev.tsv",
                    ],
                    "test_datasets": ["xnli-en-test-attempt4-test.tsv"],
                    "stress_datasets": ["wanli-test-attempt4-stress-test.tsv"],
                },
            )
            validation_runtime_path = self.write_benchmark_csv(
                root / "validation-runtime.csv",
                [
                    self.runtime_row(candidate="reference"),
                    self.runtime_row(candidate="candidate_a"),
                ],
            )
            output_prefix = root / "attempt4-cpu-summary"

            incomplete_rows = [
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-dev.tsv",
                    quantization="reference",
                    role="fidelity_validation",
                    gold_accuracy=0.90,
                    float_label_agreement=1.0,
                    size_bytes=1000,
                ),
                self.summary_row(
                    dataset="mnli-validation_matched-attempt4-dev.tsv",
                    quantization="candidate_a",
                    role="fidelity_validation",
                    gold_accuracy=0.896,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
            ]
            validation_summary_path = self.write_json(root / "validation-summary.json", {"rows": incomplete_rows})

            result = replication_cpu_final.build_attempt4_cpu_report(
                manifest_path,
                validation_summary_path,
                validation_runtime_path,
                output_prefix,
            )

            self.assertTrue(result["partial"])
            partial_paths = result["paths"]
            self.assertTrue(partial_paths["candidate_json"].is_file())
            self.assertTrue(partial_paths["report_markdown"].is_file())
            self.assertFalse((root / "attempt4-cpu-summary.json").exists())
            self.assertEqual(result["locked_quantizations"], [])
            self.assertIsNone(result["recommendation"])

            partial_payload = json.loads(partial_paths["candidate_json"].read_text(encoding="utf-8"))
            self.assertEqual(partial_payload["locked_quantizations"], [])
            candidate_row = next(row for row in partial_payload["candidates"] if row["quantization"] == "candidate_a")
            self.assertFalse(candidate_row["validation_complete"])
            self.assertFalse(candidate_row["validation_gate_pass"])

            complete_rows = incomplete_rows + [
                self.summary_row(
                    dataset="anli-r1-dev-attempt4-dev.tsv",
                    quantization="reference",
                    role="fidelity_validation",
                    gold_accuracy=0.80,
                    float_label_agreement=1.0,
                    size_bytes=1000,
                ),
                self.summary_row(
                    dataset="anli-r1-dev-attempt4-dev.tsv",
                    quantization="candidate_a",
                    role="fidelity_validation",
                    gold_accuracy=0.791,
                    float_label_agreement=0.99,
                    size_bytes=400,
                ),
            ]
            self.write_json(validation_summary_path, {"rows": complete_rows})

            final_result = replication_cpu_final.build_attempt4_cpu_report(
                manifest_path,
                validation_summary_path,
                validation_runtime_path,
                output_prefix,
            )

            self.assertFalse(final_result["partial"])
            self.assertTrue((root / "attempt4-cpu-summary.json").is_file())
            self.assertFalse((root / "attempt4-cpu-summary-partial.json").exists())
            self.assertFalse((root / "attempt4-cpu-summary-partial.csv").exists())
            self.assertFalse((root / "attempt4-cpu-summary-partial.md").exists())

            final_payload = json.loads((root / "attempt4-cpu-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(final_payload["locked_quantizations"], ["candidate_a", "reference"])


if __name__ == "__main__":
    unittest.main()
