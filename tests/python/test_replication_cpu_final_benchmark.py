from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replication_cpu_final = load_module(
    REPO_ROOT / "tools/replication_cpu_final.py",
    "replication_cpu_final_benchmark",
)


class ReplicationCpuFinalBenchmarkTest(unittest.TestCase):
    def write_probe_dataset(self, scratchpad_root: pathlib.Path) -> None:
        dataset_root = scratchpad_root / "datasets"
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "hf-core-probe.tsv").write_text(
            "\n".join(
                [
                    "benchmark\tid\tpremise\thypothesis",
                    "mnli-probe\tex-1\tpremise one\thypothesis one",
                    "xnli-probe\tex-2\tpremise two\thypothesis two",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def write_catalog(
        self,
        scratchpad_root: pathlib.Path,
        candidates: list[str],
    ) -> pathlib.Path:
        payload = []
        for candidate in candidates:
            artifact_relpath = f"candidates/bench/{candidate}.onnx"
            artifact_path = scratchpad_root / artifact_relpath
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(candidate.encode("utf-8") + b"-artifact")
            payload.append(
                {
                    "name": candidate,
                    "generator_program": "python3",
                    "generator_args_json": ["tools/stage-study-artifact.py"],
                    "source_artifact_name": None,
                    "output_relpath": artifact_relpath,
                    "calibration_role": None,
                    "validation_role": None,
                    "allowed_backends": ["cpu"],
                    "notes": candidate,
                }
            )

        catalog_path = scratchpad_root / "benchmark_catalog.json"
        catalog_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return catalog_path

    def init_workspace(self, root: pathlib.Path, candidates: list[str]) -> pathlib.Path:
        scratchpad_root = root / "workspace"
        self.write_probe_dataset(scratchpad_root)
        catalog_path = self.write_catalog(scratchpad_root, candidates)
        replication_cpu_final.initialize_study_workspace(scratchpad_root, catalog_path=catalog_path)
        return scratchpad_root

    def fake_summary(self, *, benchmark_names: list[str], load_ms: float, warm_runs: list[float]) -> dict[str, object]:
        per_benchmark = {}
        for benchmark_name in benchmark_names:
            per_benchmark[benchmark_name] = {
                "examples": 1,
                "warm_latency_ms": replication_cpu_final.summarize_numeric(warm_runs),
            }
        return {
            "examples": len(benchmark_names),
            "file_size_bytes": 1234,
            "load_ms": replication_cpu_final.summarize_numeric([load_ms]),
            "warm_latency_ms": replication_cpu_final.summarize_numeric(warm_runs * len(benchmark_names)),
            "resident_after_load_bytes": replication_cpu_final.summarize_numeric([1000.0]),
            "resident_after_warmup_bytes": replication_cpu_final.summarize_numeric([2000.0]),
            "resident_after_timed_runs_bytes": replication_cpu_final.summarize_numeric([3000.0]),
            "peak_rss_after_load_bytes": replication_cpu_final.summarize_numeric([4000.0]),
            "peak_rss_after_warmup_bytes": replication_cpu_final.summarize_numeric([5000.0]),
            "peak_rss_after_timed_runs_bytes": replication_cpu_final.summarize_numeric([6000.0]),
            "time_l_peak_rss_bytes": replication_cpu_final.summarize_numeric([]),
            "per_benchmark": per_benchmark,
            "per_example": [
                {
                    "benchmark": benchmark_name,
                    "id": f"{benchmark_name}-id",
                    "timing_mean_ms": replication_cpu_final.summarize_numeric(warm_runs)["mean"],
                    "timing_median_ms": replication_cpu_final.summarize_numeric(warm_runs)["median"],
                    "timing_p95_ms": replication_cpu_final.summarize_numeric(warm_runs)["p95"],
                    "timing_min_ms": replication_cpu_final.summarize_numeric(warm_runs)["min"],
                    "timing_max_ms": replication_cpu_final.summarize_numeric(warm_runs)["max"],
                }
                for benchmark_name in benchmark_names
            ],
        }

    def test_persistent_mode_writes_legacy_csv_schema_and_json_metric_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = self.init_workspace(pathlib.Path(tmp_dir), ["alpha", "beta"])
            benchmark_names = ["mnli-probe", "xnli-probe"]

            def worker_runner(request: dict[str, object]) -> dict[str, object]:
                self.assertEqual(request["mode"], "persistent")
                return self.fake_summary(benchmark_names=benchmark_names, load_ms=11.0, warm_runs=[1.0, 2.0, 3.0])

            output_prefix = scratchpad_root / "reports" / "attempt4-validation-cpu-persistent"
            replication_cpu_final.benchmark_runtime_phase(
                scratchpad_root,
                "persistent",
                ["alpha", "beta"],
                output_prefix,
                worker_runner=worker_runner,
            )

            with output_prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, replication_cpu_final.benchmark_csv_fieldnames())
                rows = list(reader)
                self.assertEqual(len(rows), 2)

            payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
            row = payload["results"][0]
            self.assertEqual(
                set(row.keys()),
                {
                    "candidate",
                    "backend",
                    "mode",
                    "artifact_path",
                    "artifact_sha256",
                    "examples",
                    "file_size_bytes",
                    "load_ms",
                    "warm_latency_ms",
                    "resident_after_load_bytes",
                    "resident_after_warmup_bytes",
                    "resident_after_timed_runs_bytes",
                    "peak_rss_after_load_bytes",
                    "peak_rss_after_warmup_bytes",
                    "peak_rss_after_timed_runs_bytes",
                    "time_l_peak_rss_bytes",
                    "per_benchmark",
                    "per_example",
                },
            )

    def test_coldstart_aggregation_matches_legacy_summary_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = self.init_workspace(pathlib.Path(tmp_dir), ["alpha"])
            worker_rows = iter(
                [
                    {
                        "benchmark": "mnli-probe",
                        "id": "ex-1",
                        "load_ms": 10.0,
                        "timing_mean_ms": 2.0,
                        "timing_median_ms": 2.0,
                        "timing_p95_ms": 3.0,
                        "timing_min_ms": 1.0,
                        "timing_max_ms": 3.0,
                        "timing_runs_ms": [1.0, 2.0, 3.0],
                        "resident_after_load_bytes": 100.0,
                        "resident_after_warmup_bytes": 200.0,
                        "resident_after_timed_runs_bytes": 300.0,
                        "peak_rss_after_load_bytes": 400.0,
                        "peak_rss_after_warmup_bytes": 500.0,
                        "peak_rss_after_timed_runs_bytes": 600.0,
                        "time_l_peak_rss_bytes": None,
                    },
                    {
                        "benchmark": "xnli-probe",
                        "id": "ex-2",
                        "load_ms": 20.0,
                        "timing_mean_ms": 5.0,
                        "timing_median_ms": 5.0,
                        "timing_p95_ms": 6.0,
                        "timing_min_ms": 4.0,
                        "timing_max_ms": 6.0,
                        "timing_runs_ms": [4.0, 5.0, 6.0],
                        "resident_after_load_bytes": 150.0,
                        "resident_after_warmup_bytes": 250.0,
                        "resident_after_timed_runs_bytes": 350.0,
                        "peak_rss_after_load_bytes": 450.0,
                        "peak_rss_after_warmup_bytes": 550.0,
                        "peak_rss_after_timed_runs_bytes": 650.0,
                        "time_l_peak_rss_bytes": None,
                    },
                ]
            )

            output_prefix = scratchpad_root / "reports" / "attempt4-test-cpu-cold"
            replication_cpu_final.benchmark_runtime_phase(
                scratchpad_root,
                "coldstart",
                ["alpha"],
                output_prefix,
                worker_runner=lambda request: next(worker_rows),
            )

            payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
            row = payload["results"][0]
            self.assertEqual(row["load_ms"]["mean"], 15.0)
            self.assertEqual(row["load_ms"]["median"], 15.0)
            self.assertEqual(row["warm_latency_ms"]["median"], 3.5)
            self.assertEqual(row["per_benchmark"]["mnli-probe"]["load_ms"]["median"], 10.0)
            self.assertEqual(row["per_benchmark"]["xnli-probe"]["warm_latency_ms"]["p95"], 6.0)
            self.assertEqual(len(row["per_example"]), 2)

    def test_resume_cache_sha_invalidation_partial_promotion_and_memory_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = self.init_workspace(pathlib.Path(tmp_dir), ["alpha", "beta"])
            output_prefix = scratchpad_root / "reports" / "attempt4-validation-cpu-persistent"
            calls: list[str] = []

            def flaky_runner(request: dict[str, object]) -> dict[str, object]:
                examples = list(request["examples"])
                benchmark_names = [str(item["benchmark"]) for item in examples]
                candidate = "alpha" if "alpha.onnx" in str(request["model_path"]) else "beta"
                calls.append(candidate)
                if candidate == "beta":
                    raise RuntimeError("interrupt")
                summary = self.fake_summary(benchmark_names=benchmark_names, load_ms=9.0, warm_runs=[1.0, 1.5, 2.0])
                summary["resident_after_load_bytes"] = replication_cpu_final.summarize_numeric([])
                summary["resident_after_warmup_bytes"] = replication_cpu_final.summarize_numeric([])
                summary["resident_after_timed_runs_bytes"] = replication_cpu_final.summarize_numeric([])
                summary["peak_rss_after_load_bytes"] = replication_cpu_final.summarize_numeric([])
                summary["peak_rss_after_warmup_bytes"] = replication_cpu_final.summarize_numeric([])
                summary["peak_rss_after_timed_runs_bytes"] = replication_cpu_final.summarize_numeric([])
                return summary

            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                replication_cpu_final.benchmark_runtime_phase(
                    scratchpad_root,
                    "persistent",
                    ["alpha", "beta"],
                    output_prefix,
                    worker_runner=flaky_runner,
                )

            partial_json = output_prefix.parent / f"{output_prefix.name}-partial.json"
            partial_csv = output_prefix.parent / f"{output_prefix.name}-partial.csv"
            self.assertTrue(partial_json.is_file())
            self.assertTrue(partial_csv.is_file())
            partial_payload = json.loads(partial_json.read_text(encoding="utf-8"))
            self.assertEqual([row["candidate"] for row in partial_payload["results"]], ["alpha"])

            calls.clear()

            def success_runner(request: dict[str, object]) -> dict[str, object]:
                examples = list(request["examples"])
                benchmark_names = [str(item["benchmark"]) for item in examples]
                candidate = "alpha" if "alpha.onnx" in str(request["model_path"]) else "beta"
                calls.append(candidate)
                return self.fake_summary(benchmark_names=benchmark_names, load_ms=12.0, warm_runs=[2.0, 2.5, 3.0])

            replication_cpu_final.benchmark_runtime_phase(
                scratchpad_root,
                "persistent",
                ["alpha", "beta"],
                output_prefix,
                worker_runner=success_runner,
            )
            self.assertEqual(calls, ["beta"])
            self.assertFalse(partial_json.exists())
            self.assertFalse(partial_csv.exists())

            rows = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))["results"]
            alpha_row = next(row for row in rows if row["candidate"] == "alpha")
            self.assertIsNone(alpha_row["resident_after_warmup_bytes"]["median"])

            alpha_artifact = scratchpad_root / "candidates" / "bench" / "alpha.onnx"
            alpha_artifact.write_bytes(b"alpha-artifact-updated")
            calls.clear()
            replication_cpu_final.benchmark_runtime_phase(
                scratchpad_root,
                "persistent",
                ["alpha", "beta"],
                output_prefix,
                worker_runner=success_runner,
            )
            self.assertEqual(calls, ["alpha"])

    def test_default_subprocess_runner_uses_python_self_invocation_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = self.init_workspace(pathlib.Path(tmp_dir), ["alpha"])
            output_prefix = scratchpad_root / "reports" / "attempt4-validation-cpu-persistent"
            seen_commands: list[list[str]] = []

            def fake_subprocess_run(command, **kwargs):
                seen_commands.append(list(command))
                self.assertEqual(command[0], sys.executable)
                self.assertTrue(str(command[1]).endswith("tools/replication_cpu_final.py"))
                self.assertNotIn("builddir/nli", " ".join(command))
                self.assertNotIn("builddir/nli-runtime-bench", " ".join(command))
                stdout = json.dumps(
                    self.fake_summary(
                        benchmark_names=["mnli-probe", "xnli-probe"],
                        load_ms=7.0,
                        warm_runs=[1.0, 2.0, 3.0],
                    )
                )
                return mock.Mock(returncode=0, stdout=stdout, stderr="")

            with mock.patch.object(replication_cpu_final.subprocess, "run", side_effect=fake_subprocess_run):
                replication_cpu_final.benchmark_runtime_phase(
                    scratchpad_root,
                    "persistent",
                    ["alpha"],
                    output_prefix,
                )

            self.assertEqual(len(seen_commands), 1)


if __name__ == "__main__":
    unittest.main()
