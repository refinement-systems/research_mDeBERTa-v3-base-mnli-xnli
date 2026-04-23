from __future__ import annotations

import importlib.util
import json
import pathlib
import sqlite3
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
    "replication_cpu_final",
)


class FakeDatasetsClient:
    def __init__(self) -> None:
        self.fetch_json_calls: list[tuple[str, str, str]] = []
        self.fetch_text_calls: list[str] = []

    def fetch_json(self, path: str, params: dict[str, object]) -> dict[str, object]:
        self.fetch_json_calls.append(
            (
                str(params["dataset"]),
                str(params["config"]),
                str(params["split"]),
            )
        )
        dataset = str(params["dataset"])
        config = str(params["config"])
        split = str(params["split"])
        offset = int(params["offset"])

        if offset > 0:
            return {"rows": [], "num_rows_total": 1}

        if dataset == "facebook/xnli" and config == "all_languages" and split == "test":
            return {
                "features": [
                    {
                        "name": "label",
                        "type": {"_type": "ClassLabel", "names": list(replication_cpu_final.TERNARY_LABELS)},
                    }
                ],
                "num_rows_total": 1,
                "rows": [
                    {
                        "row_idx": 7,
                        "row": {
                            "label": 1,
                            "premise": {
                                "language": list(replication_cpu_final.DEFAULT_XNLI_LANGUAGES),
                                "translation": [
                                    f"premise-{language}" for language in replication_cpu_final.DEFAULT_XNLI_LANGUAGES
                                ],
                            },
                            "hypothesis": {
                                "language": list(replication_cpu_final.DEFAULT_XNLI_LANGUAGES),
                                "translation": [
                                    f"hypothesis-{language}"
                                    for language in replication_cpu_final.DEFAULT_XNLI_LANGUAGES
                                ],
                            },
                        },
                    }
                ],
            }

        if dataset == "nyu-mll/multi_nli":
            return {
                "features": [
                    {
                        "name": "gold_label",
                        "type": {"_type": "ClassLabel", "names": list(replication_cpu_final.TERNARY_LABELS)},
                    }
                ],
                "num_rows_total": 1,
                "rows": [
                    {
                        "row_idx": 11,
                        "row": {
                            "sentence1": f"{split}-premise",
                            "sentence2": f"{split}-hypothesis",
                            "gold_label": 0,
                            "pairID": f"{split}-pair",
                        },
                    }
                ],
            }

        if dataset == replication_cpu_final.DEFAULT_ANLI_DATASET:
            return {
                "features": [],
                "num_rows_total": 1,
                "rows": [
                    {
                        "row_idx": 5,
                        "row": {
                            "premise": f"{split}-premise",
                            "hypothesis": f"{split}-hypothesis",
                            "label": "c",
                            "uid": f"{split}-uid",
                        },
                    }
                ],
            }

        if dataset == replication_cpu_final.DEFAULT_WANLI_DATASET:
            return {
                "features": [],
                "num_rows_total": 1,
                "rows": [
                    {
                        "row_idx": 3,
                        "row": {
                            "premise": "wanli-premise",
                            "hypothesis": "wanli-hypothesis",
                            "gold": "entailment",
                            "id": "wanli-id",
                        },
                    }
                ],
            }

        raise AssertionError(f"Unexpected fetch_json request: {path=} {params=}")

    def fetch_text_url(self, url: str) -> str:
        self.fetch_text_calls.append(url)
        return (
            "gold_label\tsentence1_binary_parse\tsentence2_binary_parse\tsentence1_parse\t"
            "sentence2_parse\tsentence1\tsentence2\tcaptionID\tpairID\theuristic\tsubcase\ttemplate\n"
            "entailment\t-\t-\t-\t-\thans-premise\thans-hypothesis\t0\t0\tlexical_overlap\t"
            "ln_subject/object_swap\ttemplate\n"
        )


class FakePredictor:
    def __init__(self, model_path: pathlib.Path, tokenizer_root: pathlib.Path, backend: str) -> None:
        self.model_path = model_path
        self.tokenizer_root = tokenizer_root
        self.backend = backend

    def predict_logits(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        if "Germany" in premise or "Germany" in hypothesis:
            return (0.5, 1.0, 3.0)
        if "Merkel" in premise:
            return (1.5, 2.5, 1.0)
        if self.model_path.name == "model.onnx":
            return (5.0, 1.0, 0.0)
        return (4.0, 1.0, 0.5)


class FailingPredictor:
    def __init__(self, model_path: pathlib.Path, tokenizer_root: pathlib.Path, backend: str) -> None:
        del model_path, tokenizer_root, backend

    def predict_logits(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        del premise, hypothesis
        raise RuntimeError("predictor exploded")


class ReplicationCpuFinalDatasetPrepTest(unittest.TestCase):
    def prepare(self, workspace: pathlib.Path, *, force: bool, client: FakeDatasetsClient) -> pathlib.Path:
        with mock.patch.object(replication_cpu_final, "verify_disjointness") as verify_disjointness:
            manifest_path = replication_cpu_final.prepare_attempt4_datasets(
                workspace,
                force,
                client=client,
            )
        self.last_verify_call = verify_disjointness.call_args
        return manifest_path

    def test_copy_frozen_datasets_and_compatibility_files_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = pathlib.Path(tmp_dir) / "workspace"
            manifest_path = self.prepare(workspace, force=False, client=FakeDatasetsClient())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            datasets_dir = workspace / "datasets"

            for dataset_name in (
                *replication_cpu_final.CALIBRATION_DATASET_FILENAMES,
                *replication_cpu_final.SMOKE_DATASET_FILENAMES,
                *replication_cpu_final.COMPATIBILITY_DATASET_FILENAMES,
            ):
                self.assertTrue((datasets_dir / dataset_name).is_file(), dataset_name)

            self.assertEqual(
                manifest["calibration_datasets"],
                list(replication_cpu_final.CALIBRATION_DATASET_FILENAMES),
            )
            self.assertEqual(
                manifest["smoke_datasets"],
                list(replication_cpu_final.SMOKE_DATASET_FILENAMES),
            )

    def test_default_xnli_languages_expand_to_expected_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = pathlib.Path(tmp_dir) / "workspace"
            manifest_path = self.prepare(workspace, force=False, client=FakeDatasetsClient())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            expected_xnli = [
                f"xnli-{language}-test-attempt4-test.tsv"
                for language in replication_cpu_final.DEFAULT_XNLI_LANGUAGES
            ]
            self.assertEqual(manifest["test_datasets"][: len(expected_xnli)], expected_xnli)

            sample_path = workspace / "datasets" / "xnli-ar-test-attempt4-test.tsv"
            lines = sample_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[1].split("\t")[0], "xnli-ar-test")
            self.assertIn("premise-ar", lines[1])
            self.assertIn("hypothesis-ar", lines[1])

    def test_manifest_preserves_top_level_arrays_and_exports_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = pathlib.Path(tmp_dir) / "workspace"
            manifest_path = self.prepare(workspace, force=False, client=FakeDatasetsClient())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(
                set(manifest.keys()),
                {
                    "scratchpad_root",
                    "calibration_datasets",
                    "smoke_datasets",
                    "validation_datasets",
                    "test_datasets",
                    "stress_datasets",
                    "exports",
                },
            )
            self.assertEqual(manifest["scratchpad_root"], str(workspace.resolve()))
            self.assertEqual(len(manifest["validation_datasets"]), 5)
            self.assertEqual(len(manifest["test_datasets"]), 18)
            self.assertEqual(len(manifest["stress_datasets"]), 2)

            representative = manifest["exports"]["validation"][0]
            self.assertEqual(representative["name"], "mnli-validation_matched-attempt4-dev.tsv")
            self.assertEqual(representative["dataset"], "nyu-mll/multi_nli")
            self.assertEqual(representative["config"], "default")
            self.assertEqual(representative["split"], "validation_matched")
            self.assertEqual(representative["benchmark"], "mnli-validation_matched")
            self.assertEqual(representative["label_kind"], "ternary")
            self.assertEqual(representative["row_count"], 1)

    def test_rerun_without_force_reuses_existing_exports_and_skips_fetches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = pathlib.Path(tmp_dir) / "workspace"
            self.prepare(workspace, force=False, client=FakeDatasetsClient())

            second_client = FakeDatasetsClient()
            manifest_path = self.prepare(workspace, force=False, client=second_client)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(second_client.fetch_json_calls, [])
            self.assertEqual(second_client.fetch_text_calls, [])
            self.assertTrue(all(item["skipped"] for item in manifest["exports"]["validation"]))
            self.assertTrue(all(item["skipped"] for item in manifest["exports"]["test"]))
            self.assertTrue(all(item["skipped"] for item in manifest["exports"]["stress"]))

    def test_rerun_with_force_rewrites_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = pathlib.Path(tmp_dir) / "workspace"
            self.prepare(workspace, force=False, client=FakeDatasetsClient())

            rewritten = workspace / "datasets" / "xnli-ar-test-attempt4-test.tsv"
            rewritten.write_text("premise\thypothesis\nstale\tstale\n", encoding="utf-8")

            second_client = FakeDatasetsClient()
            self.prepare(workspace, force=True, client=second_client)

            rewritten_text = rewritten.read_text(encoding="utf-8")
            self.assertIn("premise-ar", rewritten_text)
            self.assertIn(("facebook/xnli", "all_languages", "test"), second_client.fetch_json_calls)

    def test_disjointness_check_uses_the_31_non_smoke_attempt4_slices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = pathlib.Path(tmp_dir) / "workspace"
            self.prepare(workspace, force=False, client=FakeDatasetsClient())

            dataset_root_arg, dataset_names_arg = self.last_verify_call.args
            self.assertEqual(dataset_root_arg, (workspace / "datasets").resolve())
            self.assertEqual(len(dataset_names_arg), 31)
            self.assertEqual(
                dataset_names_arg[: len(replication_cpu_final.CALIBRATION_DATASET_FILENAMES)],
                list(replication_cpu_final.CALIBRATION_DATASET_FILENAMES),
            )
            self.assertNotIn("hf-probe-set.tsv", dataset_names_arg)
            self.assertNotIn("hf-core-probe.tsv", dataset_names_arg)
            self.assertNotIn("mnli-train-search-validation-skip64-64-per-label.tsv", dataset_names_arg)

    def test_compatibility_tsvs_are_excluded_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = pathlib.Path(tmp_dir) / "workspace"
            manifest_path = self.prepare(workspace, force=False, client=FakeDatasetsClient())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            manifest_dataset_names = set(
                manifest["calibration_datasets"]
                + manifest["smoke_datasets"]
                + manifest["validation_datasets"]
                + manifest["test_datasets"]
                + manifest["stress_datasets"]
            )
            for dataset_name in replication_cpu_final.COMPATIBILITY_DATASET_FILENAMES:
                self.assertNotIn(dataset_name, manifest_dataset_names)


class ReplicationCpuFinalStudyWorkflowTest(unittest.TestCase):
    def write_fixture_dataset(self, scratchpad_root: pathlib.Path, dataset_name: str) -> None:
        dataset_root = scratchpad_root / "datasets"
        dataset_root.mkdir(parents=True, exist_ok=True)
        shutil_source = REPO_ROOT / "tests" / "data" / "nli_eval_fixture.tsv"
        (dataset_root / dataset_name).write_text(shutil_source.read_text(encoding="utf-8"), encoding="utf-8")

    def connect(self, scratchpad_root: pathlib.Path) -> sqlite3.Connection:
        connection = sqlite3.connect(str(scratchpad_root / "db.sqlite3"))
        connection.execute("PRAGMA foreign_keys=ON;")
        return connection

    def predictor_factory(self, model_path: pathlib.Path, tokenizer_root: pathlib.Path, backend: str) -> FakePredictor:
        return FakePredictor(model_path, tokenizer_root, backend)

    def test_catalog_validation_rejects_duplicate_names_missing_keys_and_bad_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            valid = {
                "name": "reference",
                "generator_program": "python3",
                "generator_args_json": ["tools/stage-study-artifact.py", "--src=models/mdeberta/onnx/model.onnx", "--dest=${DEST}"],
                "source_artifact_name": None,
                "output_relpath": "models/mdeberta/onnx/model.onnx",
                "calibration_role": None,
                "validation_role": None,
                "allowed_backends": ["cpu"],
                "notes": "ok",
            }

            duplicate_path = root / "duplicate.json"
            duplicate_path.write_text(json.dumps([valid, dict(valid)]), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "must be unique"):
                replication_cpu_final.load_study_catalog(duplicate_path)

            missing_path = root / "missing.json"
            missing_payload = dict(valid)
            del missing_payload["notes"]
            missing_path.write_text(json.dumps([missing_payload]), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing keys"):
                replication_cpu_final.load_study_catalog(missing_path)

            bad_args_path = root / "bad_args.json"
            bad_args_payload = dict(valid)
            bad_args_payload["generator_args_json"] = ["ok", 1]
            bad_args_path.write_text(json.dumps([bad_args_payload]), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "generator_args_json"):
                replication_cpu_final.load_study_catalog(bad_args_path)

            bad_backend_path = root / "bad_backend.json"
            bad_backend_payload = dict(valid)
            bad_backend_payload["allowed_backends"] = ["cpu", 1]
            bad_backend_path.write_text(json.dumps([bad_backend_payload]), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "allowed_backends"):
                replication_cpu_final.load_study_catalog(bad_backend_path)

    def test_clean_workspace_init_creates_schema_and_preserves_attempt_tagged_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = pathlib.Path(tmp_dir) / "workspace"
            for dataset_name in (
                "mnli-train-calibration-64-per-label.tsv",
                "mnli-train-attempt4-dev-skip64-64-per-label.tsv",
                "xnli-en-validation-attempt4-dev-32-per-label.tsv",
                "mnli-validation_matched-attempt4-test.tsv",
                "xnli-en-test-attempt4-test.tsv",
                "hf-core-probe.tsv",
            ):
                self.write_fixture_dataset(scratchpad_root, dataset_name)

            replication_cpu_final.initialize_study_workspace(scratchpad_root)
            replication_cpu_final.initialize_study_workspace(scratchpad_root)

            connection = self.connect(scratchpad_root)
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                self.assertTrue(
                    {
                        "dataset",
                        "dataset_row",
                        "quantization",
                        "artifact",
                        "backend",
                        "evaluation_run",
                        "evaluation",
                    }.issubset(tables)
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM backend").fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM dataset").fetchone()[0],
                    6,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM dataset_row").fetchone()[0],
                    18,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM dataset WHERE role = 'calibration'"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM dataset WHERE role = 'fidelity_validation'"
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM dataset WHERE role = 'fidelity_test'"
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM dataset WHERE role = 'smoke'").fetchone()[0],
                    1,
                )
            finally:
                connection.close()

    def test_artifact_staging_records_state_and_regenerates_zero_byte_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = pathlib.Path(tmp_dir) / "workspace"
            self.write_fixture_dataset(scratchpad_root, "mnli-train-search-validation-skip64-64-per-label.tsv")
            replication_cpu_final.initialize_study_workspace(scratchpad_root)

            run_id = replication_cpu_final.run_study_evaluation(
                scratchpad_root,
                "dynamic_qint8_default",
                "mnli-train-search-validation-skip64-64-per-label.tsv",
                predictor_factory=self.predictor_factory,
            )
            self.assertGreater(run_id, 0)

            candidate_path = scratchpad_root / "candidates" / "ort" / "dynamic_qint8_default.onnx"
            self.assertTrue(candidate_path.is_file())
            connection = self.connect(scratchpad_root)
            try:
                artifact_row = connection.execute(
                    """
                    SELECT status, size_bytes, artifact_sha256, stdout_log_path
                    FROM artifact
                    WHERE path = ?
                    """,
                    (str(candidate_path.resolve()),),
                ).fetchone()
                self.assertEqual(artifact_row[0], "materialized")
                self.assertGreater(int(artifact_row[1]), 0)
                self.assertTrue(str(artifact_row[2]))
                stdout_log_path = pathlib.Path(str(artifact_row[3]))
                payload = json.loads(stdout_log_path.read_text(encoding="utf-8").strip())
                self.assertEqual(payload["dest"], str(candidate_path.resolve()))
            finally:
                connection.close()

            candidate_path.write_bytes(b"")
            replication_cpu_final.run_study_evaluation(
                scratchpad_root,
                "dynamic_qint8_default",
                "mnli-train-search-validation-skip64-64-per-label.tsv",
                predictor_factory=self.predictor_factory,
            )

            connection = self.connect(scratchpad_root)
            try:
                self.assertEqual(
                    connection.execute(
                        "SELECT status FROM artifact WHERE path = ?",
                        (str(candidate_path.resolve()),),
                    ).fetchone()[0],
                    "materialized",
                )
            finally:
                connection.close()

    def test_cpu_run_creates_reference_first_and_stores_logits_for_each_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = pathlib.Path(tmp_dir) / "workspace"
            dataset_name = "mnli-train-search-validation-skip64-64-per-label.tsv"
            self.write_fixture_dataset(scratchpad_root, dataset_name)
            replication_cpu_final.initialize_study_workspace(scratchpad_root)

            replication_cpu_final.run_study_evaluation(
                scratchpad_root,
                "dynamic_qint8_default",
                dataset_name,
                predictor_factory=self.predictor_factory,
            )

            connection = self.connect(scratchpad_root)
            try:
                runs = connection.execute(
                    """
                    SELECT q.name, er.status
                    FROM evaluation_run er
                    JOIN artifact a ON a.id = er.artifact_id
                    JOIN quantization q ON q.id = a.quantization_id
                    ORDER BY q.name
                    """
                ).fetchall()
                self.assertEqual(runs, [("dynamic_qint8_default", "completed"), ("reference", "completed")])
                eval_rows = connection.execute(
                    """
                    SELECT predicted_label, entailment_logit, neutral_logit, contradiction_logit
                    FROM evaluation
                    ORDER BY id
                    """
                ).fetchall()
                self.assertEqual(len(eval_rows), 6)
                self.assertTrue(all(row[0] in replication_cpu_final.TERNARY_LABELS for row in eval_rows))
                self.assertTrue(all(isinstance(float(row[1]), float) for row in eval_rows))
            finally:
                connection.close()

    def test_rerun_fills_only_missing_evaluation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = pathlib.Path(tmp_dir) / "workspace"
            dataset_name = "mnli-train-search-validation-skip64-64-per-label.tsv"
            self.write_fixture_dataset(scratchpad_root, dataset_name)
            replication_cpu_final.initialize_study_workspace(scratchpad_root)

            replication_cpu_final.run_study_evaluation(
                scratchpad_root,
                "dynamic_qint8_default",
                dataset_name,
                predictor_factory=self.predictor_factory,
            )

            connection = self.connect(scratchpad_root)
            try:
                connection.execute(
                    "DELETE FROM evaluation WHERE id = (SELECT MAX(id) FROM evaluation)"
                )
                connection.commit()
                count_before = connection.execute("SELECT COUNT(*) FROM evaluation").fetchone()[0]
                self.assertEqual(count_before, 5)
            finally:
                connection.close()

            replication_cpu_final.run_study_evaluation(
                scratchpad_root,
                "dynamic_qint8_default",
                dataset_name,
                predictor_factory=self.predictor_factory,
            )

            connection = self.connect(scratchpad_root)
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM evaluation").fetchone()[0], 6)
            finally:
                connection.close()

    def test_predictor_failure_marks_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = pathlib.Path(tmp_dir) / "workspace"
            dataset_name = "mnli-train-search-validation-skip64-64-per-label.tsv"
            self.write_fixture_dataset(scratchpad_root, dataset_name)
            replication_cpu_final.initialize_study_workspace(scratchpad_root)

            with self.assertRaisesRegex(RuntimeError, "predictor exploded"):
                replication_cpu_final.run_study_evaluation(
                    scratchpad_root,
                    "reference",
                    dataset_name,
                    predictor_factory=lambda model_path, tokenizer_root, backend: FailingPredictor(
                        model_path,
                        tokenizer_root,
                        backend,
                    ),
                )

            connection = self.connect(scratchpad_root)
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM evaluation").fetchone()[0], 0)
                self.assertEqual(
                    connection.execute("SELECT status FROM evaluation_run LIMIT 1").fetchone()[0],
                    "failed",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM evaluation_run WHERE status = 'completed'"
                    ).fetchone()[0],
                    0,
                )
            finally:
                connection.close()

    def test_disallowed_and_non_cpu_backends_fail_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = pathlib.Path(tmp_dir) / "workspace"
            dataset_name = "mnli-train-search-validation-skip64-64-per-label.tsv"
            self.write_fixture_dataset(scratchpad_root, dataset_name)
            replication_cpu_final.initialize_study_workspace(scratchpad_root)

            with self.assertRaisesRegex(RuntimeError, "not allowed on backend 'coreml'"):
                replication_cpu_final.run_study_evaluation(
                    scratchpad_root,
                    "dynamic_qint8_default",
                    dataset_name,
                    backend="coreml",
                    predictor_factory=self.predictor_factory,
                )

            candidate_path = scratchpad_root / "candidates" / "ort" / "dynamic_qint8_default.onnx"
            self.assertFalse(candidate_path.exists())

            custom_catalog = scratchpad_root / "custom_catalog.json"
            custom_catalog.write_text(
                json.dumps(
                    [
                        {
                            "name": "reference",
                            "generator_program": "python3",
                            "generator_args_json": [
                                "tools/stage-study-artifact.py",
                                "--src=models/mdeberta/onnx/model.onnx",
                                "--dest=${DEST}",
                            ],
                            "source_artifact_name": None,
                            "output_relpath": "models/mdeberta/onnx/model.onnx",
                            "calibration_role": None,
                            "validation_role": None,
                            "allowed_backends": ["cpu", "coreml"],
                            "notes": "reference",
                        }
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            other_root = pathlib.Path(tmp_dir) / "workspace_coreml"
            self.write_fixture_dataset(other_root, dataset_name)
            replication_cpu_final.initialize_study_workspace(other_root, catalog_path=custom_catalog)

            with self.assertRaisesRegex(RuntimeError, "Only CPU execution is implemented"):
                replication_cpu_final.run_study_evaluation(
                    other_root,
                    "reference",
                    dataset_name,
                    backend="coreml",
                    predictor_factory=self.predictor_factory,
                )
            self.assertFalse((other_root / "models" / "mdeberta" / "onnx" / "model.onnx").exists())

    def test_study_path_does_not_shell_out_to_cpp_study_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = pathlib.Path(tmp_dir) / "workspace"
            dataset_name = "mnli-train-search-validation-skip64-64-per-label.tsv"
            self.write_fixture_dataset(scratchpad_root, dataset_name)

            with mock.patch.object(replication_cpu_final.subprocess, "run", side_effect=AssertionError("no subprocess")):
                replication_cpu_final.initialize_study_workspace(scratchpad_root)
                replication_cpu_final.run_study_evaluation(
                    scratchpad_root,
                    "dynamic_qint8_default",
                    dataset_name,
                    predictor_factory=self.predictor_factory,
                )


if __name__ == "__main__":
    unittest.main()
