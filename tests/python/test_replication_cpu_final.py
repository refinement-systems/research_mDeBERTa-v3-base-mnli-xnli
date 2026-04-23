from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
