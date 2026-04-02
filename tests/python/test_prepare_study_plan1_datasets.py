from __future__ import annotations

import importlib.util
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


prepare_plan1_datasets = load_module(
    REPO_ROOT / "tools/prepare-study-plan1-datasets.py",
    "prepare_study_plan1_datasets",
)


class PrepareStudyPlan1DatasetsTest(unittest.TestCase):
    def test_copy_frozen_datasets_copies_calibration_and_smoke_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir) / "datasets"
            prepare_plan1_datasets.copy_frozen_datasets(
                dataset_root,
                force=False,
                filenames=(
                    "mnli-train-calibration-64-per-label.tsv",
                    "hf-probe-set.tsv",
                ),
            )

            for dataset_name in (
                "mnli-train-calibration-64-per-label.tsv",
                "hf-probe-set.tsv",
            ):
                copied = dataset_root / dataset_name
                source = REPO_ROOT / "benchmarks" / "nli" / dataset_name
                self.assertTrue(copied.is_file())
                self.assertEqual(
                    copied.read_text(encoding="utf-8"),
                    source.read_text(encoding="utf-8"),
                )

    def test_generation_commands_use_plan1_tags_and_skips(self) -> None:
        scratchpad_root = pathlib.Path("/tmp/study-plan1")
        dataset_root = scratchpad_root / "datasets"
        args = mock.Mock(
            mnli_validation_per_label=128,
            xnli_validation_per_label=32,
            mnli_validation_skip_per_label=128,
            xnli_validation_skip_per_label=64,
            mnli_test_per_label=100,
            xnli_test_per_label=50,
            mnli_test_skip_per_label=200,
            xnli_test_skip_per_label=50,
            seed=0,
            page_size=100,
            api_base_url="https://datasets-server.huggingface.co",
            force=True,
        )
        languages = ["en", "zh"]

        with mock.patch.object(prepare_plan1_datasets, "run_command") as run_command:
            prepare_plan1_datasets.generate_validation_datasets(dataset_root, languages, args)
            prepare_plan1_datasets.generate_test_datasets(dataset_root, languages, args)

        self.assertEqual(run_command.call_count, 2)
        validation_command = run_command.call_args_list[0].args[0]
        test_command = run_command.call_args_list[1].args[0]

        self.assertIn("--name-tag", validation_command)
        self.assertIn("plan1-search-validation", validation_command)
        self.assertIn("--mnli-skip-per-label", validation_command)
        self.assertIn("128", validation_command)
        self.assertIn("--xnli-skip-per-label", validation_command)
        self.assertIn("64", validation_command)
        self.assertEqual(validation_command.count("--xnli-language"), 2)

        self.assertIn("--name-tag", test_command)
        self.assertIn("plan1-test", test_command)
        self.assertIn("--mnli-split", test_command)
        self.assertIn("validation_matched", test_command)
        self.assertIn("validation_mismatched", test_command)
        self.assertIn("--mnli-skip-per-label", test_command)
        self.assertIn("200", test_command)
        self.assertIn("--xnli-skip-per-label", test_command)
        self.assertIn("50", test_command)


if __name__ == "__main__":
    unittest.main()
