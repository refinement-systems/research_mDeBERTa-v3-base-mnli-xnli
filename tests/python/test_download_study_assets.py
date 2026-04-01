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


download_study_assets = load_module(
    REPO_ROOT / "tools/download-study-assets.py",
    "download_study_assets",
)


class DownloadStudyAssetsTest(unittest.TestCase):
    def test_download_models_invokes_existing_script_with_tokenizer_assets(self) -> None:
        scratchpad_root = pathlib.Path("/tmp/study-scratchpad")
        with mock.patch.object(download_study_assets, "run_command") as run_command:
            download_study_assets.download_models(
                scratchpad_root=scratchpad_root,
                force=True,
                with_hf_reference_weights=True,
            )

        command = run_command.call_args.args[0]
        self.assertEqual(command[0], str(REPO_ROOT / "tools/download-mdeberta-v3-base.sh"))
        self.assertIn("--tokenizer-assets", command)
        self.assertIn("--reference-weights", command)
        self.assertIn("--force", command)
        self.assertIn(str(scratchpad_root / "models" / "mdeberta"), command)

    def test_download_datasets_invokes_prepare_and_final_suite_downloads(self) -> None:
        scratchpad_root = pathlib.Path("/tmp/study-scratchpad")
        with mock.patch.object(download_study_assets, "run_command") as run_command:
            download_study_assets.download_datasets(scratchpad_root=scratchpad_root, force=False)

        self.assertEqual(run_command.call_count, 2)
        prepare_command = run_command.call_args_list[0].args[0]
        final_suite_command = run_command.call_args_list[1].args[0]

        self.assertIn("tools/prepare-attempt1-quantization-data.py", prepare_command[1])
        self.assertIn("--skip-fine-tune", prepare_command)
        self.assertIn("tools/download-nli-eval-slices.py", final_suite_command[1])
        self.assertIn("--mnli-per-label", final_suite_command)
        self.assertIn("--xnli-per-label", final_suite_command)

    def test_main_creates_scratchpad_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratchpad_root = pathlib.Path(tmp_dir) / "scratchpad"
            with mock.patch.object(download_study_assets, "download_models") as download_models:
                with mock.patch.object(download_study_assets, "download_datasets") as download_datasets:
                    argv = ["download-study-assets.py", "--scratchpad-root", str(scratchpad_root)]
                    with mock.patch("sys.argv", argv):
                        rc = download_study_assets.main()
            self.assertEqual(rc, 0)
            self.assertTrue(scratchpad_root.is_dir())
            download_models.assert_called_once()
            download_datasets.assert_called_once()


if __name__ == "__main__":
    unittest.main()
