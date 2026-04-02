from __future__ import annotations

import pathlib
import tempfile
import unittest

import tools.study_catalog as study_catalog


class StudyCatalogTest(unittest.TestCase):
    def test_default_catalog_loads_expected_names(self) -> None:
        entries = study_catalog.load_catalog()
        names = [entry["name"] for entry in entries]
        self.assertEqual(
            names,
            [
                "reference",
                "model_quantized",
                "dynamic_qint8_default",
                "dynamic_qint8_per_channel",
                "nncf_accuracy_attention_only",
                "nncf_fidelity_attention_proj_only",
                "attention_only",
                "attention_proj_only",
            ],
        )

    def test_plan1_catalog_loads_expected_names(self) -> None:
        entries = study_catalog.load_catalog(
            pathlib.Path("research/attempt2_course-correction/study_quantization_catalog_plan1.json")
        )
        names = [entry["name"] for entry in entries]
        self.assertEqual(
            names,
            [
                "reference",
                "model_quantized",
                "nncf_accuracy_attention_only",
                "nncf_fidelity_attention_proj_only",
                "attention_only",
                "nncf_fidelity_attention_only_n128_drop0p005",
                "nncf_fidelity_attention_only_n128_drop0p002",
                "nncf_fidelity_attention_only_n300_drop0p005",
                "nncf_fidelity_attention_only_n300_drop0p002",
            ],
        )

    def test_duplicate_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "catalog.json"
            path.write_text(
                """
                [
                  {
                    "name": "dup",
                    "generator_program": "python3",
                    "generator_args_json": [],
                    "source_artifact_name": null,
                    "output_relpath": "a.onnx",
                    "calibration_role": null,
                    "validation_role": null,
                    "allowed_backends": ["cpu"],
                    "notes": ""
                  },
                  {
                    "name": "dup",
                    "generator_program": "python3",
                    "generator_args_json": [],
                    "source_artifact_name": null,
                    "output_relpath": "b.onnx",
                    "calibration_role": null,
                    "validation_role": null,
                    "allowed_backends": ["cpu"],
                    "notes": ""
                  }
                ]
                """,
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                study_catalog.load_catalog(path)


if __name__ == "__main__":
    unittest.main()
