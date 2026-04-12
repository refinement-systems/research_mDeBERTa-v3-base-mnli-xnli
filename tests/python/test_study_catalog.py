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

    def test_attempt3_catalog_enables_fp16_dependency_bootstrap(self) -> None:
        entries = study_catalog.load_catalog(
            pathlib.Path("research/attempt3_coreml/study_quantization_catalog.json")
        )
        by_name = {entry["name"]: entry for entry in entries}
        self.assertEqual(
            [entry["name"] for entry in entries],
            [
                "reference",
                "reference_fp16",
                "model_quantized",
                "dynamic_qint8_default",
            ],
        )
        self.assertIn(
            "--install-deps",
            by_name["reference_fp16"]["generator_args_json"],
        )

    def test_attempt4_catalog_loads_bounded_cpu_names(self) -> None:
        entries = study_catalog.load_catalog(
            pathlib.Path("research/attempt4_cpu-focus/study_quantization_catalog.json")
        )
        names = [entry["name"] for entry in entries]
        self.assertEqual(
            names,
            [
                "reference",
                "model_quantized",
                "dynamic_qint8_default",
                "dynamic_qint8_per_channel",
                "attention_only",
                "nncf_accuracy_attention_only",
                "nncf_fidelity_attention_proj_only",
                "nncf_fidelity_attention_only_n128_drop0p005",
                "static_attention_only_u8u8_minmax_n128",
                "static_attention_proj_only_s8s8_minmax_n300",
                "static_attention_proj_only_u8s8_rr_minmax_n128",
            ],
        )

    def test_attempt5_catalog_loads_expected_coreml_names(self) -> None:
        entries = study_catalog.load_catalog(
            pathlib.Path("research/attempt5_coreml-focus/study_quantization_catalog.json")
        )
        names = [entry["name"] for entry in entries]
        self.assertEqual(
            names,
            [
                "reference",
                "reference_fp16",
                "nncf_accuracy_attention_only",
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
