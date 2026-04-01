from __future__ import annotations

import importlib.util
import pathlib
import sqlite3
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


summarize_study_db = load_module(
    REPO_ROOT / "tools/summarize-study-db.py",
    "summarize_study_db",
)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE dataset (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL
        );
        CREATE TABLE dataset_row (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER NOT NULL,
            row_idx INTEGER NOT NULL,
            source_row_id TEXT,
            label TEXT,
            premise TEXT NOT NULL,
            hypothesis TEXT NOT NULL
        );
        CREATE TABLE quantization (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE artifact (
            id INTEGER PRIMARY KEY,
            quantization_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL
        );
        CREATE TABLE backend (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE evaluation_run (
            id INTEGER PRIMARY KEY,
            artifact_id INTEGER NOT NULL,
            backend_id INTEGER NOT NULL,
            dataset_id INTEGER NOT NULL
        );
        CREATE TABLE evaluation (
            id INTEGER PRIMARY KEY,
            evaluation_run_id INTEGER NOT NULL,
            dataset_row_id INTEGER NOT NULL,
            entailment_logit REAL NOT NULL,
            neutral_logit REAL NOT NULL,
            contradiction_logit REAL NOT NULL,
            predicted_label TEXT NOT NULL
        );
        """
    )


class SummarizeStudyDbTest(unittest.TestCase):
    def test_summarize_dataset_backend_computes_expected_metrics(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        create_schema(conn)

        conn.execute(
            "INSERT INTO dataset (id, name, role, source_path, source_sha256) VALUES (1, ?, ?, ?, ?)",
            ("mnli-train-search-validation-skip64-64-per-label.tsv", "fidelity_validation", "/tmp/d.tsv", "sha"),
        )
        for row_id, label in [(1, "entailment"), (2, "contradiction")]:
            conn.execute(
                """
                INSERT INTO dataset_row
                    (id, dataset_id, row_idx, source_row_id, label, premise, hypothesis)
                VALUES (?, 1, ?, ?, ?, ?, ?)
                """,
                (row_id, row_id - 1, f"row-{row_id}", label, f"premise-{row_id}", f"hypothesis-{row_id}"),
            )

        conn.execute("INSERT INTO quantization (id, name) VALUES (1, 'reference')")
        conn.execute("INSERT INTO quantization (id, name) VALUES (2, 'candidate')")
        conn.execute(
            "INSERT INTO artifact (id, quantization_id, path, size_bytes) VALUES (1, 1, ?, 1000)",
            ("/tmp/reference.onnx",),
        )
        conn.execute(
            "INSERT INTO artifact (id, quantization_id, path, size_bytes) VALUES (2, 2, ?, 500)",
            ("/tmp/candidate.onnx",),
        )
        conn.execute("INSERT INTO backend (id, name) VALUES (1, 'CPU')")
        conn.execute("INSERT INTO evaluation_run (id, artifact_id, backend_id, dataset_id) VALUES (1, 1, 1, 1)")
        conn.execute("INSERT INTO evaluation_run (id, artifact_id, backend_id, dataset_id) VALUES (2, 2, 1, 1)")

        conn.execute(
            """
            INSERT INTO evaluation
                (evaluation_run_id, dataset_row_id, entailment_logit, neutral_logit, contradiction_logit, predicted_label)
            VALUES (1, 1, 3.0, 1.0, 0.5, 'entailment')
            """
        )
        conn.execute(
            """
            INSERT INTO evaluation
                (evaluation_run_id, dataset_row_id, entailment_logit, neutral_logit, contradiction_logit, predicted_label)
            VALUES (1, 2, 0.5, 1.0, 3.0, 'contradiction')
            """
        )
        conn.execute(
            """
            INSERT INTO evaluation
                (evaluation_run_id, dataset_row_id, entailment_logit, neutral_logit, contradiction_logit, predicted_label)
            VALUES (2, 1, 2.5, 1.1, 0.5, 'entailment')
            """
        )
        conn.execute(
            """
            INSERT INTO evaluation
                (evaluation_run_id, dataset_row_id, entailment_logit, neutral_logit, contradiction_logit, predicted_label)
            VALUES (2, 2, 0.7, 1.1, 2.0, 'neutral')
            """
        )

        rows = summarize_study_db.summarize_dataset_backend(
            conn,
            dataset_name="mnli-train-search-validation-skip64-64-per-label.tsv",
            dataset_role="fidelity_validation",
            dataset_id=1,
            backend_name="cpu",
        )
        self.assertEqual(len(rows), 2)
        reference = next(row for row in rows if row.quantization == "reference")
        candidate = next(row for row in rows if row.quantization == "candidate")

        self.assertEqual(reference.float_label_agreement, 1.0)
        self.assertEqual(reference.mean_abs_logit_delta, 0.0)
        self.assertEqual(reference.disagreement_count, 0)

        self.assertAlmostEqual(candidate.float_label_agreement, 0.5)
        self.assertAlmostEqual(candidate.mean_abs_logit_delta, 0.31666666666666665)
        self.assertAlmostEqual(candidate.max_abs_logit_delta, 1.0)
        self.assertEqual(candidate.disagreement_count, 1)

    def test_compute_frontier_flags_prefers_smaller_and_more_faithful_rows(self) -> None:
        rows = [
            summarize_study_db.SummaryRow(
                dataset="d",
                role="fidelity_validation",
                backend="cpu",
                quantization="a",
                artifact_path="/tmp/a",
                size_bytes=1000,
                example_count=2,
                float_label_agreement=0.9,
                mean_abs_logit_delta=0.1,
                max_abs_logit_delta=0.2,
                disagreement_count=0,
                pareto_frontier=False,
            ),
            summarize_study_db.SummaryRow(
                dataset="d",
                role="fidelity_validation",
                backend="cpu",
                quantization="b",
                artifact_path="/tmp/b",
                size_bytes=500,
                example_count=2,
                float_label_agreement=0.95,
                mean_abs_logit_delta=0.2,
                max_abs_logit_delta=0.3,
                disagreement_count=0,
                pareto_frontier=False,
            ),
        ]
        flags = summarize_study_db.compute_frontier_flags(rows)
        self.assertFalse(flags[("d", "fidelity_validation", "cpu", "a")])
        self.assertTrue(flags[("d", "fidelity_validation", "cpu", "b")])


if __name__ == "__main__":
    unittest.main()
