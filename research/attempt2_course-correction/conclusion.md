# Attempt 2 Course-Correction Conclusion

## Final winner for this attempt

### CPU lane winner
The winning quantized model for the `attempt2_course-correction` line of work is:

- `nncf_accuracy_attention_only`

This is the final CPU recommendation after the locked `plan1` untouched-test readout. In `result1.md`, it remains on the final test frontier at **465.1 MiB** with **98.741% float-label agreement**, and it dominates the fidelity-oriented plan1 challengers on the stated size-vs-fidelity objective.

### CoreML lane outcome
There is **no CoreML winner** for this attempt.

`result0.md` and `plan1.md` both document that CoreML was blocked by backend-validity issues (reference fallback behavior), and follow-up CoreML ranking was deferred. So the correct repo-state conclusion is a CPU winner plus a CoreML non-result.

---

## How the winning model was created (detailed)

This section reconstructs the creation path from the repository catalog and plans/results.

### 1) Base artifact and workflow framing
Attempt 2 explicitly froze a source float ONNX reference artifact and shifted objective to **backend-separated fidelity-vs-size**, not raw gold-label task metrics, runtime, or RSS.

- Source reference artifact: `models/mdeberta/onnx/model.onnx`
- Candidate selection metric family: fidelity to backend-specific float reference
- Main ranking discipline: choose on validation frontier, then test locked frontier only

This framing is laid out in `plan0.md` and reaffirmed in `result0.md` / `result1.md`.

### 2) Quantization recipe for the winner
The winning candidate is declared in the attempt2 catalog with explicit generator provenance:

- Name: `nncf_accuracy_attention_only`
- Program: `python3`
- Script: `tools/quantize-onnx-nncf.py`
- Key arguments:
  - `--mode=accuracy-control`
  - `--metric=gold_accuracy`
  - `--preset=mixed`
  - `--ignored-scope-family=attention_only`
  - `--subset-size=300`
  - `--max-drop=0.01`
  - `--preprocess`
  - `--fast-bias-correction`
  - `--calibration-tsv=${CALIBRATION_TSVS}`
  - `--validation-tsv=${VALIDATION_TSVS}`
- Source artifact name: `reference`
- Output path (relative): `candidates/attempt1/nncf_accuracy_attention_only.onnx`

The same candidate is carried into the plan1 catalog as the incumbent CPU winner and used as the benchmark to beat.

### 3) Candidate catalog + reproducibility mechanism
Attempt 2 relies on a structured catalog (`study_quantization_catalog*.json`) and a study runner with explicit materialization/evaluation state tracking.

- Catalog entries include generator program, arguments, source artifact lineage, output path, allowed backends, and role hooks for calibration/validation.
- `tools/study_catalog.py` validates the catalog schema (required keys, uniqueness, type checks).
- The C++ study workflow stores run information and evaluation outputs in SQLite, with artifact status states like `materialized`, `missing`, `invalid`, and `failed`.

This design is exactly why the winner can be identified as reproducible provenance instead of just a filename mention.

### 4) Why this model remained the winner after plan1
`plan1` ran a narrow objective-aligned NNCF follow-up around the same family (including `hf_agreement` variants) to try to displace the incumbent.

Result: none of the new plan1 challengers beat `nncf_accuracy_attention_only` on the final untouched test frontier under the attempt objective (size vs float-fidelity). `result1.md` explicitly concludes that `nncf_accuracy_attention_only` remains the best practical CPU recommendation.

---

## How the winner was tested

### 1) Split discipline used in attempt 2
The test protocol in attempt2 follows this sequence:

1. Quantization calibration on calibration-role datasets
2. Candidate selection on fidelity-validation datasets
3. Frontier lock on validation only
4. One-pass reporting on untouched fidelity-test datasets for locked points
5. Smoke datasets used only as diagnostics, not ranking

This is explicitly documented in `plan0.md`, applied in `result0.md`, and preserved in `plan1`/`result1` logic.

### 2) Metrics used
From result reports and summarization scripts, the key fidelity metrics are:

- float-label agreement (primary)
- mean absolute logit delta (tie-break / companion)
- max absolute logit delta
- disagreement count

The summary tooling (`tools/summarize-study-db.py`) computes these by comparing each candidate’s stored logits against the backend-specific `reference` rows in SQLite.

### 3) Dataset roles and test packs
Attempt2 documents role-specific dataset sets:

- `calibration`
- `fidelity_validation`
- `fidelity_test`
- `smoke`

`result0.md` lists concrete files used per role (MNLI and XNLI slices), and `plan1.md` describes refreshed split hygiene for follow-up (`*_plan1`) to avoid selection leakage after prior touching of earlier splits.

### 4) Final CPU evidence snapshot
From `result1.md` untouched test frontier:

- `nncf_accuracy_attention_only`: **465.1 MiB**, **98.741%** float-label agreement (frontier)
- `reference`: **1064.4 MiB**, **100.000%** float-label agreement (frontier ceiling)
- smaller `model_quantized`: **323.0 MiB**, **64.148%** agreement (too lossy)
- fidelity-oriented plan1 challengers did not displace the incumbent on final frontier status

This is the direct basis for naming the winner.

---

## Bottom line
Given only the current repository state (plans, results, catalog entries, and study tooling), the attempt2 conclusion is:

- **Winning model:** `nncf_accuracy_attention_only` (CPU)
- **Creation path:** NNCF `accuracy-control` quantization pipeline with `attention_only` ignored scope and recorded catalog provenance
- **Validation/testing path:** calibration → fidelity_validation frontier lock → untouched fidelity_test reporting, with SQLite-backed logit-level fidelity accounting
- **CoreML:** no valid winner yet in this attempt due to backend-validity/fallback blockers
