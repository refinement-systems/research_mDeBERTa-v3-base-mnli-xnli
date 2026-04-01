# Attempt 2 Plan 0

Scope: course correction after `attempt0` and `attempt1`.

This document replaces the earlier objective of jointly optimizing gold-label accuracy, runtime, memory footprint, and size. The goal of this run is narrower: build a reproducible, backend-separated view of quantized-model fidelity versus on-disk size.

## 1. Revised Goal

The working rules for this run are:

1. Freeze the source reference artifact and the reference evaluation behavior.
   - Source reference artifact: `models/mdeberta/onnx/model.onnx`
   - CPU reference behavior: float ONNX evaluated on CPU
   - CoreML reference behavior: float ONNX evaluated on CoreML
   - HF/PyTorch remains an audit reference, not the main ranking reference

2. Ignore process RSS and peak memory.
   - For this run, model size means serialized size on disk only.

3. Separate CPU and CoreML evaluation.
   - We are not looking for one universal winner.
   - The output may be one CPU recommendation, one CoreML recommendation, both, or neither.

4. Optimize for fidelity to the backend-specific float reference, not direct task accuracy.
   - The evaluation corpus is still built from the existing MNLI/XNLI rows.
   - Their labels are not part of the main selection objective.

5. Keep split discipline to prevent selection overfitting.
   - Calibration data fits the quantizer.
   - Fidelity-validation data chooses candidates.
   - Fidelity-test data reports the final result once.

6. Keep a tiny labeled smoke check, but do not use it for ranking.
   - This is only to catch obviously broken exports or wiring mistakes.
   - It must not be used to promote one candidate over another.

7. Skip performance benchmarking in this run.
   - No cold-start benchmarks.
   - No persistent-session benchmarks.
   - No memory instrumentation.

## 2. Current Position

The earlier attempts already imply that the new objective will reshuffle the shortlist.

Observed size tiers from the current artifact tree:

- float reference: `model.onnx` is about `1.116 GB`
- small dynamic baselines: `model_quantized.onnx`, `dynamic_qint8_default.onnx`, and `dynamic_qint8_per_channel.onnx` are about `338 MB`
- mid-size NNCF candidates: `nncf_accuracy_attention_only.onnx` and `nncf_fidelity_attention_proj_only.onnx` are about `488-509 MB`
- attempt0 dynamic finalists: `attention_only.onnx` and `attention_proj_only.onnx` are about `1.086-1.108 GB`
- broad static QDQ survivors are also about `1.067-1.089 GB`

That changes the reading of the old frontier:

- `attention_only` and `attention_proj_only` remain important fidelity anchors
- they are no longer plausible size winners
- the NNCF artifacts become much more interesting under the new objective
- broad static QDQ becomes even less attractive because it is both machine-hostile and almost float-sized

The current tree also contains stale or invalid artifacts:

- `models/mdeberta/onnx/candidates/dynamic_qint8_matmul.onnx` is currently zero bytes
- `models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx` is currently zero bytes

So this run must not trust file presence alone. Provenance, size, and artifact validity need to be recorded explicitly.

## 3. What Carries Forward

The initial rerun pool should be built only from completed, reproducible, non-destabilizing branches plus a few fidelity anchors.

Carry forward into the initial CPU lane:

- `float`
- `model_quantized`
- `dynamic_qint8_default`
- `dynamic_qint8_per_channel`
- `nncf_accuracy_attention_only`
- `nncf_fidelity_attention_proj_only`
- `attention_only`
- `attention_proj_only`

Carry forward into the initial CoreML lane:

- `float`
- `model_quantized`
- `dynamic_qint8_default`
- `nncf_accuracy_attention_only`
- `nncf_fidelity_attention_proj_only`
- `attention_only`
- `attention_proj_only`

Do not include in the initial run:

- broad static QDQ reruns
- `percentile` static calibration
- NNCF `accurate` bias correction
- QAT
- broad family sweeps beyond the already identified finalists

Reason:

- the static branch caused severe memory pressure in `attempt1`
- `accurate` bias correction did not complete reliably
- QAT did not produce a usable artifact
- most broad family variants are already dominated on size

## 4. Backend-Specific Rules

### 4.1. CPU lane

CPU is the natural home for the current ONNX quantization work.

For this run, the CPU lane should:

- evaluate the initial carry-forward pool listed above
- establish a first CPU Pareto frontier on size versus float fidelity
- identify whether any mid-size candidate displaces the float-sized fidelity anchors

CPU may get a second generation pass later, but only if the first frontier shows a real gap worth exploring.

### 4.2. CoreML lane

CoreML must be treated as a separate lane, not as a copy of the CPU search space.

For this run, the CoreML lane should:

- evaluate only the minimal carry-forward pool listed above
- avoid opening a fresh CoreML-targeted ONNX int8 search grid
- accept the possibility that there is no current quantized CoreML winner

If none of the carry-forward artifacts produces an acceptable size-versus-fidelity tradeoff on CoreML, the correct outcome is:

- no quantized CoreML recommendation from the current ONNX quantization family

That is a valid research result. It is better than forcing a bad fit.

## 5. Fidelity Definition

This run needs a fixed fidelity protocol with separate dataset roles.

Reference definition:

- source reference artifact: `models/mdeberta/onnx/model.onnx`
- CPU candidates are compared against float evaluated on CPU
- CoreML candidates are compared against float evaluated on CoreML
- HF/PyTorch may be used only as an audit trail when export-level drift needs to be checked

Dataset roles:

- calibration:
  use the existing attempt1 calibration slices for any method that still needs calibration data
- fidelity-validation:
  use the existing attempt1 search-validation slices to choose among candidates
- fidelity-test:
  keep the current multilingual full suite untouched until the candidate choice is locked
- smoke:
  use the core probe, hard probe, or another tiny labeled set only to catch obviously broken artifacts

Selection rule:

- static and NNCF methods may use calibration data only for fitting quantization parameters
- no candidate ranking decision is allowed to use the fidelity-test split
- the backend-specific frontier is chosen on fidelity-validation only
- after the candidate set is locked, run the final report once on fidelity-test
- if fidelity-test results trigger more tuning, that split is no longer a test split

Store raw logits for every example so the repo can compute more than one fidelity metric later.

Derived metrics to compute from the stored logits:

- float-label agreement
- mean absolute logit delta
- max absolute logit delta
- disagreement count and disagreement examples

Primary frontier metric for `plan0`:

- float-label agreement

Tie-break metric:

- lower mean absolute logit delta

If float-label agreement produces too many ties, the raw logits will allow the repo to switch to a stricter scalar later without rerunning the models.

The important constraint is that all candidate-selection metrics are computed on fidelity-validation, while fidelity-test is reserved for the final locked report.

## 6. Storage Layer

The project should add a SQLite-backed storage layer plus a stable filesystem layout for artifacts and logs.

Minimum schema:

### 6.1. `dataset`

- `id`
- `name`
- `role`
- `source_path`
- `source_sha256`

`role` should be one of:

- `calibration`
- `fidelity_validation`
- `fidelity_test`
- `smoke`

### 6.2. `dataset_row`

- `id`
- `dataset_id`
- `row_idx`
- `source_row_id`
- `label`
- `premise`
- `hypothesis`

Constraint:

- unique on `dataset_id, row_idx`

### 6.3. `quantization`

- `id`
- `name`
- `program`
- `args_json`
- `source_model_path`
- `source_model_sha256`
- `notes`

This table describes how to recreate an artifact. It does not store the artifact itself.

### 6.4. `artifact`

- `id`
- `quantization_id`
- `path`
- `artifact_sha256`
- `size_bytes`
- `status`
- `stdout_log_path`
- `stderr_log_path`

Possible `status` values:

- `materialized`
- `missing`
- `invalid`
- `failed`

### 6.5. `backend`

- `id`
- `name`

Pre-initialize:

- `CPU`
- `CoreML`

### 6.6. `evaluation_run`

- `id`
- `artifact_id`
- `backend_id`
- `dataset_id`
- `command_json`
- `status`
- `started_at`
- `finished_at`

Constraint:

- unique on `artifact_id, backend_id, dataset_id`

### 6.7. `evaluation`

- `id`
- `evaluation_run_id`
- `dataset_row_id`
- `entailment_logit`
- `neutral_logit`
- `contradiction_logit`
- `predicted_label`

Constraint:

- unique on `evaluation_run_id, dataset_row_id`

This corrects the earlier bug in the storage sketch: evaluation must be one row per dataset row x quantization artifact x backend.

## 7. Runner Behavior

The runner should be idempotent and resumable.

Required behavior:

1. Validate the artifact before evaluation.
   - file exists
   - file size is non-zero
   - hash matches the recorded artifact if already known

2. Ensure the backend-specific float reference run exists for the same backend and dataset.

3. Select only dataset rows missing evaluation rows for the requested artifact and backend.

4. Insert logits incrementally.
   - do not hold the whole dataset in memory waiting for one final write

5. Mark interrupted runs as resumable.
   - rerunning the same command should continue from the missing rows

6. Preserve logs and failure state.
   - a failed artifact must be recorded as failed, not silently retried forever

## 8. Concrete Work For This Run

The work for this run should be:

1. Freeze the source reference artifact, backend-specific float references, and dataset-role assignments.

2. Implement the SQLite schema and initialization tool.

3. Inventory the current artifact tree.
   - record path, size, hash, and validity
   - mark zero-byte or missing artifacts as invalid

4. Backfill quantization metadata for the selected carry-forward artifacts.
   - enough to recreate them from the existing tools

5. Run float reference evaluation on CPU and CoreML for:
   - fidelity-validation
   - fidelity-test
   - smoke assets when used

6. Run the selected carry-forward candidates on CPU and CoreML on fidelity-validation first.

7. Compute provisional backend-specific frontiers on fidelity-validation only.

8. Lock the candidate set for each backend.

9. Run the locked finalists on fidelity-test.

10. Write one short result note stating:
   - the CPU frontier
   - the CoreML frontier
   - whether either backend currently has a quantized recommendation
   - which datasets were used for calibration, fidelity-validation, fidelity-test, and smoke

## 9. What This Run Should Not Do

This run should not:

- reopen broad model-generation sweeps
- rerun the machine-hostile static grid
- use benchmark labels as the ranking objective
- use the fidelity-test split to choose winners
- benchmark runtime
- benchmark memory
- assume that a CPU-friendly model family is automatically a CoreML-friendly family

## 10. Bottom Line

The proposal is directionally right.

The most important changes are:

- treat size on disk as the only size metric
- freeze one source reference artifact but compare candidates to backend-specific float reference behavior
- treat CPU and CoreML as separate decision problems
- keep calibration, fidelity-validation, fidelity-test, and smoke datasets in separate roles
- store per-example logits in a resumable database
- let the new size-versus-fidelity objective decide the shortlist instead of carrying forward the old accuracy-oriented frontier by inertia

If this run is executed cleanly, the repo should end with a much clearer answer than `attempt0` and `attempt1` could provide:

- which existing quantized artifacts are still interesting when size actually matters
- whether CoreML has any current quantized winner at all
- and which future quantization branches are worth reopening, if any
