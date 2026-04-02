# Attempt 3 CoreML Plan 0

Scope: first CoreML-specific follow-up after the CPU conclusion in `attempt2_course-correction/result1.md`.

The working assumption for this attempt is that the earlier CoreML failure may have been environmental or sandbox-related, not necessarily a model-level failure. Even so, this attempt should not reopen CoreML as an "int8-first" search lane. On ONNX Runtime's CoreML execution provider, the default target should be `fp16`, with `fp32` kept as the numerical reference and fallback.

This machine is an Apple `M1`, and there is no access to newer Apple hardware for this attempt. That constraint matters. It means the plan should not spend time on `int4`, per-block compression, or Neural-Engine-specific `W8A8` claims that are mainly interesting on newer chips and newer OS releases.

## 1. Revised Goal

The goal of this run is:

1. verify that CoreML EP can be exercised outside the sandbox without CPU fallback
2. establish a trustworthy `fp32` CoreML reference
3. test `fp16` as the primary CoreML candidate
4. treat integer ONNX candidates only as secondary, measurement-driven controls

This attempt is still about:

- backend-specific fidelity to a backend-specific float reference
- size on disk, not RSS
- no performance benchmarking yet

## 2. CoreML Target Order

For this backend, the candidate order should be:

1. `reference`
   - the current float ONNX export
   - used as the CoreML fidelity reference and numerical safety anchor

2. `reference_fp16`
   - a float16-converted ONNX copy of the same float export
   - this is the default target for the attempt
   - keep model I/O types compatible with the current runner, so the conversion should primarily reduce internal weights and float compute rather than rewrite the text-model interface

3. integer controls, only after the float path is valid
   - `model_quantized`
   - `dynamic_qint8_default`

The integer controls are not the mainline target. They are there only to answer a narrow question:

- after the ONNX-to-CoreML path is working, do cheap ONNX int8 candidates actually survive that path well enough to beat the `fp16` baseline on this `M1` machine?

## 3. Explicit Non-Goals

Do not make this attempt about:

- `int4`
- per-block compression
- Neural-Engine-heavy activation-quantization search
- full `W8A8`
- broad NNCF reruns
- CPU candidate transfer testing as the mainline story
- runtime benchmarking

Reason:

- the repo does not yet have a Core ML native compression path
- ONNX Runtime CoreML EP is still the active execution path here
- the available hardware is `M1`, not `A17 Pro`, `M4`, or another newer target where the more aggressive integer story is more plausible

## 4. Dataset Discipline

This attempt should use a fresh CoreML-specific dataset pack under `scratchpad/attempt3_coreml`.

Dataset roles:

- `calibration`
  - reuse the existing frozen calibration slices only for integer controls that still need them
- `smoke`
  - reuse:
    - `hf-probe-set.tsv`
    - `hf-core-probe.tsv`
- `fidelity_validation`
  - generate fresh disjoint windows:
    - `mnli-train-attempt3-coreml-validation-skip256-64-per-label.tsv`
    - `xnli-de-validation-attempt3-coreml-validation-skip96-32-per-label.tsv`
    - `xnli-en-validation-attempt3-coreml-validation-skip96-32-per-label.tsv`
    - `xnli-es-validation-attempt3-coreml-validation-skip96-32-per-label.tsv`
    - `xnli-fr-validation-attempt3-coreml-validation-skip96-32-per-label.tsv`
    - `xnli-zh-validation-attempt3-coreml-validation-skip96-32-per-label.tsv`
- `fidelity_test`
  - generate fresh disjoint windows:
    - `mnli-validation_matched-attempt3-coreml-test-skip300-50-per-label.tsv`
    - `mnli-validation_mismatched-attempt3-coreml-test-skip300-50-per-label.tsv`
    - `xnli-de-test-attempt3-coreml-test-skip100-50-per-label.tsv`
    - `xnli-en-test-attempt3-coreml-test-skip100-50-per-label.tsv`
    - `xnli-es-test-attempt3-coreml-test-skip100-50-per-label.tsv`
    - `xnli-fr-test-attempt3-coreml-test-skip100-50-per-label.tsv`
    - `xnli-zh-test-attempt3-coreml-test-skip100-50-per-label.tsv`

The plan is intentionally smaller than the CPU `plan1` pack because this attempt is still partly infrastructural.

## 5. Execution Order

### 5.1. Gate the backend first

Run `reference` on CoreML for:

- both smoke datasets
- every `fidelity_validation` dataset

If explicit `coreml` selection falls back to CPU again outside the sandbox, stop the attempt. The right conclusion would then be:

- the current ORT CoreML lane is still not trustworthy on this machine or in this configuration

### 5.2. Run the primary CoreML target

If `reference` is valid:

- materialize `reference_fp16`
- run it on smoke
- run it on `fidelity_validation`

This is the main decision point for the attempt.

### 5.3. Only then run integer controls

If both `reference` and `reference_fp16` are valid, optionally run:

- `model_quantized`
- `dynamic_qint8_default`

on:

- smoke
- `fidelity_validation`

These are not promotion targets by default. They are controls.

### 5.4. Lock the CoreML validation frontier

Summarize CoreML validation and compute the Pareto frontier on:

- size on disk
- float-label agreement versus CoreML `reference`

Tie-break:

- lower mean absolute logit delta

### 5.5. Run untouched test only on locked points

Only the locked validation frontier points plus `reference` may run on:

- the `fidelity_test` pack

If `reference_fp16` is already the only serious non-reference point, keep the test sweep narrow.

## 6. Decision Rules

The working rules for interpretation are:

1. `fp16` is the default CoreML target.
   - If `reference_fp16` stays close to `reference` on fidelity while materially reducing size on disk, that is the main success condition.

2. `fp32` is the safety anchor.
   - Use it to validate the CoreML lane and to define fidelity.

3. integer controls are secondary.
   - Promote one only if it materially beats the `fp16` result on the stated objective.
   - If they fail, fall back, or lose on fidelity-versus-size, that is not a failure of the attempt.

4. on this `M1`, absence of an integer win is expected.
   - That result would be consistent with the current engineering prior.

5. if `fp16` still fails or is unexpectedly poor, the next step should be a Core ML native path, not a broader ONNX int8 sweep.
   - that follow-up would likely mean explicit `MLProgram` conversion and then a separate CoreML-native compression story

## 7. Static Shapes

Static shapes are worth caring about for CoreML EP, but they should not block `plan0`.

For this attempt:

- first verify the current float ONNX export outside the sandbox
- then verify the `fp16` ONNX copy
- only if that baseline is promising but still unsatisfactory should the repo spend time on a static-shape export path

That keeps the attempt short and diagnostic.

## 8. Deliverables

This run should produce:

- `scratchpad/attempt3_coreml/db.sqlite3`
- CoreML validation summary JSON and CSV
- CoreML validation aggregated frontier JSON
- CoreML test summary JSON and CSV
- CoreML test aggregated frontier JSON
- a later `result0.md` in this directory

Operational entry point:

- `tools/run-attempt3-coreml-plan0.py`

Tracked catalog for the run:

- `research/attempt3_coreml/study_quantization_catalog.json`
