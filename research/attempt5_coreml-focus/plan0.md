# Attempt 5 CoreML Focus Plan 0

Scope: first CoreML-specific follow-up after the CPU lane was closed in `attempt4_cpu-focus/result1.md`.

`attempt4` answered the CPU question for the bounded quantization catalog. The repo now has a justified CPU default:

- `nncf_accuracy_attention_only`

So `attempt5` should not reopen CPU search. It should answer the next real deployment question:

- on this Apple-silicon machine, does a CoreML path produce a better deployment tradeoff than the closed CPU winner?

This attempt is also different from `attempt3`.

- `attempt3` showed that `reference_fp16` was a viable CoreML baseline on a smaller fidelity-only pack.
- `attempt5` should validate CoreML on the broader attempt4 pack and include the metrics that mattered in the closed CPU study:
  - gold-label accuracy
  - float-reference fidelity
  - runtime
  - resident memory

## 1. Goal

The goal of `attempt5` is:

1. verify that CoreML EP runs cleanly outside the sandbox on the broad attempt4 pack
2. retest `reference_fp16` as the primary CoreML candidate
3. test the closed CPU finalists as CoreML controls
4. compute a CoreML frontier on:
   - on-disk size
   - persistent warm latency
   - resident-after-warmup RSS
5. determine the default CoreML recommendation for this machine

The CoreML result will then be compared against the closed CPU recommendation in a later result document.

## 2. Attempt Boundary

This is not a new untouched model-family search.

Reason:

- `attempt4` already opened its locked final and stress sets
- the model family is already understood well enough from the CPU study
- `attempt5` is now a backend-focused deployment comparison on the same broad evaluation pack

So `attempt5` should reuse the attempt4 dataset pack structure under a new scratchpad root.

That gives two benefits:

1. direct comparability with the closed CPU result
2. no need to invent yet another split family just to answer a backend question

The right claim for `attempt5` will therefore be:

- CoreML recommendation on this machine under the reused attempt4 evaluation protocol

not:

- a fresh untouched generalization result for the model family

## 3. Candidate Set

Primary CoreML candidates:

- `reference`
- `reference_fp16`

Carry-forward CoreML controls:

- `nncf_accuracy_attention_only`
- `nncf_fidelity_attention_only_n128_drop0p005`
- `nncf_fidelity_attention_proj_only`

Interpretation of the controls:

- they are not the mainline design target for CoreML
- they are there to answer whether the CPU-winning ONNX artifacts survive the CoreML path well enough to matter

Do not add in `plan0`:

- the tiny dynamic ORT tier
- more static-QDQ candidates
- a new NNCF search
- Core ML native compression work
- Neural-Engine-specific `W8A8` or `int4` claims

That work is not justified until the baseline CoreML lane is measured on the current machine.

## 4. Dataset Pack

Reuse the attempt4 pack structure under:

- `scratchpad/attempt5_coreml_focus`

using the same role layout:

`calibration`

- reuse the attempt4 frozen calibration slices

`smoke`

- `hf-probe-set.tsv`
- `hf-core-probe.tsv`

`fidelity_validation`

- `mnli-validation_matched-attempt4-dev.tsv`
- `mnli-validation_mismatched-attempt4-dev.tsv`
- `anli-r1-dev-attempt4-dev.tsv`
- `anli-r2-dev-attempt4-dev.tsv`
- `anli-r3-dev-attempt4-dev.tsv`

`fidelity_test`

- XNLI test all 15 languages
- ANLI test rounds 1-3

`stress_test`

- HANS eval
- WANLI test

This keeps the CoreML attempt directly comparable to the CPU result.

## 5. Selection Rule

Use the same development gates as `attempt4`, but against the CoreML float reference.

Development gates:

- MNLI dev macro accuracy drop must be `<= 0.5` points vs CoreML float
- ANLI dev macro accuracy drop must be `<= 1.0` point vs CoreML float
- no individual MNLI split or ANLI dev round may drop by more than `1.5` points vs CoreML float
- aggregate float-label agreement must be `>= 98.0%`
- peak RSS must be recorded and must not exceed CoreML float by more than `25%`

Locked-final frontier axes:

- on-disk size
- persistent warm median latency
- resident-after-warmup RSS

Peak RSS is a guardrail, not a frontier axis.

Recommendation tie-break:

1. smallest nondominated locked-final survivor
2. lower persistent warm latency
3. lower steady RSS
4. higher locked-final aggregate accuracy

## 6. Expected Outcomes

The most likely outcome is still:

- `reference_fp16` remains the default CoreML recommendation

Why:

- `attempt3` already showed perfect CoreML float-label agreement on a smaller pack
- the CPU study showed that ONNX quantized artifacts can save size and RSS but often hurt warm latency
- CoreML may favor a lighter float path more than it favors ONNX int8 carry-forwards

But this should now be measured, not assumed.

There are three live possibilities.

### 6.1. `reference_fp16` wins cleanly

This is the expected mainline outcome.

That would mean:

- keep `reference_fp16` as the CoreML default
- treat the carried CPU quantized artifacts as informative negative or secondary controls

### 6.2. A carry-forward quantized ONNX artifact wins

This is less likely, but the plan should allow it.

That would mean:

- the CPU-winning artifact family transfers onto the CoreML path better than expected
- the later result should compare that directly against the CPU deployment default

### 6.3. The CoreML baseline is invalid or operationally weak

For example:

- fallback to CPU
- incomplete benchmarking
- poor warm latency and memory relative to CPU without compensating wins

That would not be a failure of the study.

It would mean the correct conclusion is:

- CoreML is not currently competitive enough on this machine under the current path

## 7. Entry Point

Prepared entry point for local terminal execution:

- `tools/run-attempt5-coreml-study.py`

Prepared catalog:

- `research/attempt5_coreml-focus/study_quantization_catalog.json`

Prepared report builder:

- `tools/build-attempt5-coreml-report.py`

Suggested full run command:

```bash
python3 tools/run-attempt5-coreml-study.py --force
```

Suggested narrow preflight command:

```bash
python3 tools/run-attempt5-coreml-study.py --skip-controls --skip-test --force
```

Important operational note:

- this attempt should be run in your own terminal outside the Codex sandbox
- CoreML benchmarking and RSS capture are expected to be more reliable there

## 8. Deliverables

This attempt should produce:

- `scratchpad/attempt5_coreml_focus/reports/attempt5-validation-summary.json`
- `scratchpad/attempt5_coreml_focus/reports/attempt5-validation-coreml-persistent.csv`
- `scratchpad/attempt5_coreml_focus/reports/attempt5-test-summary.json`
- `scratchpad/attempt5_coreml_focus/reports/attempt5-stress-summary.json`
- `scratchpad/attempt5_coreml_focus/reports/attempt5-test-coreml-cold.csv`
- `scratchpad/attempt5_coreml_focus/reports/attempt5-coreml-summary.md`
- `scratchpad/attempt5_coreml_focus/reports/attempt5-coreml-summary.json`
- `scratchpad/attempt5_coreml_focus/reports/attempt5-manifest.json`

Later research write-up:

- `research/attempt5_coreml-focus/result0.md`

## 9. Bottom Line

`attempt5` should be a narrow backend study, not another broad quantization search.

The right next question is:

- is CoreML `fp16` or any carried CPU winner actually better than the closed CPU recommendation on this machine?

The prepared scripts are aimed directly at answering that question and nothing broader.
