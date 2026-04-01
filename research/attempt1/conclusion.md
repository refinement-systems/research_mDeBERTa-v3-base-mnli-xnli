# Attempt 1 Conclusion

Scope: closing summary for attempt1 after the mixed-precision PTQ, static QDQ, Chinese-focused evaluation, and gated QAT follow-up work.

## 1. What Was Tried

Attempt1 covered five main workstreams:

1. Research tooling and data preparation.
   - disjoint calibration, search-validation, and fine-tune TSV generation
   - overlap checking for attempt1 search data versus the final suite and probes
   - family-scoped exclusion reuse across dynamic search, static QDQ, NNCF PTQ, and QAT
   - probe-gated search harness updates

2. NNCF/OpenVINO-style mixed-precision PTQ on the ONNX path.
   - `nncf_accuracy_attention_only`
   - `nncf_fidelity_attention_proj_only`
   - narrower fidelity micro-batch variants around `attention_proj_only`

3. Static QDQ tuning on the same family-scoped paths.
   - `attention_only` and `attention_proj_only`
   - `S8S8`, `U8U8`, and `U8S8 + reduce_range`
   - `minmax` and `percentile`
   - calibration caps `128` and `300`

4. Chinese-focused evaluation work.
   - extra disjoint held-out zh validation slice
   - combined `attempt1-zh-sensitive-validation.tsv`
   - held-out zh stress-pack construction and benchmarking

5. Gated QAT recovery.
   - a single-seed CPU-only QAT pilot aimed at the `attention_proj_only` / `nncf_fidelity_attention_proj_only` fidelity direction

## 2. What Worked

The main positive result from attempt1 is that the infrastructure now works well enough to answer the research question more systematically than attempt0.

Operationally successful pieces:

- Disjoint attempt1 search data generation worked.
- Family-scoped exclusions were successfully reused across search methods.
- NNCF `accuracy-control` on the ONNX path worked end to end for real candidates.
- Static QDQ on the ONNX path worked end to end for real candidates.
- Probe-gated screening worked and avoided unnecessary full-suite runs.
- Chinese-focused held-out evaluation assets were successfully built.
- The harder zh stress pack worked as a diagnostic slice and did reveal a real fault line.

Meaningful research result that worked:

- `nncf_fidelity_attention_proj_only` became a valid, benchmark-backed attempt1 candidate.
  - full accuracy: `86.36%`
  - full HF agreement: `98.82%`
  - full `xnli-zh`: `84.67%`

Useful evaluation result that worked:

- the 48-row zh stress pack separated `attention_only` from the fidelity-oriented branch
- the key row was `facebook-xnli-zh-validation-000425`, where `attention_only` matched gold and `float`, `attention_proj_only`, and `nncf_fidelity_attention_proj_only` all stayed on the same wrong label

## 3. What Attempts Completed, But With Unimpressive Results

These experiments ran successfully, but did not change the recommendation frontier.

### 3.1. First NNCF mixed-precision wave

- `nncf_accuracy_attention_only`
  - full accuracy: `86.15%`
  - full HF agreement: `98.62%`
  - full `xnli-zh`: `84.67%`
  - verdict: not competitive

- `nncf_fidelity_attention_proj_only`
  - interesting and near-frontier
  - but it only tied the attempt0 fidelity frontier and still lost on `xnli-zh`
  - verdict: respectable research result, not a new best

### 3.2. Corrected static QDQ frontier sweep

The static sweep completed and produced real survivors, but none were promotion candidates.

Most notable survivors:

- `static_attention_only_u8u8_minmax_n128`
  - full accuracy: `86.21%`
  - full HF agreement: `96.62%`
  - full `xnli-zh`: `86.00%`
  - verdict: best static accuracy / Chinese-preservation result, but fidelity loss was too large

- `static_attention_proj_only_s8s8_minmax_n300`
  - full accuracy: `86.15%`
  - full HF agreement: `96.92%`
  - full `xnli-zh`: `84.67%`
  - verdict: best static fidelity result, still well below the dynamic / NNCF fidelity frontier

Overall static verdict:

- static QDQ was viable
- some candidates preserved Chinese unusually well
- none beat `attention_only`, `attention_proj_only`, or `nncf_fidelity_attention_proj_only`

### 3.3. Narrow NNCF fidelity refinement

The plan3 micro-batch completed in its `fast` bias-correction salvage path, but both candidates were screened out before the full suite:

- `nncf_fidelity_attention_proj_only_n128_drop0p005`
- `nncf_fidelity_attention_proj_only_n128_drop0p002`

Both regressed too much on probe HF agreement, so the refinement direction did not improve the frontier.

### 3.4. Generic held-out zh validation

The larger disjoint zh validation work completed and was useful, but it was not very discriminative among the strongest candidates.

- on the broader `480`-row zh-sensitive pack, the top candidates were nearly tied
- this confirmed that generic held-out zh validation alone was too weak to guide the next decision

## 4. What Attempts Were Tried, But Didn't Finish At All

### 4.1. NNCF `accurate` bias-correction branch

This branch was attempted during the narrow fidelity refinement, but did not complete into usable candidates.

Observed failure mode:

- backend failure during NNCF ONNX `accuracy-control`
- duplicate `nncf_smooth_quant_output` definitions
- `KeyError` on `/deberta/embeddings/word_embeddings/Gather`

Conclusion:

- this was not a quality loss
- it was an unfinished tooling/backend branch

### 4.2. QAT rescue pilot

The single-seed QAT pilot was launched and partially debugged, but did not finish into a benchmarkable artifact.

What happened:

- the script needed fixes for NNCF example input handling and ignored-scope validation
- after those fixes, the long CPU-only run eventually ended without producing the planned ONNX
- no persistent crash log or final traceback was left behind after the wrapper session ended

Conclusion:

- the QAT pilot was tried
- but it did not produce a usable model or benchmark result

## 5. Which Planned Attempts Weren't Tried

These items appeared in the evolving attempt1 plans, but were not actually run before concluding the attempt.

### 5.1. Alternate-runtime quantization branch

Not tried:

- `LLM.int8()`
- `QLoRA`

Reason:

- attempt1 stayed on the ONNX/CoreML-compatible path

### 5.2. Formal tooling investigation for `accurate` bias correction

Not tried:

- isolating a minimal reproduction
- testing whether the failure was caused by SmoothQuant retry interaction
- testing whether the failure was caused by preprocessing or ONNX graph naming behavior

Reason:

- the repo treated this as an optional infrastructure branch, not the mainline research path

### 5.3. Second-seed or longer QAT follow-up

Not tried:

- second QAT seed from `attention_only`
- multi-epoch QAT escalation after the pilot

Reason:

- the first QAT pilot never produced a usable artifact

### 5.4. Further PTQ expansion after plan4

Not tried:

- broader NNCF parameter sweeps after the negative plan3 micro-batch
- more broad static QDQ matrices
- reopening `percentile` static calibration
- reopening `accurate` bias-correction search variants

Reason:

- by `plan4.md`, the repo had enough evidence that further blind PTQ tuning was low-yield

## Bottom Line

Attempt1 succeeded as a research clarification pass, not as a frontier-changing model pass.

It established that:

- systematic mixed-precision PTQ is operational on this repo
- static QDQ is operational but not competitive enough
- narrow PTQ refinement did not improve the frontier
- a harder zh stress pack is possible and useful
- the attempt0 frontier still stands

So the best practical conclusion for now is unchanged:

- `attention_only` remains the best quantized accuracy artifact
- `attention_proj_only` remains the best quantized fidelity artifact
- `nncf_fidelity_attention_proj_only` is the most credible attempt1 mixed-precision result, but not a new recommendation
