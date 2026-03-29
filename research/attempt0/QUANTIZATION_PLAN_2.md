# Quantization Plan 2

Scope: future benchmarking and candidate-selection work after `QUANTIZATION_REPORT_5.md`.

This document is intentionally forward-looking. For the historical record of what has already been tried, see:

- `QUANTIZATION_REPORT_1.md`
- `QUANTIZATION_REPORT_2.md`
- `QUANTIZATION_REPORT_3.md`
- `QUANTIZATION_REPORT_4.md`
- `QUANTIZATION_REPORT_5.md`

## 1. Current Position

The repo now has three meaningful baselines:

- reference model:
  - `models/mdeberta/onnx/model.onnx`
- accuracy-oriented quantized candidate:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`
- fidelity-oriented quantized candidate:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

Current broad benchmark status from `QUANTIZATION_REPORT_5.md`:

- float:
  - `1684/1950 = 86.36%`
  - `1950/1950 = 100.00%` HF agreement
- `attention_only`:
  - `1690/1950 = 86.67%`
  - `1926/1950 = 98.77%` HF agreement
- `attention_proj_only`:
  - `1687/1950 = 86.51%`
  - `1927/1950 = 98.82%` HF agreement

Current hard-probe status:

- `attention_only`:
  - `34/61 = 55.74%`
  - `37/61 = 60.66%` HF agreement
- `attention_proj_only`:
  - `31/61 = 50.82%`
  - `38/61 = 62.30%` HF agreement

So the current problem is no longer "find any quantized model that works."

The current problem is:

- make future comparisons stricter,
- measure runtime cost alongside accuracy and fidelity,
- avoid overfitting candidate search to the current benchmark mix.

## 2. Primary Goal

The next round should not primarily search for more random quantization variants.

The next round should build a stable benchmark gate that future candidates must pass.

The gate should answer four questions:

1. does the candidate beat `attention_only` on broad benchmark accuracy?
2. does the candidate beat `attention_proj_only` on HF fidelity?
3. does the candidate avoid regressing on the hard probe set?
4. does the candidate deliver a runtime or size benefit that justifies any remaining behavioral drift?

## 3. Benchmark Tiers To Freeze

Future work should standardize on three benchmark tiers.

### 3.1. Tier 1: Core Probe

Purpose:

- very fast iteration
- reject obviously bad candidates before running heavier benchmarks

Target:

- about `20-30` examples

Composition:

- a compact subset of `benchmarks/nli/hf-probe-set.tsv`
- must preserve:
  - HF disagreements for both finalists
  - at least one hard case from each major source family
  - at least one Chinese example
  - at least one example where `attention_only` and `attention_proj_only` differ

Why:

- the current `61`-example probe is useful but still large enough to tempt overfitting

### 3.2. Tier 2: Hard Probe

Purpose:

- focused regression detection

Current asset:

- `benchmarks/nli/hf-probe-set.tsv`

Rules:

- this stays fixed unless there is a strong reason to refresh it
- if refreshed, preserve the old version for comparability

### 3.3. Tier 3: Full Suite

Purpose:

- broad quality check
- final ranking of serious candidates

Current asset set:

- `benchmarks/nli/mnli-validation_matched-200-per-label.tsv`
- `benchmarks/nli/mnli-validation_mismatched-200-per-label.tsv`
- `benchmarks/nli/xnli-de-test-50-per-label.tsv`
- `benchmarks/nli/xnli-en-test-50-per-label.tsv`
- `benchmarks/nli/xnli-es-test-50-per-label.tsv`
- `benchmarks/nli/xnli-fr-test-50-per-label.tsv`
- `benchmarks/nli/xnli-zh-test-50-per-label.tsv`

## 4. What To Build Next

### 4.1. Paired Benchmark Reporting

This is the highest-priority tooling task.

Extend `tools/benchmark-hf-onnx-models.py` so that it reports:

- candidate fixed float error
- candidate introduced new error relative to float
- candidate fixed other quantized-candidate error
- per-source wins/losses
- per-language wins/losses
- explicit `xnli-zh` summary
- confusion-style counts for:
  - HF vs candidate
  - float vs candidate

Why:

- aggregate accuracy alone is no longer enough
- the leading candidates are close enough that we need paired outcomes, not only totals

Expected value:

- high

Risk:

- low

### 4.2. Runtime Benchmarking

This is the second-priority task.

Add a benchmark path for actual shipped inference behavior, not just model-artifact fidelity.

Measure at least:

- model file size
- model load time
- warm inference latency
- median latency
- p95 latency
- CPU backend performance
- CoreML backend performance

Candidates to measure:

- float
- `attention_only`
- `attention_proj_only`

Suggested implementation:

- use `builddir/nli` or `builddir/nli-eval`
- benchmark over:
  - the core probe
  - the hard probe
  - optionally a small random full-suite sample

Why:

- the quantized finalists are now close enough on quality that runtime may decide which one is actually worth shipping

Expected value:

- high

Risk:

- low to medium

### 4.3. Core Probe Builder

Add a small tool that derives a stable `20-30` example probe from `hf-probe-set.tsv`.

Rules for the builder:

- preserve source diversity
- preserve language diversity
- preserve finalist disagreement examples
- preserve top-drift examples
- keep the output deterministic

Why:

- future candidate sweeps need a cheap first-stage filter

Expected value:

- medium to high

Risk:

- low

### 4.4. Benchmark Dashboard Output

Add one compact summary output that combines:

- full-suite accuracy
- hard-probe accuracy
- core-probe accuracy
- HF agreement
- float agreement
- `xnli-zh` slice metrics
- runtime metrics

Possible formats:

- CSV
- JSON
- Markdown report

Why:

- future comparisons should not require reading multiple files by hand

Expected value:

- medium

Risk:

- low

## 5. What To Try After The Benchmark Gate Exists

Only after the benchmark gate is in place should the repo continue with more quantization search.

### 5.1. Narrow Candidate Search Around The Current Frontier

The broad family sweeps are largely done.

The next candidate search should be narrow and benchmark-aware.

Good directions:

- minor relaxations around `attention_only`
- minor relaxations around `attention_proj_only`
- language-aware checks for `xnli-zh` sensitivity
- targeted tests only if they are evaluated against:
  - core probe
  - hard probe
  - full suite

### 5.2. Revisit Static Quantization Only With The New Gate

Static quantization is still worth testing, but only after the new benchmark gate exists.

Why:

- the earlier static attempt was too shallow
- without a stricter benchmark flow, static experiments will be hard to compare fairly

### 5.3. Revisit Alternate Export Toolchains Only If Needed

Fresh exports or alternate toolchains remain possible, but they should now be justified by benchmark data, not curiosity.

Use them only if:

- current ORT-style quantization plateaus, and
- runtime or fidelity targets remain unmet

## 6. What Not To Do

### 6.1. Do Not Resume Blind Candidate Sweeps First

Reason:

- the repo already has a credible frontier
- more uncontrolled sweeps will add noise before the benchmark gate is improved

### 6.2. Do Not Judge Future Candidates On Aggregate Accuracy Alone

Reason:

- `attention_only` and `attention_proj_only` already show that accuracy and fidelity can diverge slightly

### 6.3. Do Not Optimize Only For The Hard Probe

Reason:

- the hard probe is intentionally biased toward difficult cases
- it is a regression detector, not a standalone benchmark

### 6.4. Do Not Ignore Runtime

Reason:

- once model quality differences become small, runtime and load cost are part of whether quantization is worth keeping

### 6.5. Do Not Refresh The Probe Sets Casually

Reason:

- frequent probe churn destroys comparability
- future reports should be able to compare candidate results against stable assets

## 7. Acceptance Criteria For Future Candidates

A future quantized candidate should only be considered better if it satisfies at least one of these:

- higher full-suite accuracy than `attention_only`
- better HF agreement than `attention_proj_only`
- better runtime and size characteristics than both current finalists with no meaningful benchmark regression

And it must also satisfy all of these:

- no obvious collapse on `benchmarks/nli/hf-probe-set.tsv`
- no obvious regression on `xnli-zh`
- no sign of single-slice overfitting

## 8. Recommended Execution Order

1. add paired benchmark reporting to `tools/benchmark-hf-onnx-models.py`
2. add a deterministic core-probe builder
3. add runtime benchmarking for float vs `attention_only` vs `attention_proj_only`
4. add a compact combined benchmark summary
5. only then resume targeted candidate search

## 9. Bottom Line

The next phase should be benchmark infrastructure, not more speculative quantization.

The repo already knows enough to say:

- `attention_only` is the current accuracy winner
- `attention_proj_only` is the current fidelity winner
- future work should now focus on stronger evaluation gates and runtime measurement

That is the most credible way to improve the quantization work from here.
