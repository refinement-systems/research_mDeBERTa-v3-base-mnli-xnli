# Quantization Plan 1

Scope: next-step quantization work after `22a3c05d157b056a609659b985169dc0fe257ea7`.

This document is intentionally forward-looking. For the historical record of what has already been tried, see `QUANTIZATION_REPORT_1.md`.

## 1. Current Position

The current reference behavior is:

- Hugging Face / PyTorch
- `models/mdeberta/onnx/model.onnx`

Those two are already close enough that quantization work should treat the float ONNX file as the local ground truth.

The best quantized candidate produced so far is:

- `models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx`

But it is still not a faithful drop-in replacement:

- float ONNX aggregate on the non-overlapping benchmark suite: `1684/1950 = 86.36%`
- best quantized candidate aggregate: `1665/1950 = 85.38%`
- agreement with float: `1845/1950 = 94.62%`

So the next work should optimize for benchmark-backed fidelity, not for fixing one anecdotal pair.

## 2. Success Criteria

The next round of work should use these acceptance bars:

- primary goal: improve on `dynamic_qint8_matmul_exclude_suggested.onnx`
- no single-pair tuning without benchmark confirmation
- benchmark against the non-overlapping suite only:
  - `benchmarks/nli/mnli-validation_matched-200-per-label.tsv`
  - `benchmarks/nli/mnli-validation_mismatched-200-per-label.tsv`
  - `benchmarks/nli/xnli-de-test-50-per-label.tsv`
  - `benchmarks/nli/xnli-en-test-50-per-label.tsv`
  - `benchmarks/nli/xnli-es-test-50-per-label.tsv`
  - `benchmarks/nli/xnli-fr-test-50-per-label.tsv`
  - `benchmarks/nli/xnli-zh-test-50-per-label.tsv`

Working thresholds:

- preferred: within `0.5` absolute accuracy points of float on the aggregate suite
- minimum plausible candidate: no worse than the current best candidate aggregate of `85.38%`
- agreement target: above `95%` against float, preferably higher
- qualitative check: no obvious collapse on `mnli-validation_mismatched` or multilingual slices

## 3. What To Try

### 3.1. Structured exclusion-family search

This is the highest-priority next step.

Why:

- the best current candidate came from excluding a narrow set of sensitive `MatMul` nodes
- the winning exclusions were concentrated in mid/high-layer FFN and output-dense projections
- broad heuristics like "exclude all Q/K/V" or "exclude whole late layers" did not hold up

What to search:

- FFN-only exclusions in layers `8-11`
- FFN-only exclusions in layers `6-11`
- FFN-only exclusions in layers `4-11`
- attention-output-dense exclusions in upper layers
- leave the full layer `11` block float
- leave layers `10-11` float
- quantize attention only
- quantize FFN only

How to evaluate:

- generate candidates systematically
- benchmark each candidate on the non-overlapping suite
- keep only the top few for HF-logit comparison

Expected value:

- high

Risk:

- medium

### 3.2. Better static quantization sweep

This is the second-priority path.

Why:

- the current static attempt was only a smoke test
- static quantization can outperform naive dynamic quantization when calibration data is representative

What to sweep:

- calibration method:
  - `minmax`
  - `entropy`
  - `percentile`
- graph format:
  - `qdq`
  - `qoperator`
- op set:
  - `MatMul`
  - `MatMul` + `Add`
- activation type:
  - `qint8`
  - `quint8`
- preprocessing:
  - on
  - off

Calibration data should mix:

- MNLI matched
- MNLI mismatched
- XNLI languages that matter to this repo, at minimum `en`, `de`, `fr`, `zh`

Expected value:

- medium to high

Risk:

- medium to high

### 3.3. Aggregate drift-driven exclusion suggestions

The debugger should be used over multiple benchmark sources, not one hand-picked pair.

Why:

- the legacy single-example approach found one decent candidate
- it also overfit the anecdotal example and did not fully generalize

What to do:

- run `tools/debug-onnx-quantization.py` over a mixed benchmark subset
- rank sensitive nodes by aggregate drift, not per-example drift
- turn the output into structured exclusion families instead of one-off manual choices

Expected value:

- medium

Risk:

- low to medium

### 3.4. Fresh export and alternate quantization toolchain

Why:

- current work is largely post-training quantization on the existing float ONNX graph
- some graph shapes quantize better than others

What to try:

- re-export the model to ONNX from the reference weights under controlled settings
- compare quantization behavior on the fresh export
- if practical, test an alternate toolchain such as Optimum or Olive instead of only the current ORT flow

Expected value:

- medium

Risk:

- medium

### 3.5. Reduced-precision fallback paths

Only do this if the project goal is "smaller/faster with acceptable fidelity" rather than "must be int8".

Why:

- this model may simply be less friendly to aggressive integer quantization than hoped
- FP16 or other milder compression paths may preserve behavior much better

What to try:

- FP16 ONNX export
- benchmark FP16 against float and int8 candidates

Expected value:

- medium

Risk:

- low

## 4. What Not To Try

These are low-value or already disproven directions and should not be the default next experiments.

### 4.1. Do not optimize against one anecdotal sentence pair

Reason:

- this already produced candidates that looked better on the Angela/Macron pair but regressed on real benchmarks

### 4.2. Do not spend more time on blind default dynamic sweeps

Reason:

- `dynamic_qint8_default`
- `dynamic_qint8_per_channel`
- `dynamic_qint8_matmul`
- `dynamic_qint8_matmul_per_channel`
- `dynamic_quint8_matmul`

were already far from HF and did not solve the problem

### 4.3. Do not assume broad Q/K/V exclusions are the answer

Reason:

- `dynamic_qint8_matmul_exclude_all_qkv.onnx` did not fix the core issue
- the best current candidate points more toward selective FFN/output sensitivity than blanket attention exclusion

### 4.4. Do not trust a candidate that only improves MNLI matched

Reason:

- the current best candidate slightly improved MNLI matched but still regressed on MNLI mismatched and all tested XNLI slices
- this repo needs robustness, not only a matched-domain win

### 4.5. Do not treat preprocessing alone as a meaningful quantization fix

Reason:

- `dynamic_qint8_matmul_preprocessed.onnx` did not materially improve fidelity

## 5. Recommended Execution Order

1. Build a benchmark-aware candidate search harness for structured exclusion families.
2. Run the structured exclusion sweep and keep the top `3-5` candidates.
3. Run a real static quantization grid using mixed calibration data.
4. Compare the best static candidates against the best structured dynamic candidates.
5. For finalists, run both:
   - full benchmark evaluation
   - HF-logit comparison on a small probe set
6. Only if those paths stall, try fresh export plus alternate toolchain work.
7. If int8 still cannot get close enough, test FP16 as the pragmatic fallback.

## 6. Minimum Useful Tooling Additions

The next code changes should probably be:

- a search harness that:
  - generates structured exclusion-family candidates
  - runs `tools/benchmark-nli-models.sh`
  - emits one sortable summary table
- optional CSV or JSON summary output from `tools/benchmark-nli-models.sh`
- optional confusion-matrix or paired win/loss reporting for candidate comparison

## 7. Practical Recommendation

The immediate next implementation should be:

1. automate structured exclusion-family generation,
2. benchmark those candidates on the non-overlapping suite,
3. keep only the candidates that beat `dynamic_qint8_matmul_exclude_suggested.onnx`.

That is the shortest path to determining whether int8 can be made acceptable for this model in this repo.
