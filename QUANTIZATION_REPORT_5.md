# Quantization Report 5

Scope: hard-case finalist benchmarking after `QUANTIZATION_REPORT_4.md`, using:

- `tools/benchmark-hf-onnx-models.py`
- `tools/build-hf-probe-set.py`
- `benchmarks/nli/hf-finalist-full.json`
- `benchmarks/nli/hf-finalist-full.csv`
- `benchmarks/nli/hf-probe-set.tsv`
- `benchmarks/nli/hf-probe-benchmark.json`
- `benchmarks/nli/hf-probe-benchmark.csv`

This report covers:

1. why a fixed hard-case probe set was added,
2. how that probe set was constructed,
3. what the full-suite HF benchmark says now,
4. what the probe benchmark says,
5. how future candidates should be judged.

## 1. Why A Fifth Benchmark Was Needed

`QUANTIZATION_REPORT_4.md` already established a stable frontier:

- accuracy-first candidate:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`
- fidelity-first candidate:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

That report used broad benchmark slices and HF-vs-ONNX logit comparisons, which answered:

- which candidate scores best overall
- which candidate stays slightly closer to HF overall

But it still left one practical gap:

- there was no fixed, reusable set of "difficult" examples to stress future candidates quickly

So this round added a second tool:

- `tools/build-hf-probe-set.py`

Its purpose is to turn the full HF benchmark output into a smaller, stable regression probe containing the exact cases where the finalists drift most or disagree most often.

## 2. Tooling Added

### 2.1. `tools/benchmark-hf-onnx-models.py`

This tool was introduced in the previous step and is now part of the reproducible benchmarking workflow.

It:

- loads the HF/PyTorch reference model
- loads the float ONNX model
- loads selected quantized ONNX candidates
- evaluates them over TSV benchmark files
- reports:
  - accuracy against gold labels
  - label agreement with HF
  - label agreement with float ONNX
  - mean and max logit drift relative to HF
  - disagreement and top-drift examples

### 2.2. `tools/build-hf-probe-set.py`

This new tool reads the full benchmark JSON and emits a smaller TSV probe set.

The probe set is built from:

- all HF disagreement examples for the selected finalists
- top-drift examples for the selected finalists
- examples where the finalists predict different labels
- optional high-drift float examples

That makes it useful as a permanent stress test for future quantization rounds.

## 3. Full-Suite HF Benchmark Recap

The full-suite benchmark is in:

- `benchmarks/nli/hf-finalist-full.json`
- `benchmarks/nli/hf-finalist-full.csv`

It covers the same non-overlapping 1,950-example benchmark suite used in the later quantization reports:

- `mnli-validation_matched-200-per-label.tsv`
- `mnli-validation_mismatched-200-per-label.tsv`
- `xnli-de-test-50-per-label.tsv`
- `xnli-en-test-50-per-label.tsv`
- `xnli-es-test-50-per-label.tsv`
- `xnli-fr-test-50-per-label.tsv`
- `xnli-zh-test-50-per-label.tsv`

Results:

| Candidate | Accuracy | HF Agreement | Mean Max Logit Drift vs HF | Max Logit Drift vs HF |
| --- | ---: | ---: | ---: | ---: |
| `attention_only` | `1690/1950 = 86.67%` | `1926/1950 = 98.77%` | `0.166767` | `2.068416` |
| `attention_proj_only` | `1687/1950 = 86.51%` | `1927/1950 = 98.82%` | `0.165053` | `2.714536` |
| `float` | `1684/1950 = 86.36%` | `1950/1950 = 100.00%` | `0.007504` | `0.119377` |

So the global reading remains unchanged:

- `attention_only` is still the best benchmark-accuracy candidate
- `attention_proj_only` is still the slightly better HF-fidelity candidate
- `float` still remains effectively label-identical to HF

## 4. Hard-Case Probe Set Construction

The generated probe set is:

- `benchmarks/nli/hf-probe-set.tsv`

It was built from the full benchmark JSON with:

```bash
.venv/bin/python tools/build-hf-probe-set.py \
  --include-finalist-label-diffs \
  --include-float-top-drift 10
```

The resulting probe summary was:

- candidates:
  - `attention_only`
  - `attention_proj_only`
- rows:
  - `61`
- `attention_only_hf_disagreements`:
  - `24`
- `attention_proj_only_hf_disagreements`:
  - `23`
- `finalist_label_differences`:
  - `19`

This is intentionally not a balanced benchmark slice.

It is a stress set, enriched for:

- disagreement with HF
- disagreement between finalists
- unusually large logit drift

That means it should not be read as a normal accuracy benchmark. It should be read as a regression detector.

## 5. Hard-Case Probe Results

The probe benchmark outputs are:

- `benchmarks/nli/hf-probe-benchmark.json`
- `benchmarks/nli/hf-probe-benchmark.csv`

Results:

| Candidate | Accuracy | HF Agreement | Mean Max Logit Drift vs HF | Max Logit Drift vs HF |
| --- | ---: | ---: | ---: | ---: |
| `attention_only` | `34/61 = 55.74%` | `37/61 = 60.66%` | `0.747011` | `2.068416` |
| `attention_proj_only` | `31/61 = 50.82%` | `38/61 = 62.30%` | `0.873891` | `2.714536` |
| `float` | `28/61 = 45.90%` | `61/61 = 100.00%` | `0.024824` | `0.119377` |

These numbers are lower than the full-suite benchmark on purpose, because the probe set is biased toward hard cases.

What matters is not the absolute number. What matters is the shape of the tradeoff:

- `attention_only` is still better on gold-label accuracy
- `attention_proj_only` is still slightly better on HF-label agreement
- `float` remains the exact HF-label baseline

So the hard probe confirms the same frontier seen in the full suite rather than overturning it.

## 6. What The Probe Adds Beyond Report 4

`QUANTIZATION_REPORT_4.md` already showed that:

- `attention_only` wins on aggregate benchmark score
- `attention_proj_only` wins slightly on fidelity

This fifth report adds a fixed regression asset:

- `benchmarks/nli/hf-probe-set.tsv`

That is useful because future candidate sweeps can now be judged on two axes:

1. broad benchmark quality:
   - MNLI/XNLI slices
2. concentrated hard-case behavior:
   - probe-set performance and HF agreement

This should prevent two common mistakes:

- picking a candidate that looks good only because it improves easy cases
- picking a candidate that matches HF on average but collapses on known drift hotspots

## 7. Current Practical Conclusion

The repo now has:

### 7.1. Accuracy-oriented quantized baseline

- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`

Why:

- best full-suite benchmark accuracy
- also better on the hard probe's gold-label accuracy

### 7.2. Fidelity-oriented quantized baseline

- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

Why:

- slightly better HF-label agreement on both the full suite and the probe set

### 7.3. Reference baseline

- `models/mdeberta/onnx/model.onnx`

Why:

- still the faithful ONNX representation of the original HF model

## 8. Recommended Benchmark Gate For Future Candidates

Future quantized candidates should now be checked against both:

1. the full non-overlapping benchmark suite
2. the hard probe set

A candidate should not be considered better unless it improves one of these goals without clearly regressing the other:

- higher full-suite accuracy than `attention_only`
- better HF agreement than `attention_proj_only`
- no obvious blow-up on the hard probe set

This is a better gate than using only one benchmark family or only one sentence-pair probe.

## 9. Reproduction

### 9.1. Rebuild The Full HF Benchmark

```bash
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --max-examples-per-source 0 \
  --sample-mode first \
  --summary-json benchmarks/nli/hf-finalist-full.json \
  --summary-csv benchmarks/nli/hf-finalist-full.csv
```

### 9.2. Build The Hard Probe Set

```bash
.venv/bin/python tools/build-hf-probe-set.py \
  --include-finalist-label-diffs \
  --include-float-top-drift 10
```

### 9.3. Benchmark The Probe Set

```bash
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --tsv benchmarks/nli/hf-probe-set.tsv \
  --sample-mode first \
  --max-examples-per-source 0 \
  --summary-json benchmarks/nli/hf-probe-benchmark.json \
  --summary-csv benchmarks/nli/hf-probe-benchmark.csv
```

## 10. Bottom Line

The new benchmarking round did not replace the current conclusion. It strengthened it.

The repo now has:

- a broad benchmark that ranks the finalists reliably
- a fixed hard-case probe set that captures known drift hotspots
- a cleaner evaluation gate for future quantization work

And the current frontier is still:

- best accuracy:
  - `attention_only`
- best fidelity:
  - `attention_proj_only`
