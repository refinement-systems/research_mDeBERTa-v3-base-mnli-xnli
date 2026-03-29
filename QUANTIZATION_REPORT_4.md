# Quantization Report 4

Scope: Hugging Face vs ONNX finalist benchmarking performed with `tools/benchmark-hf-onnx-models.py` on the current workspace.

This report covers:

1. why a new benchmark path was added,
2. what the new benchmark measured,
3. the results on a sampled multilingual probe and on the full 1,950-example suite,
4. what this says about `attention_only` vs `attention_proj_only`.

## 1. Why A Fourth Benchmark Was Needed

`QUANTIZATION_REPORT_3.md` ended with two credible quantized finalists:

- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`
- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

At that point the remaining question was not "which one scores slightly higher on MNLI/XNLI?" because both had already cleared that bar.

The remaining question was:

- which candidate is closer to the original Hugging Face/PyTorch model behavior?

To answer that, the repo added:

- `tools/benchmark-hf-onnx-models.py`

This tool:

- loads the HF/PyTorch reference model once
- loads the float ONNX model once
- loads one or more quantized ONNX candidates once
- evaluates them over TSV benchmark files
- reports:
  - accuracy against gold labels
  - label agreement with HF
  - label agreement with the float ONNX model
  - mean and max logit drift relative to HF
  - top-drift and disagreement examples

Important caveat:

- this tool runs Python `transformers` + Python `onnxruntime` directly
- it does not benchmark the C++ inference path
- so it should be read as a model-artifact fidelity benchmark, not a replacement for `builddir/nli-eval`

## 2. What Was Benchmarked

Candidates compared:

- `float`
- `attention_only`
- `attention_proj_only`

Benchmark sources:

- `mnli-validation_matched-200-per-label.tsv`
- `mnli-validation_mismatched-200-per-label.tsv`
- `xnli-de-test-50-per-label.tsv`
- `xnli-en-test-50-per-label.tsv`
- `xnli-es-test-50-per-label.tsv`
- `xnli-fr-test-50-per-label.tsv`
- `xnli-zh-test-50-per-label.tsv`

Two runs were used:

1. a small sampled diagnostic run:
   - `70` examples total
   - `10` examples per source
   - random sampling with seed `0`

2. a full-suite run:
   - all `1950` examples across the non-overlapping benchmark suite

Outputs:

- sampled run:
  - `benchmarks/nli/hf-finalist-benchmark.json`
  - `benchmarks/nli/hf-finalist-benchmark.csv`
- full run:
  - `benchmarks/nli/hf-finalist-full.json`
  - `benchmarks/nli/hf-finalist-full.csv`

## 3. Full-Suite Results

Results from `benchmarks/nli/hf-finalist-full.csv`:

| Candidate | Accuracy | HF Agreement | Float Agreement | Mean Max Logit Drift vs HF | Max Logit Drift vs HF |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention_only` | `1690/1950 = 86.67%` | `1926/1950 = 98.77%` | `1926/1950 = 98.77%` | `0.166767` | `2.068416` |
| `attention_proj_only` | `1687/1950 = 86.51%` | `1927/1950 = 98.82%` | `1927/1950 = 98.82%` | `0.165053` | `2.714536` |
| `float` | `1684/1950 = 86.36%` | `1950/1950 = 100.00%` | `1950/1950 = 100.00%` | `0.007504` | `0.119377` |

Three conclusions follow directly from this:

1. `float` is effectively label-identical to HF on the entire 1,950-example suite.
2. `attention_only` still wins on raw benchmark accuracy.
3. `attention_proj_only` is slightly closer to HF by label agreement and by average logit drift, but not by enough to beat `attention_only` on accuracy.

## 4. What Changed Relative To Earlier Reports

Earlier reports already suggested:

- `attention_only` is the best accuracy-oriented candidate
- `attention_proj_only` is the most plausible fidelity-oriented candidate

This fourth report strengthens that conclusion rather than overturning it.

What is new here is:

- the fidelity-oriented claim is no longer based only on one or two probe sentences
- it is now backed by a full 1,950-example HF-vs-ONNX run

So the tradeoff is now measured directly:

- `attention_only` gains `+3` correct examples over `attention_proj_only`
- `attention_proj_only` gains `+1` HF agreement over `attention_only`
- `attention_proj_only` also has slightly lower average drift:
  - `0.165053` vs `0.166767`

That is a real but very small fidelity edge.

## 5. The Pareto Frontier

The current frontier is:

### 5.1. Best benchmark-accuracy candidate

- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`

Why:

- best accuracy on both the C++ benchmark path and the Python HF/ORT benchmark path

### 5.2. Best fidelity-oriented candidate

- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

Why:

- slightly higher HF-label agreement than `attention_only`
- slightly smaller average logit drift over the full suite
- much smaller logit drift on the original Angela/Macron probe case from earlier reports

## 6. Sampled Diagnostic Run

The smaller `70`-example run in `benchmarks/nli/hf-finalist-benchmark.json` was still useful for diagnostics.

It showed:

- `attention_only` had `1` HF disagreement in that sample
- `attention_proj_only` had `0` HF disagreements in that sample

But it also showed why label agreement alone is not enough:

- there were examples where labels still matched HF while logits drifted substantially

That is why the full-suite run tracked both:

- label agreement
- logit drift

## 7. Important Diagnostic Examples

The full benchmark exposed concrete high-drift examples worth keeping around as probes.

Examples for `attention_only` from `benchmarks/nli/hf-finalist-full.json`:

- `xnli-es-test-50-per-label.tsv`
  - `facebook-xnli-es-test-000003`
  - HF label: `neutral`
  - model label: `neutral`
  - max abs logit drift: `2.068416`

- `xnli-zh-test-50-per-label.tsv`
  - `facebook-xnli-zh-test-000132`
  - HF label: `neutral`
  - model label: `contradiction`
  - max abs logit drift: `1.712358`

Examples for `attention_proj_only`:

- `xnli-fr-test-50-per-label.tsv`
  - `facebook-xnli-fr-test-000051`
  - HF label: `entailment`
  - model label: `entailment`
  - max abs logit drift: `2.714536`

- `nyu-mll-multi_nli-default-validation_mismatched-000672`
  - HF label: `contradiction`
  - model label: `neutral`
  - max abs logit drift: `2.052989`

These examples are useful because they show:

- some quantized errors are outright label flips
- some quantized cases preserve the top label but still move logits a great deal

## 8. What This Means Technically

By now, the repo has strong evidence for all of the following:

- the float ONNX export is faithful to HF
- quantizing FFN dense layers is the primary source of bad degradation
- quantizing attention-side layers only is the right broad direction
- among attention-only strategies, there is no obvious follow-up sweep that strictly dominates

So the remaining problem is no longer "find a better broad quantization family."

The remaining problem is one of fine-grained optimization:

- either pick `attention_only` and accept the tiny fidelity cost in exchange for better accuracy
- or pick `attention_proj_only` and accept the tiny accuracy cost in exchange for slightly better fidelity

## 9. Reproduction

### 9.1. Sampled Diagnostic Run

```bash
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --max-examples-per-source 10 \
  --sample-mode random \
  --seed 0 \
  --show-slices \
  --summary-json benchmarks/nli/hf-finalist-benchmark.json \
  --summary-csv benchmarks/nli/hf-finalist-benchmark.csv
```

### 9.2. Full-Suite Run

```bash
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --max-examples-per-source 0 \
  --sample-mode first \
  --summary-json benchmarks/nli/hf-finalist-full.json \
  --summary-csv benchmarks/nli/hf-finalist-full.csv
```

### 9.3. Inspect The Full Summary

```bash
sed -n '1,80p' benchmarks/nli/hf-finalist-full.csv
```

## 10. Bottom Line

`QUANTIZATION_REPORT_3.md` established the accuracy-vs-fidelity frontier.

This fourth report makes that frontier more defensible:

- `attention_only` is still the best accuracy-oriented quantized candidate
- `attention_proj_only` is still the slightly better fidelity-oriented quantized candidate
- the float ONNX model matches HF labels on the full 1,950-example suite, so it remains the reliable reference

If a single quantized default must be chosen, the practical choice is still:

- `attention_only`

If the repo wants to preserve a second artifact explicitly for fidelity-sensitive comparisons, it should keep:

- `attention_proj_only`
