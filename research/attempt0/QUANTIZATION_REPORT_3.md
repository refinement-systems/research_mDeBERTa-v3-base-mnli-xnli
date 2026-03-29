# Quantization Report 3

Scope: attention-focused follow-up sweep after `QUANTIZATION_REPORT_2.md`, based on `benchmarks/nli/attention-sweep-summary.json` and `benchmarks/nli/attention-sweep-summary.csv` in the current workspace.

This report covers:

1. what was tested in the attention follow-up sweep,
2. the aggregate and per-slice results,
3. what changed relative to `QUANTIZATION_REPORT_2.md`,
4. what the current best quantized candidates are.

## 1. What Was Tested

`QUANTIZATION_REPORT_2.md` established that the best candidate from the structured family sweep was:

- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`

That candidate:

- quantizes attention-side weight matmuls
- keeps all FFN dense matmuls in float

The next question was whether this result could be improved by narrowing or relaxing the attention-side quantization policy.

The follow-up sweep tested these candidates:

- `attention_only`
- `attention_proj_only`
- `attention_only_layer_11_float`
- `attention_only_layers_10_11_float`
- `attention_only_attention_output_layers_8_11_float`
- `attention_proj_only_layer_11_float`
- `attention_proj_only_layers_10_11_float`

Interpretation of the variants:

- `attention_proj_only`: quantize only attention projections, leave attention output dense and all FFN dense layers float
- `attention_only_layer_11_float`: start from `attention_only`, but leave all quantizable attention matmuls in layer 11 float
- `attention_only_layers_10_11_float`: same idea, but for layers 10 and 11
- `attention_only_attention_output_layers_8_11_float`: start from `attention_only`, but leave upper-layer attention output dense matmuls float
- `attention_proj_only_layer_11_float` and `attention_proj_only_layers_10_11_float`: projection-only quantization with additional upper-layer projection exclusions

The benchmark suite was unchanged:

- `mnli-validation_matched-200-per-label.tsv`
- `mnli-validation_mismatched-200-per-label.tsv`
- `xnli-de-test-50-per-label.tsv`
- `xnli-en-test-50-per-label.tsv`
- `xnli-es-test-50-per-label.tsv`
- `xnli-fr-test-50-per-label.tsv`
- `xnli-zh-test-50-per-label.tsv`

## 2. Aggregate Results

Float reference on this suite:

- `1684/1950 = 86.36%`

Results from `benchmarks/nli/attention-sweep-summary.csv`:

| Rank | Candidate | Accuracy | Delta vs float | Agreement |
| --- | --- | ---: | ---: | ---: |
| 1 | `attention_only` | `1689/1950 = 86.62%` | `+0.26` points | `98.72%` |
| 2 | `attention_proj_only` | `1686/1950 = 86.46%` | `+0.10` points | `98.77%` |
| 3 | `attention_only_attention_output_layers_8_11_float` | `1685/1950 = 86.41%` | `+0.05` points | `98.77%` |
| 4 | `attention_proj_only_layers_10_11_float` | `1683/1950 = 86.31%` | `-0.05` points | `99.23%` |
| 5 | `attention_only_layers_10_11_float` | `1682/1950 = 86.26%` | `-0.10` points | `99.08%` |
| 6 | `attention_only_layer_11_float` | `1681/1950 = 86.21%` | `-0.15` points | `98.92%` |
| 7 | `attention_proj_only_layer_11_float` | `1680/1950 = 86.15%` | `-0.21` points | `99.28%` |

## 3. Main Outcome

The follow-up sweep did not beat:

- `dynamic_qint8_matmul_attention_only.onnx`

So the accuracy winner from `QUANTIZATION_REPORT_2.md` remains the accuracy winner now.

However, the follow-up sweep clarified the frontier much better:

- `attention_only` is still the best benchmark-accuracy candidate
- `attention_proj_only` is the best alternative if fidelity to the float model matters more than the last fraction of a benchmark point

This produces a clean Pareto frontier:

- accuracy-first: `attention_only`
- fidelity-first: `attention_proj_only`

## 4. Per-Slice Results For The Leading Candidates

### 4.1. `attention_only`

| Slice | Float | `attention_only` | Agreement |
| --- | ---: | ---: | ---: |
| `mnli-validation_matched-200-per-label.tsv` | `520/600 = 86.67%` | `523/600 = 87.17%` | `98.67%` |
| `mnli-validation_mismatched-200-per-label.tsv` | `517/600 = 86.17%` | `517/600 = 86.17%` | `98.67%` |
| `xnli-de-test-50-per-label.tsv` | `132/150 = 88.00%` | `132/150 = 88.00%` | `100.00%` |
| `xnli-en-test-50-per-label.tsv` | `134/150 = 89.33%` | `135/150 = 90.00%` | `99.33%` |
| `xnli-es-test-50-per-label.tsv` | `128/150 = 85.33%` | `130/150 = 86.67%` | `97.33%` |
| `xnli-fr-test-50-per-label.tsv` | `124/150 = 82.67%` | `125/150 = 83.33%` | `98.67%` |
| `xnli-zh-test-50-per-label.tsv` | `129/150 = 86.00%` | `127/150 = 84.67%` | `98.67%` |

Interpretation:

- still best on aggregate
- still strongest when judged purely by accuracy
- still shows the same main weakness as before: Chinese is below the float model

### 4.2. `attention_proj_only`

| Slice | Float | `attention_proj_only` | Agreement |
| --- | ---: | ---: | ---: |
| `mnli-validation_matched-200-per-label.tsv` | `520/600 = 86.67%` | `521/600 = 86.83%` | `99.50%` |
| `mnli-validation_mismatched-200-per-label.tsv` | `517/600 = 86.17%` | `517/600 = 86.17%` | `99.00%` |
| `xnli-de-test-50-per-label.tsv` | `132/150 = 88.00%` | `131/150 = 87.33%` | `97.33%` |
| `xnli-en-test-50-per-label.tsv` | `134/150 = 89.33%` | `135/150 = 90.00%` | `98.67%` |
| `xnli-es-test-50-per-label.tsv` | `128/150 = 85.33%` | `131/150 = 87.33%` | `98.00%` |
| `xnli-fr-test-50-per-label.tsv` | `124/150 = 82.67%` | `124/150 = 82.67%` | `97.33%` |
| `xnli-zh-test-50-per-label.tsv` | `129/150 = 86.00%` | `127/150 = 84.67%` | `98.67%` |

Interpretation:

- slightly below `attention_only` on aggregate
- slightly higher agreement with float
- still not enough to solve the Chinese regression

### 4.3. `attention_only_attention_output_layers_8_11_float`

This was the best “middle” variant:

- `1685/1950 = 86.41%`
- `+0.05` points vs float
- `98.77%` agreement

It is respectable, but it does not improve enough on either of the two front-runners to justify preferring it.

## 5. What Did Not Help

The most informative negative result is that dequantizing upper attention layers did not improve the benchmark enough to beat the simpler baselines.

In particular:

- `attention_only_layer_11_float`
- `attention_only_layers_10_11_float`
- `attention_proj_only_layer_11_float`
- `attention_proj_only_layers_10_11_float`

all increased agreement with float, but they all reduced aggregate accuracy relative to `attention_only`.

So the current evidence does not support the idea that “upper attention layers should be left float” is the next major gain.

## 6. HF Logit Probe Interpretation

The benchmark winner and the fidelity winner are not the same object.

On the original Angela/Macron probe used earlier in this investigation:

- `attention_only` had max abs logit delta vs HF about `0.107787`
- `attention_proj_only` had max abs logit delta vs HF about `0.038964`

So:

- `attention_only` wins on benchmark accuracy
- `attention_proj_only` is much closer to HF on at least that probe case

That is why the correct reading of this sweep is not “one candidate dominates everything.” The result is a tradeoff frontier.

## 7. Current Best Candidates

### 7.1. Best benchmark-accuracy candidate

- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`

Reason:

- best aggregate accuracy on the non-overlapping benchmark suite
- still above the float reference

### 7.2. Best fidelity-oriented candidate

- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

Reason:

- second-best aggregate accuracy
- slightly higher agreement with float than `attention_only`
- much smaller logit drift on the original HF probe pair

## 8. Reproduction

Run the attention-focused sweep:

```bash
.venv/bin/python tools/search-onnx-quantization-families.py \
  --candidate attention_only \
  --candidate attention_proj_only \
  --candidate attention_only_layer_11_float \
  --candidate attention_only_layers_10_11_float \
  --candidate attention_only_attention_output_layers_8_11_float \
  --candidate attention_proj_only_layer_11_float \
  --candidate attention_proj_only_layers_10_11_float \
  --summary-json benchmarks/nli/attention-sweep-summary.json \
  --summary-csv benchmarks/nli/attention-sweep-summary.csv \
  --resume
```

Inspect the benchmark summary:

```bash
sed -n '1,80p' benchmarks/nli/attention-sweep-summary.csv
```

Inspect the top candidates against HF logits:

```bash
.venv/bin/python tools/compare-hf-onnx-logits.py \
  --quantized-model models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx
```

```bash
.venv/bin/python tools/compare-hf-onnx-logits.py \
  --quantized-model models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx
```

## 9. Bottom Line

`QUANTIZATION_REPORT_2.md` found that quantizing attention while keeping FFN dense layers float was the right high-level direction.

This third report sharpens that conclusion:

- the follow-up attention sweep did not beat `attention_only` on benchmark accuracy
- `attention_only` remains the best accuracy-oriented quantized candidate
- `attention_proj_only` is the most credible fidelity-oriented alternative
- dequantizing upper attention layers increased agreement slightly but did not improve enough to win

At this point, the repo has two sensible baselines for any further work:

- `attention_only` for best benchmark accuracy
- `attention_proj_only` for closer match to the original float/HF model
