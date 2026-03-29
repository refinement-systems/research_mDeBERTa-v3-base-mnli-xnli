# Quantization Report 2

Scope: structured exclusion-family search performed after `QUANTIZATION_REPORT_1.md`, on the current workspace rooted at commit `0b4e8c07b320841f60b7e6bf12eb9be2adac6d5a` plus uncommitted tooling changes in `tools/search-onnx-quantization-families.py`.

This report covers:

1. the structured exclusion-family sweep,
2. the benchmark results,
3. what the results imply about this model,
4. how to reproduce or resume the search safely.

## 1. What Was Tried

This round moved away from one-off node lists and tested structured `MatMul` exclusion families derived from the ONNX layer naming patterns.

New tool:

- `tools/search-onnx-quantization-families.py`

It:

- derives exclusion families from `models/mdeberta/onnx/model.onnx`
- generates candidate ONNX files with `tools/quantize-onnx-model.py`
- benchmarks each candidate against the float reference with `builddir/nli-eval`
- writes ranked summaries to JSON and CSV
- now validates ONNX outputs before reuse
- now checkpoints after each candidate and supports `--resume`

Families tested:

- `attention_only`
- `ffn_only`
- `ffn_layers_8_11_float`
- `ffn_layers_6_11_float`
- `ffn_layers_4_11_float`
- `attention_output_layers_8_11_float`
- `layer_11_block_float`
- `layers_10_11_block_float`
- `baseline_dynamic_matmul`
- `current_best_reference`

The benchmark suite remained the non-overlapping set:

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

Structured-family ranking from `benchmarks/nli/family-search-summary.csv`:

| Rank | Candidate | Accuracy | Delta vs float | Agreement |
| --- | --- | ---: | ---: | ---: |
| 1 | `attention_only` | `1689/1950 = 86.62%` | `+0.26` points | `98.72%` |
| 2 | `ffn_layers_4_11_float` | `1674/1950 = 85.85%` | `-0.51` points | `96.46%` |
| 3 | `current_best_reference` | `1665/1950 = 85.38%` | `-0.97` points | `94.62%` |
| 4 | `ffn_layers_6_11_float` | `1627/1950 = 83.44%` | `-2.92` points | `90.97%` |
| 5 | `ffn_layers_8_11_float` | `1606/1950 = 82.36%` | `-4.00` points | `88.56%` |
| 6 | `layers_10_11_block_float` | `1316/1950 = 67.49%` | `-18.87` points | `68.62%` |
| 7 | `ffn_only` | `1299/1950 = 66.62%` | `-19.74` points | `67.85%` |
| 8 | `layer_11_block_float` | `1294/1950 = 66.36%` | `-20.00` points | `67.28%` |
| 9 | `baseline_dynamic_matmul` | `1271/1950 = 65.18%` | `-21.18` points | `65.90%` |
| 10 | `attention_output_layers_8_11_float` | `1264/1950 = 64.82%` | `-21.54` points | `65.49%` |

## 3. Most Important Finding

The clear winner is:

- `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`

Definition:

- quantize attention-side weight projections
- keep all FFN dense matmuls in float

This candidate is better than both:

- the previous best candidate `dynamic_qint8_matmul_exclude_suggested.onnx`
- the float ONNX reference on this benchmark slice

Aggregate comparison:

- float ONNX: `1684/1950 = 86.36%`
- `attention_only`: `1689/1950 = 86.62%`
- agreement with float: `1925/1950 = 98.72%`

This is the first quantized candidate in this repo that:

- exceeds the float reference on the aggregate suite
- keeps agreement with float above `98%`
- avoids the large multilingual regressions seen in earlier candidates

## 4. Per-Slice Readout For The Top Candidate

For `attention_only`:

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

- better on MNLI matched
- tied on MNLI mismatched
- tied on XNLI German
- better on XNLI English, Spanish, and French
- only worse on XNLI Chinese

So the gain is not coming from one lucky slice. It is broad enough to take seriously.

## 5. What The Sweep Tells Us

This sweep strongly suggests:

- this model is much more sensitive to quantizing FFN dense layers than to quantizing attention-side weight projections
- broad "leave late layers float" policies are not good enough
- the previous best candidate was improving through selective avoidance of FFN damage, but not in the cleanest way

The failed families are informative:

- `ffn_layers_8_11_float` and `ffn_layers_6_11_float` remained clearly worse than the float model
- `ffn_only` collapsed badly, which reinforces that quantizing FFN while leaving attention float is the wrong direction here
- `layer_11_block_float` and `layers_10_11_block_float` were also poor, so the useful distinction is not simply "top layers are sensitive"
- `attention_output_layers_8_11_float` was among the worst candidates, so leaving only the attention output-dense projections float is not the answer

The simplest current reading is:

- FFN dense quantization is the main source of behavioral damage
- attention-side projection quantization is comparatively safe

## 6. Reliability Note: Interrupted Runs

One interrupted search run was stopped and resumed via shell job control.

Observed failure mode:

- two candidate ONNX files became zero-byte placeholders:
  - `dynamic_qint8_matmul_ffn_layers_8_11_float.onnx`
  - `dynamic_qint8_matmul_ffn_layers_6_11_float.onnx`

Those files had to be regenerated from scratch.

The family-search tool was then hardened so that it now:

- validates candidate ONNX files before reuse
- treats zero-byte or empty-graph outputs as invalid
- regenerates invalid outputs automatically
- checkpoints summary JSON and CSV after each completed candidate
- supports `--resume` to continue from a partial summary
- supports `--resume --force` to rerun selected candidates intentionally

This means the current workflow is now restartable without trusting partial outputs blindly.

## 7. Reproduction

### 7.1. Run The Full Structured-Family Search

```bash
.venv/bin/python tools/search-onnx-quantization-families.py \
  --summary-json benchmarks/nli/family-search-summary.json \
  --summary-csv benchmarks/nli/family-search-summary.csv
```

### 7.2. Resume A Partial Search

```bash
.venv/bin/python tools/search-onnx-quantization-families.py \
  --summary-json benchmarks/nli/family-search-summary.json \
  --summary-csv benchmarks/nli/family-search-summary.csv \
  --resume
```

### 7.3. Rerun Selected Candidates Even If Cached

```bash
.venv/bin/python tools/search-onnx-quantization-families.py \
  --candidate ffn_layers_8_11_float \
  --candidate ffn_layers_6_11_float \
  --summary-json benchmarks/nli/family-search-summary.json \
  --summary-csv benchmarks/nli/family-search-summary.csv \
  --resume --force
```

### 7.4. Inspect The Best Candidate

```bash
.venv/bin/python tools/compare-hf-onnx-logits.py \
  --quantized-model models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx
```

## 8. Bottom Line

`QUANTIZATION_REPORT_1.md` established that the published quantized model was not faithful and that selective exclusions could help.

This second round improves that materially:

- the structured-family sweep found a better candidate than all previous quantized models in this repo
- that candidate is `dynamic_qint8_matmul_attention_only.onnx`
- it slightly outperforms the float ONNX reference on the current non-overlapping benchmark suite
- it matches the float model on `98.72%` of examples

At this point, `attention_only` should be treated as the new working quantized baseline for further experiments.
