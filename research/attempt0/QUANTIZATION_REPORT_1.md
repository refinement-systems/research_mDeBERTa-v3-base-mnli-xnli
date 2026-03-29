# Quantization Report 1

Scope: work through commit `22a3c05d157b056a609659b985169dc0fe257ea7` on `main`.

This report covers:

1. what quantization-related approaches were tried,
2. what worked well and what did not,
3. how to reproduce the benchmark runs and the best candidate built so far.

## 1. Approaches Tried Up To `22a3c05`

### 1.1. Establish the reference behavior first

Before tuning quantization, the repo added tooling to prove which model artifact was faithful to the original checkpoint:

- `b070d4f` added `tools/compare-hf-onnx-logits.py`
- the C++ CLI gained `--dump-logits`
- tokenizer parity work landed earlier, so the C++ preprocessing path matched Hugging Face closely enough not to be the main source of drift

Result:

- `models/mdeberta/onnx/model.onnx` tracks the Hugging Face/PyTorch reference very closely
- `models/mdeberta/onnx/model_quantized.onnx` is the outlier

Representative original failure case:

- premise: `Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU`
- hypothesis: `Emmanuel Macron is the President of France`
- HF logits: `[-3.285156, -0.886230, 3.873047]`
- float ONNX logits: `[-3.287450, -0.884211, 3.871160]`
- published quantized logits: `[-1.611540, 1.460460, 0.140061]`

So the float export is the correct baseline for all later quantization work.

### 1.2. Inspect the published quantized ONNX and generate fresh dynamic candidates

`a2c4aa1` added:

- `tools/inspect-onnx-model.py`
- `tools/quantize-onnx-model.py`

This was used to inspect the published quantized file and generate fresh ORT dynamic-quantization candidates from the float export:

- `dynamic_qint8_default.onnx`
- `dynamic_qint8_per_channel.onnx`
- `dynamic_qint8_matmul.onnx`
- `dynamic_qint8_matmul_per_channel.onnx`
- `dynamic_quint8_matmul.onnx`

Key finding:

- the published quantized model is an `onnx.quantize` rewrite using dynamic integer quantization
- naive re-quantization did not improve fidelity
- on the original Angela/Macron pair, those fresh candidates were still far from HF, with max abs logit drift roughly in the `3.9` to `4.1` range

### 1.3. Single-example activation-drift debugging and exclusion search

`3339662` added `tools/debug-onnx-quantization.py`.

That first version was a single-example debugger:

- it collected float and quantized activations
- ranked tensors by drift
- suggested `nodes_to_exclude` using the then-current quantized graph heuristic

This produced several experimental candidates:

- `dynamic_qint8_matmul_exclude_late.onnx`
- `dynamic_qint8_matmul_exclude_hotspots.onnx`
- `dynamic_qint8_matmul_exclude_l11_query.onnx`
- `dynamic_qint8_matmul_exclude_l11_query_layer5_dense.onnx`
- `dynamic_qint8_matmul_exclude_suggested.onnx`

Results:

- `exclude_late`, `exclude_hotspots`, `exclude_l11_query`, and `exclude_l11_query_layer5_dense` did not materially help
- `dynamic_qint8_matmul_exclude_suggested.onnx` was the first clearly better candidate

For the original Angela/Macron pair:

- published quantized model: wrong label (`neutral`), max abs logit drift vs HF `3.732986`
- `dynamic_qint8_matmul_exclude_suggested.onnx`: correct label (`contradiction`), max abs logit drift vs HF `0.456847`

This candidate became the best quantized model produced in this repo up to `22a3c05`.

### 1.4. Build real benchmark slices and benchmark wrapper

`b6cab0e` added `tools/download-nli-eval-slices.py`.

`e1ee73c` added `tools/benchmark-nli-models.sh`.

This replaced the earlier Topical Chat proxy with actual NLI benchmark slices:

- MNLI `validation_matched`
- MNLI `validation_mismatched`
- XNLI `test` for `de`, `en`, `es`, `fr`, `zh`

The benchmark report currently present in the working tree is:

- `benchmarks/nli/full-report.txt`

Important note:

- `benchmarks/` is generated output and is currently untracked
- the `100-per-label` MNLI slices overlap with the `200-per-label` MNLI slices, so they should not both be counted in a single aggregate score

### 1.5. Aggregate debugger changes and additional candidate families

`22a3c05` expanded the quantization tooling:

- `tools/debug-onnx-quantization.py` gained multi-TSV and directory sampling, aggregate drift metrics, JSON mode, and quantized-source detection
- `tools/quantize-onnx-model.py` gained a pre-processing path via `quant_pre_process`
- `tools/quantize-onnx-static.py` was added for static calibration experiments

Additional candidates tried at this stage:

- `dynamic_qint8_matmul_preprocessed.onnx`
- `static_qdq_qint8_matmul_smoke.onnx`
- `dynamic_qint8_matmul_exclude_layers9_11.onnx`
- `dynamic_qint8_matmul_exclude_all_qkv.onnx`

Results:

- `dynamic_qint8_matmul_preprocessed.onnx`: still wrong on the original pair, max abs logit drift vs HF about `3.95`
- `static_qdq_qint8_matmul_smoke.onnx`: still wrong on the original pair, max abs logit drift vs HF about `4.22`
- `dynamic_qint8_matmul_exclude_all_qkv.onnx`: still wrong on the original pair, max abs logit drift vs HF about `3.80`
- `dynamic_qint8_matmul_exclude_layers9_11.onnx`: improved the original pair to the correct label with max abs logit drift about `0.70`, but failed badly on the first real benchmark slice and was abandoned

The only quantized candidate that held up reasonably under both the original failure case and benchmarking was still `dynamic_qint8_matmul_exclude_suggested.onnx`.

## 2. What Worked Well, And What Did Not

### 2.1. Worked well

#### Tooling

- `tools/compare-hf-onnx-logits.py` worked well for proving whether a candidate was close to HF or not.
- `tools/download-nli-eval-slices.py` and `tools/benchmark-nli-models.sh` worked well for moving from anecdotal single-pair checks to reproducible MNLI/XNLI evaluation.
- The original single-example debugger from `3339662` worked well enough to find a noticeably better dynamic quantized candidate than either the published quantized model or the naive ORT sweeps.

#### Model artifacts

- `models/mdeberta/onnx/model.onnx` worked well as the reference ONNX export. It is the faithful export.
- `models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx` is the best quantized candidate found up to this commit.

For the benchmark file currently in `benchmarks/nli/full-report.txt`, the per-slice results for `dynamic_qint8_matmul_exclude_suggested.onnx` are:

| Slice | Float accuracy | Suggested candidate accuracy | Agreement |
| --- | ---: | ---: | ---: |
| `mnli-validation_matched-100-per-label.tsv` | `259/300 = 86.33%` | `261/300 = 87.00%` | `283/300 = 94.33%` |
| `mnli-validation_matched-200-per-label.tsv` | `520/600 = 86.67%` | `523/600 = 87.17%` | `571/600 = 95.17%` |
| `mnli-validation_mismatched-100-per-label.tsv` | `260/300 = 86.67%` | `252/300 = 84.00%` | `286/300 = 95.33%` |
| `mnli-validation_mismatched-200-per-label.tsv` | `517/600 = 86.17%` | `507/600 = 84.50%` | `573/600 = 95.50%` |
| `xnli-de-test-50-per-label.tsv` | `132/150 = 88.00%` | `131/150 = 87.33%` | `140/150 = 93.33%` |
| `xnli-en-test-50-per-label.tsv` | `134/150 = 89.33%` | `131/150 = 87.33%` | `143/150 = 95.33%` |
| `xnli-es-test-50-per-label.tsv` | `128/150 = 85.33%` | `127/150 = 84.67%` | `140/150 = 93.33%` |
| `xnli-fr-test-50-per-label.tsv` | `124/150 = 82.67%` | `121/150 = 80.67%` | `139/150 = 92.67%` |
| `xnli-zh-test-50-per-label.tsv` | `129/150 = 86.00%` | `125/150 = 83.33%` | `139/150 = 92.67%` |

Using the non-overlapping suite only:

- `mnli-validation_matched-200-per-label.tsv`
- `mnli-validation_mismatched-200-per-label.tsv`
- `xnli-{de,en,es,fr,zh}-test-50-per-label.tsv`

the aggregate result is:

- float ONNX: `1684/1950 = 86.36%`
- suggested quantized candidate: `1665/1950 = 85.38%`
- model agreement: `1845/1950 = 94.62%`

That is not good enough to call it a faithful drop-in replacement, but it is much better than the published quantized export and better than every later candidate tried in this branch of work.

### 2.2. Did not work well

#### Published or naive quantized models

- `models/mdeberta/onnx/model_quantized.onnx` is not faithful to HF.
- fresh ORT dynamic candidates generated without exclusions were also poor.

#### Overly simple exclusion heuristics

- excluding only late attention matmuls or a couple of manually chosen nodes did not help enough
- excluding whole late-layer blocks (`dynamic_qint8_matmul_exclude_layers9_11.onnx`) fixed the original anecdotal pair but over-corrected and damaged benchmark performance
- excluding all Q/K/V projections (`dynamic_qint8_matmul_exclude_all_qkv.onnx`) still left the original anecdotal pair in the bad `neutral` regime

#### Pre-processing and static calibration, as tried here

- dynamic quantization with pre-processing did not improve the original failure case
- the first static QDQ smoke candidate did not improve the original failure case either

At this commit, there is no benchmark-backed evidence that either of those newer directions is better than `dynamic_qint8_matmul_exclude_suggested.onnx`.

## 3. Benchmark And Candidate Reproduction

### 3.1. Environment

These commands assume:

- repo root is the current working directory
- `.venv` exists
- the Python packages needed by the tooling are installed in `.venv`

If needed:

```bash
.venv/bin/python -m pip install --upgrade pip \
  onnx onnxruntime transformers sentencepiece torch numpy
```

Build the local binaries:

```bash
tools/build.sh --target nli nli-eval
```

### 3.2. Download model files and reference weights

```bash
tools/download-mdeberta-v3-base.sh --force
tools/download-mdeberta-v3-base.sh --tokenizer-assets --reference-weights --force
```

### 3.3. Generate the benchmark slices

The benchmark tooling uses Hugging Face datasets-server and produces deterministic balanced slices with the default seed.

```bash
tools/download-nli-eval-slices.py \
  --mnli-per-label 200 \
  --xnli-per-label 50 \
  --xnli-language en \
  --xnli-language de \
  --xnli-language es \
  --xnli-language fr \
  --xnli-language zh \
  --force
```

This creates TSVs under `benchmarks/nli/`.

### 3.4. Reproduce the current benchmark report for the best candidate

The best candidate found up to this commit is:

- `models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx`

To benchmark it against the float reference:

```bash
tools/benchmark-nli-models.sh \
  --compare-model models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx \
  --report benchmarks/nli/full-report.txt
```

That reproduces the style of report discussed above.

If you want the simpler manual loop instead of the wrapper:

```bash
for tsv in benchmarks/nli/*.tsv; do
  echo "== $tsv =="
  builddir/nli-eval -b cpu \
    --model models/mdeberta/onnx/model.onnx \
    --compare-model models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx \
    "$tsv"
  echo
done | tee benchmarks/nli/full-report.txt
```

### 3.5. Reproduce the original “suggested exclusions” candidate exactly

`tools/debug-onnx-quantization.py` at `HEAD` is not the same tool that produced the original suggested candidate. It was expanded in `22a3c05` to support aggregate and multi-source analysis, and its exclusion heuristic changed.

To preserve the original behavior from `3339662`, this repo now includes:

- `tools/legacy-debug-onnx-quantization-3339662.sh`

This wrapper pulls the old script from git history and runs it with the current `.venv`.

#### Step 1: build the base MatMul-only dynamic candidate

```bash
.venv/bin/python tools/quantize-onnx-model.py \
  --preset single \
  --output models/mdeberta/onnx/candidates/dynamic_qint8_matmul.onnx \
  --op-type MatMul \
  --force
```

#### Step 2: run the legacy debugger on the default Angela/Macron pair

```bash
tools/legacy-debug-onnx-quantization-3339662.sh \
  --quantized-model models/mdeberta/onnx/candidates/dynamic_qint8_matmul.onnx \
  --op-type MatMul \
  --op-type Add \
  --top 25 \
  --suggest-exclusions 10
```

The run that produced the best candidate at this stage yielded this exclusion CSV:

```text
/deberta/encoder/layer.11/attention/self/query_proj/MatMul,/deberta/encoder/layer.5/output/dense/MatMul,/deberta/encoder/layer.5/intermediate/dense/MatMul,/deberta/encoder/layer.6/output/dense/MatMul,/deberta/encoder/layer.6/intermediate/dense/MatMul,/deberta/encoder/layer.9/output/dense/MatMul,/deberta/encoder/layer.11/output/dense/MatMul,/deberta/encoder/layer.9/intermediate/dense/MatMul,/deberta/encoder/layer.4/output/dense/MatMul,/deberta/encoder/layer.11/attention/output/dense/MatMul
```

#### Step 3: build the suggested candidate

```bash
.venv/bin/python tools/quantize-onnx-model.py \
  --preset single \
  --output models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx \
  --op-type MatMul \
  --nodes-to-exclude '/deberta/encoder/layer.11/attention/self/query_proj/MatMul,/deberta/encoder/layer.5/output/dense/MatMul,/deberta/encoder/layer.5/intermediate/dense/MatMul,/deberta/encoder/layer.6/output/dense/MatMul,/deberta/encoder/layer.6/intermediate/dense/MatMul,/deberta/encoder/layer.9/output/dense/MatMul,/deberta/encoder/layer.11/output/dense/MatMul,/deberta/encoder/layer.9/intermediate/dense/MatMul,/deberta/encoder/layer.4/output/dense/MatMul,/deberta/encoder/layer.11/attention/output/dense/MatMul' \
  --force
```

#### Step 4: spot-check the original failure case

```bash
.venv/bin/python tools/compare-hf-onnx-logits.py \
  --quantized-model models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx \
  --premise "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU" \
  --hypothesis "Emmanuel Macron is the President of France"
```

Expected qualitative result:

- float ONNX matches HF closely
- the suggested candidate stays on the correct `contradiction` label for this pair
- max abs logit drift is much smaller than either the published quantized model or the naive ORT dynamic candidates

### 3.6. Reproduce the later failed experiments from `22a3c05`

These were useful to rule out directions, but they are not recommended candidates.

Dynamic preprocessed candidate:

```bash
.venv/bin/python tools/quantize-onnx-model.py \
  --preset single \
  --preprocess \
  --output models/mdeberta/onnx/candidates/dynamic_qint8_matmul_preprocessed.onnx \
  --op-type MatMul \
  --force
```

Static smoke candidate:

```bash
.venv/bin/python tools/quantize-onnx-static.py \
  --output models/mdeberta/onnx/candidates/static_qdq_qint8_matmul_smoke.onnx \
  --calibration-tsv benchmarks/nli/mnli-validation_matched-100-per-label.tsv \
  --calibration-tsv benchmarks/nli/xnli-en-test-50-per-label.tsv \
  --max-examples-per-source 8 \
  --op-type MatMul \
  --force
```

Late-layer exclusion candidate:

```bash
.venv/bin/python tools/quantize-onnx-model.py \
  --preset single \
  --output models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_layers9_11.onnx \
  --op-type MatMul \
  --nodes-to-exclude '/deberta/encoder/layer.9/attention/self/query_proj/MatMul,/deberta/encoder/layer.9/attention/self/key_proj/MatMul,/deberta/encoder/layer.9/attention/self/value_proj/MatMul,/deberta/encoder/layer.9/attention/output/dense/MatMul,/deberta/encoder/layer.9/intermediate/dense/MatMul,/deberta/encoder/layer.9/output/dense/MatMul,/deberta/encoder/layer.10/attention/self/query_proj/MatMul,/deberta/encoder/layer.10/attention/self/key_proj/MatMul,/deberta/encoder/layer.10/attention/self/value_proj/MatMul,/deberta/encoder/layer.10/attention/output/dense/MatMul,/deberta/encoder/layer.10/intermediate/dense/MatMul,/deberta/encoder/layer.10/output/dense/MatMul,/deberta/encoder/layer.11/attention/self/query_proj/MatMul,/deberta/encoder/layer.11/attention/self/key_proj/MatMul,/deberta/encoder/layer.11/attention/self/value_proj/MatMul,/deberta/encoder/layer.11/attention/output/dense/MatMul,/deberta/encoder/layer.11/intermediate/dense/MatMul,/deberta/encoder/layer.11/output/dense/MatMul' \
  --force
```

This one is the clearest example of an anecdotal fix that did not generalize:

- it repaired the original Angela/Macron pair
- but it dropped to `243/300 = 81.00%` on `mnli-validation_matched-100-per-label.tsv`

## Bottom Line

Up to `22a3c05`, the repo established a clear reference and a reproducible benchmark loop.

The strongest conclusion is:

- `model.onnx` is the faithful export
- `model_quantized.onnx` is not
- the best regenerated quantized candidate so far is `dynamic_qint8_matmul_exclude_suggested.onnx`
- even that candidate is still materially different from the float/HF reference and should be treated as experimental, not a drop-in replacement
