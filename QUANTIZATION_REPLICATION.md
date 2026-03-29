# Quantization Replication

This is the canonical way to regenerate the artifacts referenced by the `QUANTIZATION_*.md` files without relying on a preexisting untracked `benchmarks/` directory.

## What Changed Since The Reports Were Written

- `benchmarks/nli/` is generated output. None of the reports require the current untracked `benchmarks/` tree.
- `tools/debug-onnx-quantization.py` changed after commit `3339662`. Use `tools/legacy-debug-onnx-quantization-3339662.sh` for the original single-example exclusion search from `QUANTIZATION_REPORT_1.md`.
- `tools/search-onnx-quantization-families.py` changed after commit `8369b92`. Use `tools/legacy-search-onnx-quantization-families-8369b92.sh` for the exact default family sweep from `QUANTIZATION_REPORT_2.md`.
- `QUANTIZATION_REPORT_1.md` implicitly depends on both `100-per-label` and `200-per-label` MNLI slices. Its bare `tools/benchmark-nli-models.sh` example is only exact if the input directory contains only the original per-label benchmark TSVs. Once later probe TSVs exist, use `--pattern '*-per-label.tsv'`.
- Persistent runtime benchmarks from `QUANTIZATION_REPORT_6.md` and `QUANTIZATION_REPORT_7.md` require `builddir/nli-runtime-bench` in addition to `builddir/nli`.

## Environment

These steps assume:

- repo root is the current working directory
- the checkout includes git history for commits `3339662` and `8369b92`
- network access is available for Hugging Face model downloads and datasets-server benchmark downloads
- CoreML runs are only attempted on a machine that supports the CoreML execution provider

If needed, create the Python environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip \
  onnx onnxruntime transformers sentencepiece torch numpy
```

Configure and build the binaries:

```bash
tools/setup.sh
tools/build.sh --target nli nli-eval nli-runtime-bench
```

Download the model artifacts and reference weights:

```bash
tools/download-mdeberta-v3-base.sh --force
tools/download-mdeberta-v3-base.sh --tokenizer-assets --reference-weights --force
```

## Rebuild The Benchmark TSVs

Create the original `QUANTIZATION_REPORT_1.md` per-label slices:

```bash
tools/download-nli-eval-slices.py \
  --mnli-per-label 100 \
  --xnli-per-label 50 \
  --xnli-language en \
  --xnli-language de \
  --xnli-language es \
  --xnli-language fr \
  --xnli-language zh \
  --force
```

Create the non-overlapping suite used by `QUANTIZATION_REPORT_2.md` onward:

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

After those two commands, `benchmarks/nli/` contains every TSV needed by the reports:

- `mnli-validation_matched-100-per-label.tsv`
- `mnli-validation_mismatched-100-per-label.tsv`
- `mnli-validation_matched-200-per-label.tsv`
- `mnli-validation_mismatched-200-per-label.tsv`
- `xnli-{de,en,es,fr,zh}-test-50-per-label.tsv`

## Report 1

Rebuild the base MatMul-only candidate:

```bash
.venv/bin/python tools/quantize-onnx-model.py \
  --preset single \
  --output models/mdeberta/onnx/candidates/dynamic_qint8_matmul.onnx \
  --op-type MatMul \
  --force
```

Reproduce the original single-example exclusion suggestion exactly:

```bash
tools/legacy-debug-onnx-quantization-3339662.sh \
  --quantized-model models/mdeberta/onnx/candidates/dynamic_qint8_matmul.onnx \
  --op-type MatMul \
  --op-type Add \
  --top 25 \
  --suggest-exclusions 10
```

Build the historical suggested candidate:

```bash
.venv/bin/python tools/quantize-onnx-model.py \
  --preset single \
  --output models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx \
  --op-type MatMul \
  --nodes-to-exclude '/deberta/encoder/layer.11/attention/self/query_proj/MatMul,/deberta/encoder/layer.5/output/dense/MatMul,/deberta/encoder/layer.5/intermediate/dense/MatMul,/deberta/encoder/layer.6/output/dense/MatMul,/deberta/encoder/layer.6/intermediate/dense/MatMul,/deberta/encoder/layer.9/output/dense/MatMul,/deberta/encoder/layer.11/output/dense/MatMul,/deberta/encoder/layer.9/intermediate/dense/MatMul,/deberta/encoder/layer.4/output/dense/MatMul,/deberta/encoder/layer.11/attention/output/dense/MatMul' \
  --force
```

Rebuild the `full-report.txt` discussed in `QUANTIZATION_REPORT_1.md`:

```bash
tools/benchmark-nli-models.sh \
  --pattern '*-per-label.tsv' \
  --compare-model models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx \
  --report benchmarks/nli/full-report.txt
```

Reproduce the later failed experiments from the same report:

```bash
.venv/bin/python tools/quantize-onnx-model.py \
  --preset single \
  --preprocess \
  --output models/mdeberta/onnx/candidates/dynamic_qint8_matmul_preprocessed.onnx \
  --op-type MatMul \
  --force

.venv/bin/python tools/quantize-onnx-static.py \
  --output models/mdeberta/onnx/candidates/static_qdq_qint8_matmul_smoke.onnx \
  --calibration-tsv benchmarks/nli/mnli-validation_matched-100-per-label.tsv \
  --calibration-tsv benchmarks/nli/xnli-en-test-50-per-label.tsv \
  --max-examples-per-source 8 \
  --op-type MatMul \
  --force

.venv/bin/python tools/quantize-onnx-model.py \
  --preset single \
  --output models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_layers9_11.onnx \
  --op-type MatMul \
  --nodes-to-exclude '/deberta/encoder/layer.9/attention/self/query_proj/MatMul,/deberta/encoder/layer.9/attention/self/key_proj/MatMul,/deberta/encoder/layer.9/attention/self/value_proj/MatMul,/deberta/encoder/layer.9/attention/output/dense/MatMul,/deberta/encoder/layer.9/intermediate/dense/MatMul,/deberta/encoder/layer.9/output/dense/MatMul,/deberta/encoder/layer.10/attention/self/query_proj/MatMul,/deberta/encoder/layer.10/attention/self/key_proj/MatMul,/deberta/encoder/layer.10/attention/self/value_proj/MatMul,/deberta/encoder/layer.10/attention/output/dense/MatMul,/deberta/encoder/layer.10/intermediate/dense/MatMul,/deberta/encoder/layer.10/output/dense/MatMul,/deberta/encoder/layer.11/attention/self/query_proj/MatMul,/deberta/encoder/layer.11/attention/self/key_proj/MatMul,/deberta/encoder/layer.11/attention/self/value_proj/MatMul,/deberta/encoder/layer.11/attention/output/dense/MatMul,/deberta/encoder/layer.11/intermediate/dense/MatMul,/deberta/encoder/layer.11/output/dense/MatMul' \
  --force
```

## Report 2

Run the exact historical structured family sweep:

```bash
tools/legacy-search-onnx-quantization-families-8369b92.sh \
  --summary-json benchmarks/nli/family-search-summary.json \
  --summary-csv benchmarks/nli/family-search-summary.csv
```

That wrapper matters because the current `tools/search-onnx-quantization-families.py` includes later attention-follow-up candidates by default.

## Report 3

Run the attention follow-up sweep with the current tool and explicit candidate selection:

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

## Reports 4 And 5

Rebuild the full HF-vs-ONNX finalist benchmark:

```bash
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --max-examples-per-source 0 \
  --sample-mode first \
  --summary-json benchmarks/nli/hf-finalist-full.json \
  --summary-csv benchmarks/nli/hf-finalist-full.csv
```

Optional sampled diagnostic run from `QUANTIZATION_REPORT_4.md`:

```bash
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --max-examples-per-source 10 \
  --sample-mode random \
  --seed 0 \
  --show-slices \
  --summary-json benchmarks/nli/hf-finalist-benchmark.json \
  --summary-csv benchmarks/nli/hf-finalist-benchmark.csv
```

Build the hard probe and benchmark it:

```bash
.venv/bin/python tools/build-hf-probe-set.py \
  --include-finalist-label-diffs \
  --include-float-top-drift 10

.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --tsv benchmarks/nli/hf-probe-set.tsv \
  --sample-mode first \
  --max-examples-per-source 0 \
  --summary-json benchmarks/nli/hf-probe-benchmark.json \
  --summary-csv benchmarks/nli/hf-probe-benchmark.csv
```

## Report 6

Build the core probe and benchmark it:

```bash
.venv/bin/python tools/build-hf-core-probe.py

.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --tsv benchmarks/nli/hf-core-probe.tsv \
  --sample-mode first \
  --max-examples-per-source 0 \
  --summary-json benchmarks/nli/hf-core-probe-benchmark.json \
  --summary-csv benchmarks/nli/hf-core-probe-benchmark.csv
```

Cold-start runtime:

```bash
.venv/bin/python tools/benchmark-nli-runtime.py \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend cpu \
  --summary-json benchmarks/nli/runtime-cpu-core-probe.json \
  --summary-csv benchmarks/nli/runtime-cpu-core-probe.csv

.venv/bin/python tools/benchmark-nli-runtime.py \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend coreml \
  --summary-json benchmarks/nli/runtime-coreml-core-probe.json \
  --summary-csv benchmarks/nli/runtime-coreml-core-probe.csv
```

Persistent-session runtime:

```bash
.venv/bin/python tools/benchmark-nli-runtime.py \
  --mode persistent \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend cpu \
  --summary-json benchmarks/nli/runtime-cpu-core-probe-persistent.json \
  --summary-csv benchmarks/nli/runtime-cpu-core-probe-persistent.csv

.venv/bin/python tools/benchmark-nli-runtime.py \
  --mode persistent \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend coreml \
  --summary-json benchmarks/nli/runtime-coreml-core-probe-persistent.json \
  --summary-csv benchmarks/nli/runtime-coreml-core-probe-persistent.csv
```

## Report 7 And Recommendation

Refresh the persistent runtime CSVs with RSS data and rebuild the generated dashboard and recommendation:

```bash
.venv/bin/python tools/benchmark-nli-runtime.py \
  --mode persistent \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend cpu \
  --summary-json benchmarks/nli/runtime-cpu-core-probe-persistent.json \
  --summary-csv benchmarks/nli/runtime-cpu-core-probe-persistent.csv

.venv/bin/python tools/benchmark-nli-runtime.py \
  --mode persistent \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend coreml \
  --summary-json benchmarks/nli/runtime-coreml-core-probe-persistent.json \
  --summary-csv benchmarks/nli/runtime-coreml-core-probe-persistent.csv

.venv/bin/python tools/build-quantization-dashboard.py
```

That rebuilds:

- `benchmarks/nli/quantization-dashboard.json`
- `benchmarks/nli/quantization-dashboard.csv`
- `benchmarks/nli/quantization-dashboard.md`
- `QUANTIZATION_RECOMMENDATION_1.md`

## Expected Reproducibility

These outputs should be deterministic once the model files and TSVs are fixed:

- benchmark TSV contents
- candidate search rankings
- HF agreement, float agreement, accuracy, and logit-drift summaries
- hard-probe and core-probe membership

These outputs are environment-sensitive:

- CPU/CoreML load time
- warm latency
- RSS and peak RSS

For reports 6 and 7, match the qualitative result first:

- float should remain the fidelity reference
- `attention_only` should remain the best quantized accuracy candidate
- `attention_proj_only` should remain the best quantized fidelity candidate
- persistent-session quantization gains should remain modest rather than transformational
