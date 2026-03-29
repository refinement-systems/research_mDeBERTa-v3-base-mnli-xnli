# Quantization Report 6

Scope: combined quality and runtime benchmark results after `QUANTIZATION_REPORT_5.md`, using:

- `tools/benchmark-hf-onnx-models.py`
- `tools/build-hf-probe-set.py`
- `tools/build-hf-core-probe.py`
- `tools/benchmark-nli-runtime.py`
- `benchmarks/nli/hf-finalist-full.csv`
- `benchmarks/nli/hf-probe-benchmark.csv`
- `benchmarks/nli/hf-core-probe-benchmark.csv`
- `benchmarks/nli/runtime-cpu-core-probe.csv`
- `benchmarks/nli/runtime-coreml-core-probe.csv`

This report covers:

1. the finalized benchmark ladder,
2. the current quality frontier,
3. the first shipped-runtime measurements,
4. what the combined evidence now says about the default model choice.

## 1. Why A Sixth Report Was Needed

`QUANTIZATION_REPORT_5.md` froze the quality side of the benchmark gate:

- full-suite benchmark
- hard probe
- fixed finalists

But it still left one unresolved question:

- do the quantized finalists buy enough runtime or size improvement to justify their behavioral drift?

To answer that, the repo added:

- a smaller deterministic core probe:
  - `benchmarks/nli/hf-core-probe.tsv`
- a runtime benchmark for the actual C++ CLI path:
  - `tools/benchmark-nli-runtime.py`

This sixth report is the first point where quality and runtime evidence can be read together.

## 2. The Benchmark Ladder Is Now Stable

The repo now has three quality tiers and one runtime tier.

### 2.1. Quality Tier 1: Core Probe

- `benchmarks/nli/hf-core-probe.tsv`
- `25` examples
- preserves all `19` finalist label-difference cases
- preserves all benchmark sources
- preserves all XNLI languages, including Chinese

Purpose:

- fast stress screen for future candidate pruning

### 2.2. Quality Tier 2: Hard Probe

- `benchmarks/nli/hf-probe-set.tsv`
- `61` examples

Purpose:

- concentrated regression detector for disagreement and high-drift cases

### 2.3. Quality Tier 3: Full Suite

- `mnli-validation_matched-200-per-label.tsv`
- `mnli-validation_mismatched-200-per-label.tsv`
- `xnli-{de,en,es,fr,zh}-test-50-per-label.tsv`

Total:

- `1950` examples

Purpose:

- final quality ranking

### 2.4. Runtime Tier

Current runtime benchmark tool:

- `tools/benchmark-nli-runtime.py`

Current completed runtime outputs:

- `benchmarks/nli/runtime-cpu-core-probe.csv`
- `benchmarks/nli/runtime-coreml-core-probe.csv`

Important note:

- runtime load cost is measured as true cold model construction in `builddir/nli`
- warm latency is measured from repeated inference calls after model construction inside the same process

## 3. Final Quality Frontier

### 3.1. Full-Suite Results

From `benchmarks/nli/hf-finalist-full.csv`:

| Candidate | Accuracy | HF Agreement | Fixed Float Errors | New Errors vs Float | `xnli-zh` Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention_only` | `1690/1950 = 86.67%` | `1926/1950 = 98.77%` | `13` | `7` | `85.33%` |
| `attention_proj_only` | `1687/1950 = 86.51%` | `1927/1950 = 98.82%` | `10` | `7` | `85.33%` |
| `float` | `1684/1950 = 86.36%` | `1950/1950 = 100.00%` | `0` | `0` | `86.00%` |

Reading:

- `attention_only` remains the best benchmark-accuracy quantized candidate
- `attention_proj_only` remains the slightly better HF-fidelity quantized candidate
- float remains the exact HF-label reference

### 3.2. Hard-Probe Results

From `benchmarks/nli/hf-probe-benchmark.csv`:

| Candidate | Accuracy | HF Agreement | Fixed Float Errors | New Errors vs Float | `xnli-zh` Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention_only` | `34/61 = 55.74%` | `37/61 = 60.66%` | `13` | `7` | `37.50%` |
| `attention_proj_only` | `31/61 = 50.82%` | `38/61 = 62.30%` | `10` | `7` | `37.50%` |
| `float` | `28/61 = 45.90%` | `61/61 = 100.00%` | `0` | `0` | `50.00%` |

Reading:

- the hard probe preserves the same tradeoff frontier
- `attention_only` still wins on gold-label accuracy
- `attention_proj_only` still wins slightly on HF agreement
- Chinese remains a weak spot for both quantized finalists

### 3.3. Core-Probe Results

From `benchmarks/nli/hf-core-probe-benchmark.csv`:

| Candidate | Accuracy | HF Agreement | Fixed Float Errors | New Errors vs Float | `xnli-zh` Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention_only` | `13/25 = 52.00%` | `14/25 = 56.00%` | `6` | `4` | `0.00%` |
| `attention_proj_only` | `10/25 = 40.00%` | `15/25 = 60.00%` | `3` | `4` | `0.00%` |
| `float` | `11/25 = 44.00%` | `25/25 = 100.00%` | `0` | `0` | `100.00%` |

Reading:

- the core probe successfully compresses the frontier without changing it
- `attention_only` remains the best accuracy screen candidate
- `attention_proj_only` remains the better HF-fidelity screen candidate
- the single Chinese core-probe example is a useful tripwire and currently fails under both quantized finalists

## 4. Runtime Results

### 4.1. CPU Runtime On The Full Core Probe

From `benchmarks/nli/runtime-cpu-core-probe.csv`:

| Candidate | Size | Load Median | Warm Median | Warm P95 |
| --- | ---: | ---: | ---: | ---: |
| `attention_proj_only` | `1056.31 MB` | `4349.300 ms` | `471.889 ms` | `736.860 ms` |
| `attention_only` | `1036.07 MB` | `4295.920 ms` | `472.244 ms` | `744.704 ms` |
| `float` | `1064.41 MB` | `4121.520 ms` | `474.279 ms` | `741.968 ms` |

Reading:

- warm latency is effectively the same across all three models
- float is actually the fastest to cold-load on CPU
- quantization only reduces disk size modestly:
  - `attention_only` saves about `28 MB`
  - `attention_proj_only` saves about `8 MB`

That means the current quantized finalists do **not** buy a meaningful CPU runtime advantage.

### 4.2. CoreML Runtime On The Full Core Probe

From `benchmarks/nli/runtime-coreml-core-probe.csv`:

| Candidate | Size | Load Median | Warm Median | Warm P95 |
| --- | ---: | ---: | ---: | ---: |
| `attention_proj_only` | `1056.31 MB` | `17293.000 ms` | `489.929 ms` | `770.095 ms` |
| `float` | `1064.41 MB` | `16611.600 ms` | `492.284 ms` | `760.965 ms` |
| `attention_only` | `1036.07 MB` | `17387.700 ms` | `494.356 ms` | `759.103 ms` |

Reading:

- warm latency is effectively the same across all three models on CoreML as well
- float is still the fastest to cold-load
- the small warm-latency edge for `attention_proj_only` is only a few milliseconds and is not operationally decisive
- quantization still does not buy a meaningful runtime advantage in the current CoreML path either

## 5. Combined Interpretation

The combined quality and runtime evidence now says:

### 5.1. `attention_only`

Strength:

- best quality result among quantized candidates

Weakness:

- slightly worse fidelity than `attention_proj_only`
- no meaningful CPU or CoreML runtime advantage over float
- Chinese still regresses

### 5.2. `attention_proj_only`

Strength:

- slightly better fidelity than `attention_only`

Weakness:

- slightly worse accuracy than `attention_only`
- still not faithful enough to match float/HF
- no meaningful CPU or CoreML runtime advantage over float
- Chinese still regresses

### 5.3. `float`

Strength:

- exact HF-label match on the full suite
- fastest cold load on both CPU and CoreML
- only marginally slower warm latency than the quantized finalists

Weakness:

- largest file size

## 6. Current Best Overall Default

If the decision were purely "best quantized model", the answer is still:

- accuracy-first:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`
- fidelity-first:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

But if the decision is "best default model for this repo today", the evidence now points back to:

- `models/mdeberta/onnx/model.onnx`

Why:

- it is the faithful ONNX representation of the HF reference
- the CPU and CoreML runtime cost relative to the quantized finalists is negligible in practice
- the quantized finalists do not currently deliver enough speed benefit to justify their drift

## 7. What Changed Relative To Report 5

`QUANTIZATION_REPORT_5.md` stopped after quality and fidelity benchmarking.

This report adds the missing operational point:

- the current quantized finalists are not obviously faster in the shipped CPU or CoreML paths

That materially changes the decision context.

Before the runtime data, the open question was:

- which quantized finalist should be preferred?

After the runtime data, the more credible question is:

- should a quantized finalist be preferred at all?

Right now, the answer is:

- not as the default model

## 8. Reproduction

### 8.1. Full Quality Ladder

```bash
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --max-examples-per-source 0 \
  --sample-mode first \
  --summary-json benchmarks/nli/hf-finalist-full.json \
  --summary-csv benchmarks/nli/hf-finalist-full.csv

.venv/bin/python tools/build-hf-probe-set.py \
  --include-finalist-label-diffs \
  --include-float-top-drift 10

.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --tsv benchmarks/nli/hf-probe-set.tsv \
  --sample-mode first \
  --max-examples-per-source 0 \
  --summary-json benchmarks/nli/hf-probe-benchmark.json \
  --summary-csv benchmarks/nli/hf-probe-benchmark.csv

python3 tools/build-hf-core-probe.py

.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --tsv benchmarks/nli/hf-core-probe.tsv \
  --sample-mode first \
  --max-examples-per-source 0 \
  --summary-json benchmarks/nli/hf-core-probe-benchmark.json \
  --summary-csv benchmarks/nli/hf-core-probe-benchmark.csv
```

### 8.2. CPU Runtime On The Core Probe

```bash
python3 tools/benchmark-nli-runtime.py \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend cpu \
  --summary-json benchmarks/nli/runtime-cpu-core-probe.json \
  --summary-csv benchmarks/nli/runtime-cpu-core-probe.csv
```

### 8.3. CPU/CoreML Snapshot

```bash
python3 tools/benchmark-nli-runtime.py \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend coreml \
  --summary-json benchmarks/nli/runtime-coreml-core-probe.json \
  --summary-csv benchmarks/nli/runtime-coreml-core-probe.csv
```

## 9. Bottom Line

The repo now has enough evidence to say all of the following with confidence:

- the quality frontier is stable
- `attention_only` is the best quantized model on accuracy
- `attention_proj_only` is the best quantized model on fidelity
- neither quantized finalist currently delivers a meaningful CPU or CoreML runtime advantage
- float remains the most defensible default overall
