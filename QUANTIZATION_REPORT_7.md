# Quantization Report 7

Scope: memory-aware follow-up after `QUANTIZATION_REPORT_6.md`, covering:

- in-process resident-memory sampling added to the runtime benchmarks
- refreshed persistent CPU and CoreML runtime benchmarks
- the consolidated dashboard and recommendation outputs

Primary inputs:

- `tools/benchmark-nli-runtime.py`
- `tools/build-quantization-dashboard.py`
- `benchmarks/nli/runtime-cpu-core-probe-persistent.csv`
- `benchmarks/nli/runtime-coreml-core-probe-persistent.csv`
- `benchmarks/nli/quantization-dashboard.md`
- `QUANTIZATION_RECOMMENDATION_1.md`

This report answers one remaining operational question:

- does memory usage make the quantized finalists more attractive than the float model?

## 1. Why A Seventh Report Was Needed

`QUANTIZATION_REPORT_6.md` established the main quality/runtime picture:

- float is the exact HF-faithful reference
- `attention_only` is the best quantized accuracy candidate
- `attention_proj_only` is the best quantized fidelity candidate
- quantized finalists have only a modest persistent-session warm-latency edge

But one operational dimension was still missing:

- resident memory

That mattered because the runtime gains were already small. If memory had improved materially under quantization, the shipping decision might still have changed.

## 2. What Was Added

### 2.1. Runtime Memory Sampling

The repo now samples process memory directly during benchmark runs via:

- `src/process_memory.h`
- `src/process_memory.cpp`

The benchmark executables now emit:

- `resident_after_load_bytes`
- `resident_after_warmup_bytes`
- `resident_after_timed_runs_bytes`
- `peak_rss_after_load_bytes`
- `peak_rss_after_warmup_bytes`
- `peak_rss_after_timed_runs_bytes`

Those fields are available in:

- `builddir/nli`
- `builddir/nli-runtime-bench`

### 2.2. Runtime Summary Integration

`tools/benchmark-nli-runtime.py` now parses and writes those RSS fields into its JSON and CSV summaries.

That means the persistent benchmark outputs now include both latency and memory, not just latency.

### 2.3. Consolidated Decision Outputs

The repo now has one generated dashboard and one generated recommendation:

- `benchmarks/nli/quantization-dashboard.md`
- `benchmarks/nli/quantization-dashboard.json`
- `benchmarks/nli/quantization-dashboard.csv`
- `QUANTIZATION_RECOMMENDATION_1.md`

Those are generated from the current benchmark artifacts, rather than being hand-written reports.

## 3. Current Persistent Runtime Results

### 3.1. CPU Persistent Runtime

From `benchmarks/nli/runtime-cpu-core-probe-persistent.csv`:

| Candidate | Load Median | Warm Median | Warm P95 | Steady RSS | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention_only` | `5014.310 ms` | `502.312 ms` | `828.630 ms` | `2866.45 MB` | `2867.70 MB` |
| `attention_proj_only` | `5127.250 ms` | `508.066 ms` | `788.708 ms` | `2880.80 MB` | `2882.03 MB` |
| `float` | `5170.690 ms` | `518.880 ms` | `834.833 ms` | `2862.61 MB` | `2862.89 MB` |

Reading:

- `attention_only` is now the best CPU persistent quantized candidate on latency as well as quality
- both quantized finalists still only improve warm median modestly versus float
- CPU steady RSS is effectively flat across all three models
- quantization does **not** buy a meaningful memory win on CPU

### 3.2. CoreML Persistent Runtime

From `benchmarks/nli/runtime-coreml-core-probe-persistent.csv`:

| Candidate | Load Median | Warm Median | Warm P95 | Steady RSS | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention_proj_only` | `15875.900 ms` | `501.197 ms` | `807.691 ms` | `3976.81 MB` | `4345.50 MB` |
| `attention_only` | `16141.300 ms` | `510.985 ms` | `794.288 ms` | `4347.11 MB` | `4348.34 MB` |
| `float` | `18297.500 ms` | `518.399 ms` | `792.368 ms` | `2432.17 MB` | `3196.09 MB` |

Reading:

- quantized finalists still have only a modest warm-latency edge over float on CoreML
- memory changes the decision materially:
  - float uses far less steady resident memory than either quantized finalist
  - float also has a much lower peak RSS than either quantized finalist
- this is not a close result; on CoreML, quantization is substantially worse on memory

## 4. What Memory Changes In The Interpretation

Before memory was measured, the case for quantization was:

- slightly better benchmark accuracy in the best case
- slightly better HF agreement in the other best case
- modest steady-state latency improvement
- modest size reduction

After memory was measured, the case is weaker:

- CPU memory is basically unchanged
- CoreML memory is materially worse for quantized models
- so memory does not offset the already small fidelity/runtime tradeoff

This is the most important new finding in this report.

## 5. Current Candidate Positions

### 5.1. `float`

Current position:

- still the best default overall

Why:

- exact HF label agreement on the full suite
- strongest `xnli-zh` result
- simplest behavior to reason about
- no memory disadvantage
- substantially better CoreML memory profile than either quantized finalist

### 5.2. `attention_only`

Current position:

- best optional experimental quantized model

Why:

- best full-suite accuracy among quantized models
- best net improvement over float on gold labels
- best CPU persistent warm latency among the current finalists

Limitations:

- lower fidelity than float
- no CPU memory win
- much worse CoreML memory than float

### 5.3. `attention_proj_only`

Current position:

- best quantized fidelity baseline for research

Why:

- slightly better HF agreement than `attention_only`

Limitations:

- weaker quality than `attention_only`
- no meaningful memory win
- much worse CoreML memory than float

## 6. Updated Overall Conclusion

The repo now has enough evidence to say:

- quantization does not win on memory
- quantization only wins modestly on persistent-session warm latency
- quantization still loses on fidelity
- float remains the most defensible default overall

So the recommendation from `QUANTIZATION_RECOMMENDATION_1.md` is reinforced, not weakened:

- default:
  - `models/mdeberta/onnx/model.onnx`
- optional experimental quantized path:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`
- secondary quantized fidelity baseline:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

## 7. Reproduction

### 7.1. Build The Runtime Binaries

```bash
cmake --build builddir --target nli nli-runtime-bench
```

### 7.2. Run The Persistent CPU Memory Benchmark

```bash
python3 tools/benchmark-nli-runtime.py \
  --mode persistent \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend cpu \
  --summary-json benchmarks/nli/runtime-cpu-core-probe-persistent.json \
  --summary-csv benchmarks/nli/runtime-cpu-core-probe-persistent.csv
```

### 7.3. Run The Persistent CoreML Memory Benchmark

```bash
python3 tools/benchmark-nli-runtime.py \
  --mode persistent \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend coreml \
  --summary-json benchmarks/nli/runtime-coreml-core-probe-persistent.json \
  --summary-csv benchmarks/nli/runtime-coreml-core-probe-persistent.csv
```

### 7.4. Regenerate The Dashboard And Recommendation

```bash
python3 tools/build-quantization-dashboard.py
```

## 8. Bottom Line

The seventh report closes the memory question.

The answer is:

- memory does not justify preferring the quantized finalists

In fact, on CoreML it pushes the other way.

That makes the current repo-level decision cleaner:

- keep float as the default model
- keep `attention_only` as the optional experimental quantized model
- only continue quantization research if there is a concrete deployment need that values modest persistent-session latency gains enough to accept worse fidelity and worse CoreML memory use
