# 1. Full HF-vs-ONNX finalist benchmark on the 1,950-example suite
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --max-examples-per-source 0 \
  --sample-mode first \
  --summary-json benchmarks/nli/hf-finalist-full.json \
  --summary-csv benchmarks/nli/hf-finalist-full.csv

# 2. Rebuild the hard probe from the full benchmark
.venv/bin/python tools/build-hf-probe-set.py \
  --include-finalist-label-diffs \
  --include-float-top-drift 10

# 3. Benchmark the 61-example hard probe
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --tsv benchmarks/nli/hf-probe-set.tsv \
  --sample-mode first \
  --max-examples-per-source 0 \
  --summary-json benchmarks/nli/hf-probe-benchmark.json \
  --summary-csv benchmarks/nli/hf-probe-benchmark.csv

# 4. Rebuild the 25-example core probe
python3 tools/build-hf-core-probe.py

# 5. Benchmark the core probe
.venv/bin/python tools/benchmark-hf-onnx-models.py \
  --tsv benchmarks/nli/hf-core-probe.tsv \
  --sample-mode first \
  --max-examples-per-source 0 \
  --summary-json benchmarks/nli/hf-core-probe-benchmark.json \
  --summary-csv benchmarks/nli/hf-core-probe-benchmark.csv

# 6. Runtime benchmark on the core probe, CPU only
python3 tools/benchmark-nli-runtime.py \
  --max-examples 0 \
  --repeat 3 \
  --warmup 1 \
  --backend cpu \
  --summary-json benchmarks/nli/runtime-cpu-core-probe.json \
  --summary-csv benchmarks/nli/runtime-cpu-core-probe.csv

