# Winning CPU Candidate: `nncf_accuracy_attention_only`

## 1) What was the winning CPU candidate?

The final CPU winner in this repository is:

- **`nncf_accuracy_attention_only`**

This is explicitly recorded as the final CPU recommendation in the final summary and the full CPU-focused result package. It won as the smallest locked, nondominated quantized candidate under the final attempt4 objective that combined accuracy checks, memory, size, and runtime-frontier reporting.

---

## 2) Where did this candidate come from?

The artifact used as the CPU winner in the final attempt was not created from scratch during attempt4 itself. It came from an earlier NNCF quantization flow and was then carried forward.

### Stage A — Original generation recipe (NNCF accuracy-control)

The candidate recipe is documented in the attempt2 catalog and conclusion notes:

- Script: `tools/quantize-onnx-nncf.py`
- Mode: `--mode=accuracy-control`
- Metric: `--metric=gold_accuracy`
- Preset: `--preset=mixed`
- Scope exclusion family: `--ignored-scope-family=attention_only`
- Subset/drop controls: `--subset-size=300`, `--max-drop=0.01`
- With preprocessing and fast bias correction
- Uses calibration and validation TSV bundles

So the winning artifact identity is strongly tied to **NNCF accuracy-controlled mixed-precision PTQ** with the `attention_only` ignored-scope family.

### Stage B — Candidate family search that produced this style of artifact

The broader attempt1 search script (`tools/search-onnx-quantization-attempt1.py`) builds NNCF candidate grids by family/track and emits commands that call `tools/quantize-onnx-nncf.py` with accuracy/fidelity metrics and parameter sweeps.

That means this winner was born in the NNCF search family, not as an ad-hoc one-off command.

### Stage C — Carry-forward and staging in the final CPU study

In attempt4 (the final CPU closing study), the catalog entry for `nncf_accuracy_attention_only` uses:

- `tools/stage-study-artifact.py`
- source: `models/mdeberta/onnx/candidates/attempt1/nncf_accuracy_attention_only.onnx`
- destination: the attempt4 scratchpad candidate path

So attempt4 treats the model as an incumbent artifact, stages it into the bounded study workspace, and then evaluates it under the locked CPU pipeline.

---

## 3) What did each script do?

## `tools/quantize-onnx-nncf.py` (creation script)

This is the generator that actually quantizes ONNX models using NNCF/OpenVINO tooling.

At a high level it:

1. Parses quantization controls (mode, metric, preset, subset size, ignored-scope family, drop limit, preprocess flags).
2. Validates required calibration/validation TSVs.
3. Ensures required Python packages exist (`onnx`, `onnxruntime`, `transformers`, `sentencepiece`, `numpy`, `nncf`; can auto-install into a venv).
4. Runs helper logic that loads model/tokenizer + examples and executes NNCF PTQ / accuracy-control flow.
5. Emits JSON summary metadata, which downstream tooling can parse.

So this is the **actual model-construction step** behind `nncf_accuracy_attention_only`.

## `tools/search-onnx-quantization-attempt1.py` (sweep/orchestration script)

This script creates many candidate specs across tracks and families.

For NNCF tracks it:

- iterates families (like `attention_only`, `attention_proj_only`)
- iterates tracks (`nncf_accuracy`, `nncf_fidelity`)
- varies subset size / max-drop / bias-correction settings
- assembles concrete commands to run `tools/quantize-onnx-nncf.py`

This is how the repo systematically explored the search space that included the future winner.

## `tools/stage-study-artifact.py` (carry-forward/staging script)

In the final CPU study, this script does not re-quantize. It:

1. Copies a source artifact into the study scratchpad.
2. Optionally parses original stdout logs to preserve generation metadata (`smooth_quant_disabled`, `retry_reason`).
3. Emits a small JSON payload with source/destination/size/metadata.

This makes older artifacts reproducibly traceable inside new study runs.

## `tools/prepare-attempt4-cpu-datasets.py` (dataset pack builder)

This script prepares the frozen dataset pack for attempt4:

- copies frozen calibration and smoke TSVs
- downloads/exports larger dev/test/stress datasets (MNLI/XNLI/ANLI/HANS/WANLI flow)
- writes a manifest under the scratchpad reports
- includes retry/backoff and request pacing for dataset API calls

This gives the locked evaluation inputs used to judge the winner.

## `tools/run-attempt4-cpu-study.py` (main final CPU pipeline)

This is the end-to-end attempt4 runner. It:

1. runs dataset prep
2. stages tokenizer runtime assets
3. initializes the study database from the attempt4 catalog
4. runs `nli-study` evaluations across bounded candidates/datasets on CPU
5. summarizes validation rows
6. benchmarks persistent CPU runtime + RSS
7. builds intermediate report + locked candidate set
8. runs final test/stress and cold benchmarks (unless skipped)
9. builds final report + manifest

So this is the script that elevated the staged incumbent into a **final winner under locked study rules**.

## `tools/build-attempt4-cpu-report.py` (gate + frontier/report logic)

This report-builder joins summary rows and runtime CSVs, computes aggregated metrics, applies validation gates, and performs final frontier logic.

Important behaviors include:

- aggregate MNLI/ANLI dev accuracy checks
- float-label agreement threshold checks
- peak RSS ratio checks vs reference
- nondominated frontier logic on size/warm-latency/resident memory axes

Its outputs drive the final table/readout that selects `nncf_accuracy_attention_only` as default CPU recommendation in attempt4 result docs.

---

## 4) Why this candidate specifically won in the final CPU study

In attempt4 final reporting, the locked frontier includes float + three NNCF mid-size survivors. The selected default recommendation is `nncf_accuracy_attention_only` because it is the smallest locked nondominated quantized point and wins the frozen tie-break among equal-size candidates while staying very close to float on locked-final gold accuracy.

Operationally, the study documents that this winner trades faster warm latency (lost to float) for much smaller disk footprint and lower steady RSS.

---

## 5) One-sentence summary

**`nncf_accuracy_attention_only` was created by the NNCF accuracy-control quantization flow (`tools/quantize-onnx-nncf.py`), discovered/organized via the attempt1 search workflow, staged forward with provenance into attempt4, and finally selected by the attempt4 locked CPU study runner + report gates/frontier logic.**
