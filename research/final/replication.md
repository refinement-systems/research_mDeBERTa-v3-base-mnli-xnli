# Replicating the Final Research Results

This guide explains how to reproduce the **final conclusions** in this repository:

1. CPU recommendation: `nncf_accuracy_attention_only`
2. CoreML conclusion on this machine/path: do not use current ORT CoreML as primary deployment backend

It covers:

1. which scripts to run,
2. what external software/libraries/datasets are downloaded,
3. what each script does in detail.

---

## 1) End-to-end commands (recommended order)

Run from repo root.

```bash
git submodule update --init --recursive
./tools/setup.sh
./tools/build.sh
./tools/download-mdeberta-v3-base.sh --tokenizer-assets
python3 tools/run-attempt4-cpu-study.py --force
python3 tools/run-attempt5-coreml-study.py --skip-test --force
```

Notes:

- `run-attempt4-cpu-study.py` reproduces the full CPU final package (validation + locked test + stress + runtime/cold benchmarks + report).
- `run-attempt5-coreml-study.py --skip-test` reproduces the CoreML preflight that drove the final “not recommended” backend conclusion.
- On Apple Silicon with CoreML available, you can run full attempt5 (without `--skip-test`) if you want the entire locked-final CoreML artifact set.

---

## 2) What is downloaded from outside the repo

## 2.1 Code/libraries/toolchain inputs

### Git submodules

`git submodule update --init --recursive` pulls external code used by CMake:

- `subprojects/onnxruntime` from `https://github.com/microsoft/onnxruntime`
- `subprojects/sentencepiece` from `https://github.com/google/sentencepiece.git`
- `subprojects/Topical-Chat` from `https://github.com/alexa/Topical-Chat.git`

### CMake/system libs

- Requires local `SQLite3` (`find_package(SQLite3 REQUIRED)`).
- Builds ONNX Runtime and SentencePiece through subprojects during `./tools/setup.sh` + `./tools/build.sh`.

## 2.2 Model assets downloaded externally

`./tools/download-mdeberta-v3-base.sh --tokenizer-assets` downloads from Hugging Face model repo:

- Model repo: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- Endpoint pattern: `https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli/resolve/main/...`
- Main files used by this study flow:
  - `spm.model`
  - `onnx/model.onnx`
  - `onnx/model_quantized.onnx`
  - tokenizer/config assets (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `config.json`)

## 2.3 Datasets downloaded externally

The attempt4/attempt5 runners call `tools/prepare-attempt4-cpu-datasets.py`, which fetches from:

- Hugging Face datasets-server API (`https://datasets-server.huggingface.co`) for:
  - MNLI (`nyu-mll/multi_nli`)
  - XNLI (`facebook/xnli`)
  - ANLI (`facebook/anli`)
  - WANLI (`alisawuffles/WANLI`)
  - HANS dataset metadata (`jhu-cogsci/hans`)
- Direct text fetch for HANS evaluation file:
  - `https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt`

It also copies local frozen datasets already checked into `benchmarks/nli` (calibration + smoke TSVs).

---

## 3) What each script is doing in detail

## 3.1 `./tools/setup.sh`

- Configures CMake build directory (`builddir`) with Ninja generator.
- Enables compile commands export.
- Sets install prefix to repo root.
- Validates `builddir` is a CMake tree if it already exists.

## 3.2 `./tools/build.sh`

- Runs `cmake --build builddir` to compile binaries, especially:
  - `builddir/nli`
  - `builddir/nli-runtime-bench`
  - `builddir/nli-study`

These binaries are required by the study runners.

## 3.3 `./tools/download-mdeberta-v3-base.sh --tokenizer-assets`

- Downloads baseline ONNX artifacts + tokenizer assets into `models/mdeberta`.
- Skips already existing files unless `--force` is provided.
- Uses `curl` with fail-on-error and temp-file swap for safer writes.

## 3.4 `python3 tools/run-attempt4-cpu-study.py --force`

This is the **main CPU final-results reproducer**.

### Inputs

- Catalog: `research/attempt4_cpu-focus/study_quantization_catalog.json`
- Scratchpad default: `scratchpad/attempt4_cpu_focus`
- Required binaries: `builddir/nli-study`, `builddir/nli`, `builddir/nli-runtime-bench`

### Pipeline

1. **Prepare dataset pack** via `tools/prepare-attempt4-cpu-datasets.py`.
   - Creates/reuses calibration, smoke, validation, test, stress datasets.
   - Writes a dataset manifest JSON under scratchpad reports.
2. **Stage tokenizer runtime assets** into scratchpad model directory.
3. **Initialize study DB** (`nli-study init`) using the catalog.
4. **Validate dataset-role assignments** in SQLite (calibration/smoke/fidelity_validation/fidelity_test/stress_test).
5. **Run study evaluations** (`nli-study run`) on CPU backend for bounded quantization list:
   - smoke datasets + development validation datasets.
6. **Summarize validation rows** with `tools/summarize-study-db.py`.
7. **Benchmark development-complete candidates** on CPU with `tools/benchmark-nli-runtime.py`:
   - persistent mode,
   - RSS measurement,
   - outputs JSON/CSV.
8. **Build intermediate report** with `tools/build-attempt4-cpu-report.py`.
   - Computes development gates and locked candidate set.
9. **Run locked final datasets** (unless `--skip-test`):
   - fidelity test datasets,
   - stress datasets.
10. **Run cold-start benchmark** for locked candidates.
11. **Build final report** and write a manifest (`attempt4-manifest.json`) listing all output artifact paths.

### Outputs you should inspect

- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary.md`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-manifest.json`

## 3.5 `python3 tools/run-attempt5-coreml-study.py --skip-test --force`

This is the **CoreML follow-up reproducer** for the final project-level backend conclusion.

### Inputs

- Catalog: `research/attempt5_coreml-focus/study_quantization_catalog.json`
- Scratchpad default: `scratchpad/attempt5_coreml_focus`
- Required binaries: same as attempt4 runner

### Pipeline

1. Rebuild/reuse the same broad dataset pack via `tools/prepare-attempt4-cpu-datasets.py`.
2. Stage tokenizer assets into attempt5 scratchpad.
3. Initialize study DB from attempt5 catalog.
4. Run CoreML backend evaluation for primary candidates:
   - `reference`
   - `reference_fp16`
5. Optionally run CPU-winner control (`nncf_accuracy_attention_only`) unless `--skip-controls`.
   - Control failures are tolerated and recorded.
6. Summarize validation rows.
7. Benchmark validation-complete CoreML candidates (persistent mode + RSS).
8. Build intermediate CoreML report (`tools/build-attempt5-coreml-report.py`) and lock surviving candidates.
9. If not `--skip-test`, run locked final test + stress + cold-start benchmark, then build final report.
10. Write manifest (`attempt5-manifest.json`).

### Why `--skip-test` is commonly used

The attempt5 plan/result record treats the preflight as sufficient for the final backend decision on this machine/path. `--skip-test` reproduces that shorter decision path.

---

## 4) Mapping to the final conclusions

After running the above, compare with the final summaries in `research/final`:

- CPU lane closes on `nncf_accuracy_attention_only`.
- CoreML lane (ORT CoreML path on this machine) is not promoted as primary backend.

The study manifests and generated report markdown/JSON files under each attempt scratchpad are the reproducibility backbone for those final claims.

---

## 5) Optional: reproduce earlier historical steps

If you also want to recreate older slice-generation workflows used earlier in the project:

```bash
python3 tools/download-study-assets.py --force
```

That script orchestrates:

- model downloads (same HF repo),
- attempt1 quantization data prep (`tools/prepare-attempt1-quantization-data.py`),
- final eval-slice pulls (`tools/download-nli-eval-slices.py`),
- and copying frozen smoke probe datasets.

This is useful for archival reproducibility, but not required for the final attempt4/attempt5 conclusions.
