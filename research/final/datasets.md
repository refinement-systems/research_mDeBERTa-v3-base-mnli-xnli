# Final Dataset Summary

This document describes the datasets used across `attempt0` through `attempt5`.

The research used two kinds of data:

- upstream benchmark families such as MNLI, XNLI, ANLI, WANLI, and HANS
- repo-local derived TSVs built from those families, plus custom probe/stress packs built from benchmark outputs

## 1. Normalized TSV Format

Most study datasets were normalized into TSV rows with this shape:

- `benchmark`
- `id`
- `label`
- `premise`
- `hypothesis`
- `dataset`
- `config`
- `split`
- `row_idx`

The study runner only needs `premise` and `hypothesis` for inference, and `label` when gold scoring is enabled.

Some custom probe packs add extra columns such as:

- `selection_reasons`
- model labels from earlier finalists
- logit-drift summaries

Those extra columns exist to explain why a row was included in a probe; they are not required for ordinary study execution.

## 2. Upstream Benchmark Families

### 2.1. MNLI

MNLI is the main English three-way NLI source used throughout the work.

It describes:

- sentence-pair natural language inference in English
- label space:
  - `entailment`
  - `neutral`
  - `contradiction`

How it was used:

- early benchmark suite in `attempt0`
- calibration/search/fine-tune slices in `attempt1`
- backend-separated study roles in `attempt2`
- development gates in `attempt4` and `attempt5`

Important splits:

- `train`
- `validation_matched`
- `validation_mismatched`

### 2.2. XNLI

XNLI is the multilingual NLI family used for cross-lingual preservation checks.

It describes:

- translated NLI sentence pairs across multiple languages
- the same three-way label space as MNLI

How it was used:

- early multilingual benchmark suite in `attempt0`
- calibration/search/fine-tune slices in `attempt1`
- smaller five-language CoreML pack in `attempt3`
- full 15-language locked final-test pack in `attempt4` and `attempt5`

Language coverage changed by attempt:

- early work:
  - `de`
  - `en`
  - `es`
  - `fr`
  - `zh`
- broad deployment pack:
  - `ar`
  - `bg`
  - `de`
  - `el`
  - `en`
  - `es`
  - `fr`
  - `hi`
  - `ru`
  - `sw`
  - `th`
  - `tr`
  - `ur`
  - `vi`
  - `zh`

### 2.3. ANLI

ANLI is the hard adversarial NLI family added in the later deployment studies.

It describes:

- adversarially collected three-way NLI examples
- label space:
  - `entailment`
  - `neutral`
  - `contradiction`

How it was used:

- not part of the early attempt0/attempt1 benchmark suite
- added in `attempt4` and reused in `attempt5`
- used as both development and locked-final evaluation material

Important splits:

- `dev_r1`
- `dev_r2`
- `dev_r3`
- `test_r1`
- `test_r2`
- `test_r3`

### 2.4. WANLI

WANLI is a later stress-only dataset.

It describes:

- harder or more adversarial three-way NLI examples

How it was used:

- only in `attempt4` and `attempt5`
- report-only stress set
- never used for candidate promotion

### 2.5. HANS

HANS is the syntactic-heuristics challenge set.

It describes:

- examples designed to expose heuristic shortcuts in NLI models
- binary label space:
  - `entailment`
  - `non-entailment`

How it was used:

- only in `attempt4` and `attempt5`
- report-only stress set
- never used for candidate promotion

Special scoring note:

- model outputs stayed three-way internally
- for HANS scoring, `neutral` and `contradiction` were collapsed into `non-entailment`

## 3. Frozen Local Benchmark Slices

These are the checked-in local slices that became the stable gates for the early and middle attempts.

### 3.1. Early Full-Suite Evaluation Gates

Main frozen benchmark suite:

- `mnli-validation_matched-200-per-label.tsv`
- `mnli-validation_mismatched-200-per-label.tsv`
- `xnli-de-test-50-per-label.tsv`
- `xnli-en-test-50-per-label.tsv`
- `xnli-es-test-50-per-label.tsv`
- `xnli-fr-test-50-per-label.tsv`
- `xnli-zh-test-50-per-label.tsv`

What they describe:

- a deterministic, class-balanced English + multilingual evaluation suite
- this was the main broad benchmark in `attempt0` and `attempt1`

Why they mattered:

- they replaced anecdotal single-example checks
- they became the basis for the HF-vs-ONNX finalist comparisons

### 3.2. Smaller Exploratory MNLI Slices

Earlier small MNLI slices also exist:

- `mnli-validation_matched-100-per-label.tsv`
- `mnli-validation_mismatched-100-per-label.tsv`

What they describe:

- smaller class-balanced MNLI validation subsets used in early exploratory work

Important caution:

- they overlap with the later `200-per-label` MNLI slices
- they should not be aggregated together with the final frozen full suite

## 4. Calibration, Search, and Fine-Tune Slices

These were introduced mainly in `attempt1` so PTQ/QAT work would not reuse the same rows for every role.

### 4.1. Calibration Slices

Files:

- `mnli-train-calibration-64-per-label.tsv`
- `xnli-de-validation-calibration-32-per-label.tsv`
- `xnli-en-validation-calibration-32-per-label.tsv`
- `xnli-es-validation-calibration-32-per-label.tsv`
- `xnli-fr-validation-calibration-32-per-label.tsv`
- `xnli-zh-validation-calibration-32-per-label.tsv`

What they describe:

- representative NLI inputs reserved for PTQ calibration or activation-range estimation
- not used as final ranking data

Later role:

- these same frozen calibration slices were reused in `attempt4` and `attempt5`

### 4.2. Search-Validation Slices

Files follow these patterns:

- `mnli-train-search-validation-skip64-64-per-label.tsv`
- `xnli-<lang>-validation-search-validation-skip32-32-per-label.tsv`

What they describe:

- disjoint development slices for trying many candidate quantizers
- drawn later in the same upstream split so they do not overlap the calibration rows

### 4.3. Fine-Tune Slices

Files follow these patterns:

- `mnli-train-fine-tune-skip128-128-per-label.tsv`
- `xnli-<lang>-validation-fine-tune-skip64-32-per-label.tsv`

What they describe:

- reserved rows for optional QAT or follow-up training experiments

Practical note:

- the QAT lane never became a major positive result
- but these slices document the intended separation between calibration, selection, and training-like roles

## 5. Probe and Stress Packs Built Inside The Repo

These are not raw benchmark families. They are custom packs built from earlier benchmark outputs to make regression checks faster and more informative.

### 5.1. `hf-finalist-full.json` / `hf-finalist-full.csv`

These are derived benchmark outputs, not raw corpora.

What they describe:

- per-example comparisons between HF, float ONNX, and finalist quantized models over the frozen early full suite

Why they matter:

- they were the source material for later probe/stress construction

### 5.2. `hf-probe-set.tsv`

Built by:

- `tools/build-hf-probe-set.py`

What it describes:

- a fixed hard probe built from the HF-vs-ONNX finalist comparison
- it preserves rows with:
  - HF/model label disagreements
  - highest logit-drift examples
  - finalist label disagreements

How it was used:

- fast smoke dataset
- qualitative regression detector
- later copied into `attempt4` and `attempt5`

### 5.3. `hf-core-probe.tsv`

Built by:

- `tools/build-hf-core-probe.py`

What it describes:

- a smaller deterministic core subset of `hf-probe-set.tsv`
- default target size:
  - `25` rows

Selection goals:

- preserve finalist disagreement rows
- preserve benchmark/source diversity
- preserve language diversity, including Chinese
- preserve top-drift cases

How it was used:

- smoke screening
- runtime benchmarking
- RSS benchmarking

### 5.4. `attempt1-zh-sensitive-validation.tsv`

Built by:

- `tools/build-attempt1-zh-eval-pack.py`

What it describes:

- a held-out Chinese-sensitive validation pack built from local zh validation slices
- explicitly verified to be disjoint from the frozen final evaluation gates

How it was used:

- to test whether candidate changes harmed Chinese disproportionately
- mainly in `attempt1`

### 5.5. `attempt1-zh-stress-pack.tsv`

Built by:

- `tools/build-attempt1-zh-stress-pack.py`

What it describes:

- a compact zh-only hard set built from the benchmark JSON
- preserves rows with:
  - candidate/HF disagreements
  - candidate/gold errors
  - finalist label disagreements
  - top logit drift

How it was used:

- diagnostic stress test for the Chinese fault line in `attempt1`
- not the primary selector for the overall frontier

## 6. Attempt-Specific Study Packs

### 6.1. Attempt 2

`attempt2` did not introduce a new raw corpus family.

What changed in `attempt2` was the evaluation protocol:

- calibration
- fidelity_validation
- fidelity_test
- smoke

The data still came mainly from the existing MNLI/XNLI-derived slices and probe sets. The important change was role separation and locked split discipline, not new benchmark content.

### 6.2. Attempt 3 CoreML Pack

`attempt3` used a deliberately smaller CoreML-specific pack to answer a narrow fidelity question cheaply.

Validation-like slices:

- `mnli-train-attempt3-coreml-search-validation-skip256-64-per-label.tsv`
- `xnli-en-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`
- `xnli-de-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`
- `xnli-es-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`
- `xnli-fr-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`
- `xnli-zh-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`

Test slices:

- `mnli-validation_matched-attempt3-coreml-test-skip300-50-per-label.tsv`
- `mnli-validation_mismatched-attempt3-coreml-test-skip300-50-per-label.tsv`
- `xnli-en-test-attempt3-coreml-test-skip100-50-per-label.tsv`
- `xnli-de-test-attempt3-coreml-test-skip100-50-per-label.tsv`
- `xnli-es-test-attempt3-coreml-test-skip100-50-per-label.tsv`
- `xnli-fr-test-attempt3-coreml-test-skip100-50-per-label.tsv`
- `xnli-zh-test-attempt3-coreml-test-skip100-50-per-label.tsv`

What they describe:

- a smaller five-language CoreML evaluation pack
- enough to check whether `reference_fp16` preserved backend-specific float behavior before spending time on full deployment benchmarking

### 6.3. Attempt 4 / Attempt 5 Broad Deployment Pack

Built by:

- `tools/prepare-attempt4-cpu-datasets.py`

This became the final broad deployment-study pack and was reused in both `attempt4` and `attempt5`.

`calibration`

- the six frozen calibration slices listed above

`smoke`

- `hf-probe-set.tsv`
- `hf-core-probe.tsv`

`fidelity_validation`

- `mnli-validation_matched-attempt4-dev.tsv`
- `mnli-validation_mismatched-attempt4-dev.tsv`
- `anli-r1-dev-attempt4-dev.tsv`
- `anli-r2-dev-attempt4-dev.tsv`
- `anli-r3-dev-attempt4-dev.tsv`

What it describes:

- the development set used for gated promotion in the final CPU/CoreML study framework

`fidelity_test`

- `xnli-<15 languages>-test-attempt4-test.tsv`
- `anli-r1-test-attempt4-test.tsv`
- `anli-r2-test-attempt4-test.tsv`
- `anli-r3-test-attempt4-test.tsv`

What it describes:

- the locked final evaluation pack touched only after development selection

`stress_test`

- `hans-evaluation-attempt4-stress-test.tsv`
- `wanli-test-attempt4-stress-test.tsv`

What it describes:

- report-only robustness sets used after selection

## 7. Naming Conventions

The file names encode the sampling plan.

- `*-per-label`
  - class-balanced sampling count per label

- `skipNN`
  - later disjoint slice from the same upstream split

- `attempt3-coreml-search-validation`
  - smaller CoreML-specific development pack

- `attempt4-dev`
  - development-role slice in the final broad study pack

- `attempt4-test`
  - locked-final slice in the final broad study pack

- `stress-test`
  - report-only robustness slice

## 8. Bottom Line

The dataset story across the research is:

- early work relied on frozen MNLI/XNLI slices plus custom HF-derived probes
- `attempt1` added disciplined calibration/search/fine-tune separation and zh-specific diagnostics
- `attempt3` used a smaller CoreML-specific fidelity pack
- `attempt4` and `attempt5` converged on the final broad deployment-study pack:
  - MNLI for English NLI
  - XNLI for multilingual generalization
  - ANLI for harder adversarial NLI
  - WANLI and HANS for report-only stress checks
  - `hf-probe-set.tsv` and `hf-core-probe.tsv` for smoke and operational benchmarking

That is the dataset base behind the final conclusions in `summary.md`.
