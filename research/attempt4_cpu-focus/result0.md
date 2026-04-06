# Attempt 4 CPU Focus Result 0

Scope: first executable state of the review-driven `attempt4` CPU deployment study.

This is not the final CPU study result yet. It is the first checkpoint showing that the redesigned workflow is implemented, the dataset pack is valid, the bounded CPU catalog is wired into the scratchpad-backed study runner, and the reporting stack can consume real study and benchmark outputs.

The important conclusion from this checkpoint is:

- the `attempt4` CPU-only study design is now implemented end to end
- the new accuracy, runtime, memory, and evidence plumbing is working
- the only missing step is the full bounded run, which is long enough that it should be treated as an overnight execution on this machine

## 1. What Changed Relative To Attempt 2

`attempt4` adopts the review recommendations directly.

- CPU only
  - CoreML is fully deferred until CPU model selection is closed.
- Gold-label accuracy restored as a first-class metric
  - candidate summaries now include labeled counts and gold accuracy
  - development gates now use accuracy, not only float-reference fidelity
- Runtime and memory added to the study objective
  - persistent CPU warm latency
  - resident-after-warmup RSS
  - peak RSS guardrail
- Broader locked final test and explicit stress sets
  - `fidelity_validation`: MNLI matched/mismatched + ANLI dev r1-r3
  - `fidelity_test`: XNLI test all 15 languages + ANLI test r1-r3
  - `stress_test`: HANS eval + WANLI test
- Evidence made inspectable
  - report markdown now links repo-relative evidence files instead of absolute local paths
  - summary rows surface `smooth_quant_disabled` and `retry_reason`

## 2. Implemented Workflow State

The following pieces are now in place.

- dataset pack generator:
  - `tools/prepare-attempt4-cpu-datasets.py`
- bounded CPU catalog:
  - `research/attempt4_cpu-focus/study_quantization_catalog.json`
- artifact staging helper:
  - `tools/stage-study-artifact.py`
- updated study summarizer:
  - `tools/summarize-study-db.py`
- attempt4 CPU report builder:
  - `tools/build-attempt4-cpu-report.py`
- end-to-end runner:
  - `tools/run-attempt4-cpu-study.py`

The dataset pack now contains:

- `6` calibration slices
- `2` smoke probes
- `5` development slices
- `18` locked final-test slices
- `2` stress slices

The disjointness check completed successfully across the non-smoke pack:

- `31` attempt4 dataset slices verified disjoint

## 3. Validation Of The New Study Plumbing

The redesigned workflow was validated at three levels.

### 3.1. Static and unit validation

These checks passed:

- `python3 -m py_compile tools/prepare-attempt4-cpu-datasets.py tools/run-attempt4-cpu-study.py tools/build-attempt4-cpu-report.py tools/summarize-study-db.py tools/stage-study-artifact.py`
- `python3 -m unittest tests.python.test_summarize_study_db tests.python.test_study_catalog`

The new tests specifically cover:

- HANS binary gold scoring
- role-separated Pareto grouping
- the expanded summary schema
- the bounded `attempt4` catalog contents

### 3.2. Dataset pack validation

This completed successfully:

- `python3 tools/prepare-attempt4-cpu-datasets.py --scratchpad-root scratchpad/attempt4_cpu_focus`

Observed result:

- `verified disjointness across 31 attempt4 dataset slices`
- manifest written to `scratchpad/attempt4_cpu_focus/reports/attempt4-datasets-manifest.json`

### 3.3. Partial live run validation

The full dev-only runner was started with:

- `python3 tools/run-attempt4-cpu-study.py --skip-test --force`

This successfully completed:

- dataset preparation
- study DB initialization
- catalog staging into the `attempt4` scratchpad
- smoke evaluation start
- transition into full development evaluation

The run was then stopped intentionally once it was clear that a complete bounded execution would take hours on this machine.

That matters because it means the remaining work is now operational time, not missing implementation.

## 4. Partial Live Readout

The interrupted run still produced real study outputs for the first development slice and a real persistent CPU benchmark for the reference artifact.

Partial development evidence currently available:

- development dataset completed:
  - `mnli-validation_matched-attempt4-dev.tsv`
- candidate completed on that slice:
  - `reference`

From `scratchpad/attempt4_cpu_focus/reports/attempt4-validation-summary-partial.csv`:

- examples: `1367`
- labeled examples: `1367`
- correct predictions: `1193`
- gold accuracy: `87.271%`
- float-label agreement: `100.000%`

From `scratchpad/attempt4_cpu_focus/reports/attempt4-validation-cpu-persistent-partial.csv`:

- reference size on disk: `1116115065` bytes (`1064.4 MiB`)
- persistent load median: `5038.030 ms`
- persistent warm median: `517.587 ms`
- resident after warmup median: `2999533568` bytes (`2860.6 MiB`)
- peak RSS after timed runs median: `2999648256` bytes (`2860.7 MiB`)

Important interpretation note:

- the partial report currently shows `validation_complete=false`
- it therefore reports `validation_gate_pass=false` with reason `missing validation datasets`
- that is expected and correct for an interrupted run
- this partial output is a plumbing check, not a candidate ranking result

## 5. Current Recommendation Status

There is no final CPU recommendation yet from `attempt4`.

Current honest status:

- implementation complete
- evidence plumbing complete
- partial live outputs complete
- full bounded study not yet completed

So the correct present statement is:

- `attempt4` is ready to run as the new CPU deployment study
- `attempt4` has not yet produced its final locked CPU Pareto frontier

## 6. Evidence

Primary engineering-checkpoint artifacts:

- `scratchpad/attempt4_cpu_focus/reports/attempt4-datasets-manifest.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-validation-summary-partial.csv`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-validation-summary-partial.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-validation-cpu-persistent-partial.csv`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-validation-cpu-persistent-partial.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary-partial.csv`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary-partial.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary-partial.md`

## 7. Bottom Line

`attempt4` has crossed the important threshold:

- it is no longer a plan
- it is now a runnable CPU deployment study with the intended gates, datasets, metrics, and report outputs

The next step is straightforward:

- run the full bounded CPU study to completion
- then write the first final-result document from the locked output artifacts
