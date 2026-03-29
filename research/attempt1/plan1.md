# Attempt 1 Plan 1

Scope: possible research directions after the initial review and the first round of attempt1 tooling work.

This document is forward-looking. It reflects:

- the attempt0 conclusion that `attention_only` and `attention_proj_only` are the current quantized frontier
- the reviewer feedback in `initial_review.md`
- the current repo state after adding attempt1 data-prep, static-QDQ, NNCF PTQ, and gated QAT tooling

## 1. Current Position

The repo is no longer blocked on missing tooling.

Attempt1 now has:

- shared family-exclusion helpers so `attention_only` and `attention_proj_only` can be reused across search methods
- extended slice generation for disjoint calibration, search-validation, and optional fine-tune datasets
- a disjointness checker for TSV overlap verification
- a static QDQ path that can reuse the same family exclusions as the dynamic family search
- an NNCF/OpenVINO PTQ driver for:
  - plain PTQ
  - accuracy-controlled PTQ
  - `gold_accuracy` and `hf_agreement` validation metrics
  - `none`, `attention_only`, and `attention_proj_only` ignored-scope seeds
- a gated QAT recovery script that is designed to export back to ONNX
- an attempt1 sweep harness that can generate and screen new candidates

What is already validated:

- the new static QDQ path works in a small smoke test
- the new NNCF PTQ path works in a small smoke test
- the NNCF PTQ path currently needs a SmoothQuant fallback on this model/backend

What is not yet fully validated:

- the long-running NNCF `accuracy-control` mode
- the full QAT training/export path
- the full attempt1 sweep on disjoint search data

That means the repo should treat attempt1 as ready for focused research work, but not yet as a finished benchmark campaign.

## 2. Mainline Direction: Finish The ONNX/CoreML-Compatible Search

This should remain the primary direction.

Reason:

- it matches the deployment path the repo already cares about
- it directly addresses the reviewer's strongest criticism
- the repo now has most of the required scaffolding

### 2.1. Prepare disjoint search data

Use the new attempt1 data-prep flow to build:

- calibration slices from `mnli/train` and `xnli/validation`
- held-out search-validation slices from later non-overlapping windows of the same splits
- optional fine-tune slices from a later non-overlapping window

Then verify:

- calibration does not overlap search-validation
- neither overlaps the existing final full-suite TSVs
- neither overlaps the hard probe or core probe

Deliverable:

- a stable attempt1 search-data set under `benchmarks/nli/`

### 2.2. Make NNCF accuracy-control reliable enough to use

This is the main immediate blocker.

The plain NNCF PTQ path works, but the full accuracy-control path has not yet completed a clean smoke run. The next work should be:

- rerun a tiny `accuracy-control` candidate on the disjoint search-validation set
- inspect the exact failure mode if it still fails
- prefer a targeted fix over broad rewrites

Likely sub-directions if the current path remains unstable:

- keep the SmoothQuant fallback and verify it works for accuracy-control as well as PTQ
- try OpenVINO IR as the internal working representation if the ONNX route remains brittle
- reduce the initial validation slice size to confirm correctness before scaling up

Deliverable:

- one confirmed working `accuracy-control` candidate run

### 2.3. Run the first serious NNCF candidate set

Once `accuracy-control` is stable, run the smallest useful matrix:

- metrics:
  - `gold_accuracy`
  - `hf_agreement`
- ignored scopes:
  - `none`
  - `attention_only`
  - `attention_proj_only`
- preset:
  - `mixed`

This should produce six first-pass NNCF candidates.

Screen them in this order:

- core probe
- hard probe
- full-suite HF/ONNX benchmark

Only promoted candidates should get:

- persistent runtime
- RSS / peak RSS

Deliverable:

- the first benchmark-backed answer to whether productized mixed-precision search beats the manual family frontier

## 3. Secondary Direction: Systematic Static QDQ Tuning

This should run in parallel with the NNCF work once the disjoint search data exists.

Reason:

- the reviewer correctly identified datatype/format tuning as an underexplored axis
- the static tooling is already implemented and smoke-tested
- the static search is cheaper to operationalize than QAT

The first sweep should stay narrow:

- ignored scopes:
  - `none`
  - `attention_only`
  - `attention_proj_only`
- activation/weight combinations:
  - `S8S8`
  - `U8U8`
  - `U8S8 + reduce_range`
- calibration methods:
  - `minmax`
  - `percentile`
- calibration subset sizes:
  - one small setting
  - one larger setting

Do not expand to `QOperator` unless there is a concrete reason.

Deliverable:

- a ranked static-QDQ table that can be compared directly against both attempt0 finalists and the NNCF results

## 4. Gated Direction: QAT Recovery

This should start only after the repo has one or two serious PTQ seeds worth rescuing.

Reason:

- QAT is the most expensive path in engineering and runtime time
- the local machine is currently CPU-only for PyTorch
- the repo already has a strong PTQ frontier, so QAT should be used for recovery, not blind exploration

The first QAT pass should be deliberately small:

- start from the best PTQ seed, not from raw float
- `1` epoch
- `1e-5` learning rate
- reduced training slice
- validation against either gold accuracy or HF agreement on the held-out search-validation data

Only escalate to longer runs if:

- the pilot improves the selected metric materially, or
- an accelerator becomes available

Deliverable:

- one exported ONNX QAT recovery artifact and its benchmark results

## 5. Quality-Focused Direction: Localized Repair, Especially Chinese

Attempt0 showed that:

- FFN quantization was the main trap
- Chinese remained the clearest unresolved weakness
- translated XNLI stress cases were disproportionately informative

Once the new NNCF and static candidates exist, the repo should explicitly ask:

- do any new candidates improve `xnli-zh` without giving back the rest of the suite?
- do they repair the known translated stress patterns or only improve average metrics?

If no candidate clears that bar, a targeted follow-up is justified:

- small ignored-scope adjustments around the best new mixed-precision candidate
- a narrow QAT recovery pass against a Chinese-sensitive validation subset

Deliverable:

- a direct answer to whether attempt1 reduces the known Chinese guardrail failures

## 6. Operational Direction: Reconfirm Whether Better Quality Still Loses On Memory

Any promoted attempt1 candidate must be re-judged on:

- full-suite quality
- hard/core probes
- persistent warm latency
- CPU RSS
- CoreML RSS

This matters because attempt0 already showed that:

- quantized finalists were only modestly faster
- CoreML memory was much worse for quantized artifacts than for float

So even if attempt1 improves quality, it still may not change the default-model answer.

Deliverable:

- a refreshed dashboard and recommendation once the attempt1 frontier stabilizes

## 7. Optional Research Branch: Alternate Runtime, Not Mainline

This branch should stay explicitly separate from the main attempt1 track.

Candidates:

- `LLM.int8()`
- `QLoRA`

Reasons to keep it separate:

- it does not preserve the current ONNX/CoreML deployment story
- it answers a different research question
- xCOMET-lite suggests GPTQ is not the obvious move for this architecture

This branch only makes sense if the repo wants to answer:

- is there an alternate-runtime path that preserves more fidelity than ONNX/CoreML quantization?

Deliverable:

- a research-only comparison, not a shipping recommendation

## 8. Recommended Execution Order

1. prepare disjoint attempt1 search data
2. finish validating NNCF `accuracy-control`
3. run the first NNCF mixed-precision candidate matrix
4. run the first static QDQ candidate matrix
5. compare both against the attempt0 frontier
6. run QAT only on the best PTQ seeds
7. benchmark promoted candidates on runtime and memory
8. update the dashboard and recommendation

## 9. Bottom Line

The best research direction from here is not another broad manual exclusion sweep.

The repo should now test whether:

- productized mixed-precision PTQ beats the hand-built frontier
- scoped static QDQ tuning closes the remaining gap
- a small QAT recovery pass can recover fidelity without losing the operational story

If none of those paths produce a candidate that improves quality without preserving the attempt0 memory/runtime disadvantages, the likely conclusion will remain:

- float stays the default
- quantization remains an opt-in experimental path
- the real gain from attempt1 is better evidence, not necessarily a different shipping decision
