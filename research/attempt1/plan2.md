# Attempt 1 Plan 2

Scope: updated research plan after the first benchmark-backed attempt1 NNCF results in `result1.md`.

This document supersedes the broad "prove the tooling works" phase from `plan1.md`. That phase is complete. The repo now needs a narrower plan that reacts to the actual evidence.

## 1. Current Position

Attempt1 now has three important facts established:

1. The attempt1 infrastructure is operational.
   Disjoint calibration and search-validation data exist, NNCF `accuracy-control` runs complete on this repo, and new ONNX candidates can be screened through the existing benchmark harness.

2. The first NNCF mixed-precision wave did not beat the attempt0 frontier.
   - `nncf_accuracy_attention_only` is not competitive.
   - `nncf_fidelity_attention_proj_only` is the best attempt1 NNCF result so far, but it only ties the attempt0 fidelity frontier and still loses on `xnli-zh`.

3. The main unresolved quality problem is still localized, not global.
   The first NNCF runs did not repair the known Chinese weakness, and they did not produce a balanced new best model.

That means the repo should stop spending time on broad first-pass validation work and move to narrower, evidence-driven searches.

## 2. Revised Main Hypothesis

The most promising next path is no longer "more generic NNCF search."

The updated mainline hypothesis is:

- static QDQ datatype and calibration tuning is now the cheapest serious unexplored axis
- the only NNCF branch still worth follow-up is the fidelity-oriented `attention_proj_only` branch
- any training-based work should be treated as recovery for near-frontier seeds, not as a primary search method

## 3. Updated Research Priorities

### 3.1. First Priority: Static QDQ on the frontier families

This should become the mainline attempt1 direction.

Reason:

- it directly targets an underexplored axis from the review
- it stays on the ONNX/CoreML-compatible path
- it is cheaper and easier to iterate than QAT
- the first NNCF wave did not clearly beat the attempt0 dynamic frontier

Initial batch:

- ignored scopes:
  - `attention_only`
  - `attention_proj_only`
- activation/weight settings:
  - `S8S8`
  - `U8U8`
  - `U8S8 + reduce_range`
- calibration methods:
  - `minmax`
  - `percentile`
- calibration subset sizes:
  - `128`
  - `300`

Deliberate exclusions for this batch:

- do not start with `none`
- do not start with `QOperator`
- do not expand into broad FFN-heavy quantization again unless a result justifies it

Deliverable:

- a ranked static-QDQ shortlist directly comparable to `attention_only`, `attention_proj_only`, and the two first attempt1 NNCF results

### 3.2. Second Priority: Focused NNCF follow-up, not broad NNCF expansion

The only NNCF branch that earned more investigation is the fidelity-oriented branch:

- base seed: `nncf_fidelity_attention_proj_only`

The next NNCF round should be narrow:

- keep ignored scope centered on `attention_proj_only`
- keep `metric=hf_agreement`
- vary `max_drop` more aggressively:
  - `0.002`
  - `0.005`
  - `0.01`
- add one slower bias-correction pass:
  - `--accurate-bias-correction`
- optionally compare `subset_size=128` vs `300`

Do not spend time on another broad six-candidate NNCF matrix yet.

Reason:

- `gold_accuracy + attention_only` already failed cleanly
- `hf_agreement + attention_proj_only` is the only serious NNCF survivor
- the repo needs refinement, not another full first-pass sweep

Deliverable:

- one small NNCF follow-up table that answers whether a tighter fidelity-oriented search can beat the attempt0 `attention_proj_only` baseline without losing `xnli-zh`

### 3.3. Third Priority: Chinese-aware validation and diagnosis

The first attempt1 results make this mandatory.

The repo should create a small Chinese-sensitive validation pack that combines:

- held-out `xnli-zh` search-validation rows
- rows related to the known translated-stress failures
- the specific full-suite `xnli-zh` disagreements observed in the new NNCF runs

Use this pack in two ways:

- as a reporting slice for every promoted attempt1 candidate
- as an optional alternate validation slice for the focused NNCF follow-up

Key question:

- can the repo improve `xnli-zh` without losing the broader suite?

Deliverable:

- a reusable Chinese-sensitive evaluation slice and a short note on whether the next candidates improve that slice

### 3.4. Fourth Priority: QAT as rescue only

QAT should remain gated.

The best current QAT seeds are:

- `nncf_fidelity_attention_proj_only` for fidelity recovery
- `attention_only` for accuracy recovery

The first QAT pass should stay intentionally small:

- `1` epoch
- `1e-5` learning rate
- reduced fine-tune slice
- export back to ONNX
- benchmark only if the pilot completes cleanly

Escalation condition:

- only continue beyond the pilot if the pilot materially improves the held-out validation target or visibly repairs `xnli-zh`

Deliverable:

- at most one or two QAT rescue artifacts, not a broad training campaign

## 4. Evaluation Changes

The repo should tighten the attempt1 evaluation flow before running a large matrix.

### 4.1. Add probe-gated early stopping

The current attempt1 harness still runs the full suite even for candidates that are already clearly weak on the core and hard probes.

That should change.

Updated screening order:

1. core probe
2. hard probe
3. full suite only for promoted candidates
4. runtime and RSS only for serious finalists

Proposed promotion rules to earn a full-suite benchmark:

- accuracy-oriented candidate:
  - must be at least float-level on both core and hard accuracy
  - must not regress `xnli-zh` probe behavior versus float by more than one example
- fidelity-oriented candidate:
  - must be at least `85%` HF agreement on both core and hard probes
  - must not regress `xnli-zh` probe behavior versus float by more than one example

These are screening gates, not final recommendation rules.

### 4.2. Make recommendation gates explicit

A candidate should only be considered a real attempt1 promotion target if it meets one of these bars:

- accuracy promotion:
  - full accuracy strictly above `86.67%`
  - full HF agreement not meaningfully worse than `98.77%`
  - full `xnli-zh` at least `85.33%`
- fidelity promotion:
  - full HF agreement strictly above `98.82%`
  - full accuracy at least `86.51%`
  - full `xnli-zh` at least `85.33%`
- tie-break promotion:
  - ties the best relevant quality metric and materially improves runtime or memory without worsening `xnli-zh`

If a candidate misses those bars, it can still be documented as a research result, but it should not displace the attempt0 frontier.

## 5. Concrete Experiment Batches

### Batch A: Static QDQ frontier sweep

Run this first.

Candidate family:

- `static_attention_only_*`
- `static_attention_proj_only_*`

Grid:

- `S8S8`, `U8U8`, `U8S8 + reduce_range`
- `minmax`, `percentile`
- `128`, `300` example calibration caps

Then:

- core probe all
- hard probe promoted subset
- full suite only top survivors

### Batch B: Focused NNCF fidelity refinement

Run this after Batch A or in parallel if compute time is acceptable.

Candidate family:

- `nncf_fidelity_attention_proj_only` variants only

Grid:

- `max_drop`: `0.002`, `0.005`, `0.01`
- `fast` vs `accurate` bias correction
- optional `subset_size`: `128` vs `300`

Then:

- core probe
- hard probe
- full suite only if probes justify it

### Batch C: Chinese-aware repair pass

Run this only if Batch A or Batch B produces a near-frontier candidate.

Options:

- rerun the best NNCF fidelity candidate with a Chinese-aware validation slice
- run a tiny QAT recovery pass using the best fidelity or accuracy seed

## 6. Recommended Execution Order

1. tighten the attempt1 sweep harness with probe-gated early stopping
2. run the narrow static QDQ frontier sweep
3. benchmark only the surviving static candidates on the full suite
4. run the focused NNCF fidelity refinement batch
5. compare the best static and best NNCF candidates against the attempt0 frontier
6. build and use the Chinese-aware validation slice for any near-frontier candidate
7. run QAT only on the strongest surviving seed
8. run runtime and memory benchmarks only for candidates that genuinely challenge the frontier
9. update the dashboard and recommendation only if the frontier changes

## 7. Bottom Line

The first attempt1 result changed the planning problem.

The repo no longer needs to prove that NNCF mixed-precision tooling works. It does. The repo now needs to answer a narrower question:

- can static QDQ tuning or a tighter fidelity-oriented follow-up beat the current `attention_only` / `attention_proj_only` frontier without worsening the known Chinese weakness?

That is the right next plan for attempt1.
