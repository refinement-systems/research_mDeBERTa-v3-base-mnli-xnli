# Attempt 1 Plan 3

Scope: updated research plan after the corrected static-QDQ frontier sweep in `result2.md`.

This document supersedes `plan2.md`. The repo now has enough evidence to narrow the attempt1 direction again.

## 1. Current Position

Attempt1 now has four material findings:

1. The attempt1 infrastructure is operational and flexible.
   The harness supports probe-gated screening, family-scoped search, narrow candidate filters, and per-candidate incremental summary writes.

2. The first NNCF fidelity candidate is still the strongest attempt1 result.
   `nncf_fidelity_attention_proj_only` remains the best attempt1 artifact because it matches the attempt0 fidelity frontier on full-suite HF agreement and stays reasonably close on full accuracy.

3. The corrected static-QDQ sweep did not change the frontier.
   Static QDQ produced real survivors, but none beat the attempt0 dynamic finalists or the best attempt1 NNCF fidelity result.

4. Broad static sweeps are too memory-expensive on this machine.
   A Python process peaking above `50 GB` on a `16 GB` Mac is enough evidence to stop treating large static matrices as the default next move.

## 2. Revised Main Hypothesis

The new mainline hypothesis is:

- the best remaining chance inside the current ONNX/CoreML path is narrow NNCF fidelity refinement around `attention_proj_only`

Corollaries:

- broad static QDQ is no longer the main attempt1 path
- static remains useful only as a targeted side branch when Chinese preservation is the primary question
- QAT stays gated as rescue work for a near-frontier seed, not as a blind next search

## 3. Updated Research Priorities

### 3.1. First Priority: NNCF fidelity micro-batch

This is now the main attempt1 direction.

Base candidate:

- `nncf_fidelity_attention_proj_only`

First micro-batch:

- `metric=hf_agreement`
- `ignored_scope_family=attention_proj_only`
- `preset=mixed`
- `subset_size`:
  - `128`
- `max_drop`:
  - `0.005`
  - `0.002`
- bias correction:
  - `accurate`

Reason:

- this is the only branch that has already shown near-frontier behavior
- the batch is small enough to run without another large search bill
- it directly tests whether stricter fidelity control can recover the remaining gap

If either candidate is promising, expand only one axis at a time:

- compare `fast` vs `accurate` bias correction on the winning `max_drop`
- compare `subset_size=128` vs `300` only after a `128` candidate earns it

Do not reopen the broad six-candidate NNCF matrix yet.

### 3.2. Second Priority: Chinese-aware evaluation pack

The repo should now make the Chinese sensitivity explicit in the search workflow.

Build a small evaluation pack from:

- held-out `xnli-zh` search-validation rows
- known translated-stress examples
- disagreements observed in `result1.md` and `result2.md`

Use this pack in two ways:

- report it for every promoted candidate
- optionally reuse it as a focused validation slice for the narrow NNCF follow-up

Question to answer:

- can the repo improve `xnli-zh` without giving back the broader suite?

### 3.3. Third Priority: Static only as a targeted side branch

Static QDQ should no longer run as a broad matrix.

Allowed static follow-up, if any:

- `minmax` only
- one candidate at a time
- only on already interesting families
- only when the explicit goal is to test Chinese preservation

Most defensible static follow-ups:

- `static_attention_only_u8u8_minmax_n128`
- `static_attention_proj_only_u8s8_rr_minmax_n128`
- `static_attention_proj_only_u8u8_minmax_n300`

Do not include:

- `percentile`
- `entropy`
- whole-family breadth expansion
- another full static grid

If static is revisited technically, the first implementation improvement should be reducing repeated preprocessing / memory overhead rather than expanding the search space.

### 3.4. Fourth Priority: QAT as rescue only

QAT still stays behind the gating line.

Best seed:

- the strongest surviving NNCF fidelity candidate from Priority 1

Pilot constraints:

- `1` epoch
- learning rate `1e-5`
- reduced fine-tune slice
- export back to ONNX before taking the result seriously

Escalate only if the pilot does one of:

- reaches the full-suite frontier on quality
- materially repairs the Chinese-sensitive slice

## 4. Evaluation Rules

### 4.1. Keep probe-gated early stopping

This is now mandatory, not optional.

Screening order:

1. core probe
2. hard probe
3. full suite only for promoted candidates
4. runtime and RSS only for real finalists

### 4.2. Keep recommendation bars explicit

A candidate should only be treated as a true promotion target if it satisfies one of:

- accuracy promotion:
  - full accuracy strictly above `86.67%`
  - full HF agreement not materially worse than `98.77%`
  - full `xnli-zh` at least `85.33%`
- fidelity promotion:
  - full HF agreement strictly above `98.82%`
  - full accuracy at least `86.51%`
  - full `xnli-zh` at least `85.33%`
- tie-break promotion:
  - ties the best relevant quality metric and materially improves runtime or memory without worsening `xnli-zh`

### 4.3. Treat memory as a real selection criterion

After `result2.md`, memory is no longer a secondary operational detail.

Rules:

- do not run broad matrices that are known to trigger heavy swap pressure
- prefer narrow batches that can be stopped and resumed safely
- only measure runtime and RSS for candidates that already challenge the quality frontier

## 5. Concrete Experiment Batches

### Batch A: NNCF fidelity micro-batch

Run this first.

Candidates:

- `attention_proj_only`
- `hf_agreement`
- `subset_size=128`
- `max_drop=0.005`
- `accurate` bias correction

- `attention_proj_only`
- `hf_agreement`
- `subset_size=128`
- `max_drop=0.002`
- `accurate` bias correction

Deliverable:

- one short result table directly compared with:
  - `attention_proj_only`
  - `nncf_fidelity_attention_proj_only`
  - the strongest static survivor

### Batch B: One-axis refinement

Run this only if Batch A gets close.

Allowed expansions:

- `fast` vs `accurate` bias correction on the best `max_drop`
- `subset_size=300` on the best Batch A configuration

Do not expand both axes at once.

### Batch C: Chinese-aware repair pass

Run this only for a near-frontier candidate.

Options:

- rerun the best NNCF fidelity candidate with a Chinese-aware validation slice
- run a tiny QAT rescue pass from that candidate

### Batch D: Optional static spot-check

Run this only if the repo needs a Chinese-preserving fallback artifact.

Limit:

- one static candidate at a time
- `minmax` only
- no broad reruns

## 6. Recommended Execution Order

1. record the static conclusion from `result2.md`
2. retire `percentile` static calibration from the attempt1 mainline
3. build the Chinese-aware evaluation pack
4. run the two-candidate NNCF fidelity micro-batch
5. compare the best new NNCF result against:
   - `attention_proj_only`
   - `nncf_fidelity_attention_proj_only`
   - `attention_only`
6. expand only one NNCF axis if the micro-batch earns it
7. run QAT only for a near-frontier NNCF survivor
8. run runtime and memory benchmarks only for candidates that challenge the frontier
9. update the dashboard only if the recommendation changes

## 7. Bottom Line

Attempt1 no longer needs another broad search.

The updated question is narrower:

- can a small fidelity-oriented NNCF refinement beat or surpass `attention_proj_only` without losing `xnli-zh`, and can it do so without the operational cost seen in the static batch?

That is the most promising remaining research direction on this machine and within the current deployment path.
