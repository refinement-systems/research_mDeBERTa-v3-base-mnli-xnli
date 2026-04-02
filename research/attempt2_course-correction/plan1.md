# Attempt 2 Plan 1

Scope: proposed follow-up after `result0.md`.

Execution note as of 2026-04-02:

- the exact carry-forward recreation of `nncf_fidelity_attention_proj_only` succeeded, and it is currently the strongest validated quantized CPU point in `plan1`
- `nncf_fidelity_attention_only_n128_drop0p005` also materialized and reached the validation frontier, but it was slow enough on this machine to be treated as only somewhat applicable
- `nncf_fidelity_attention_only_n128_drop0p002` was stopped and retired as too slow for this machine
- the `n300` pair was not started after the machine-based pruning decision
- CoreML remains blocked at the backend-validity stage, so the active `plan1` run is still CPU-only

`plan0` answered the first course-correction question. It produced a credible CPU recommendation, showed that the tiny dynamic tier is too lossy under the new fidelity objective, and showed that the current CoreML lane is not yet trustworthy in this environment.

So the next work should not be another broad rerun. It should be a narrow follow-up that asks:

- can a small, objective-aligned CPU search beat `nncf_accuracy_attention_only` on size versus fidelity?
- is the old `attention_proj_only` / `nncf_fidelity_attention_proj_only` branch still reproducible enough to matter?
- is CoreML blocked by candidate quality, or blocked earlier by backend execution validity?

## 1. Current Position

`result0.md` established five important facts.

1. The current CPU frontier has a clear middle point.
   - `nncf_accuracy_attention_only` is the best size-versus-fidelity compromise seen so far.
   - It is much smaller than float and much more faithful than the ~323 MiB dynamic baselines.

2. The very small dynamic tier is not good enough.
   - `dynamic_qint8_default`
   - `model_quantized`
   - and by implication `dynamic_qint8_per_channel`
   are useful lower-bound anchors, but not strong default candidates.

3. The `attention_only` / `attention_proj_only` family is now mostly an anchor, not a search destination.
   - `attention_only` is very faithful.
   - It is also almost float-sized.
   - `attention_proj_only` was already dominated on the new objective.

4. The old fidelity-oriented NNCF branch is unresolved, not validated.
   - `nncf_fidelity_attention_proj_only` did not materialize in `plan0`.
   - That means the branch is currently neither promoted nor cleanly retired.

5. CoreML is blocked before frontier search even starts.
   - the CoreML reference run fell back to CPU
   - the runner correctly refused to record that as CoreML data
   - therefore the current CoreML lane has no valid backend-specific float baseline

Those facts imply that the best next step is a narrow CPU follow-up plus a separate CoreML validity investigation.

## 2. First Constraint: Reset Split Hygiene

`plan0` already touched both:

- the original `fidelity_validation` split
- the original `fidelity_test` split

So if `plan1` is used to choose or tune new candidates, those old splits become development evidence, not a fresh final report.

Therefore `plan1` should create a new evaluation pack:

- `calibration_plan1`
  - may reuse the current calibration slices if needed for continuity
- `fidelity_validation_plan1`
  - must be built from rows not used in `plan0` calibration, validation, test, or smoke
- `fidelity_test_plan1`
  - must be held out until the new CPU shortlist is locked
- `smoke_plan1`
  - may reuse the frozen smoke probes, because smoke is diagnostic only

The exact row windows are less important than the discipline:

- disjoint from all `plan0` ranking/reporting rows
- disjoint across roles
- documented in the scratchpad/DB layer with the same provenance rules used in `plan0`

Deliverable:

- a fresh `plan1` dataset pack and a disjointness check before any candidate ranking run

## 3. Mainline Direction: Narrow CPU Search Around The Current Winner

The CPU lane is the only lane that currently has a real answer. So the next research work should stay narrow and try to beat that answer, not reopen the full search space.

### 3.1. Freeze the CPU anchors

Keep these as the fixed comparison set for `plan1`:

- `reference`
- `model_quantized`
- `nncf_accuracy_attention_only`
- `attention_only`

Use them for every `plan1` CPU summary.

Reason:

- `reference` is the ceiling
- `model_quantized` is the practical small-size lower bound
- `nncf_accuracy_attention_only` is the current champion
- `attention_only` is the current high-fidelity quantized anchor

Do not reopen as search families:

- `dynamic_qint8_default`
- `dynamic_qint8_per_channel`
- `attention_proj_only`
- broad family-search layer variants

These may remain as historical context, but they should not drive the next batch.

### 3.2. Run one small objective-aligned NNCF batch

The most interesting unresolved question is:

- can the repo beat `nncf_accuracy_attention_only` with a fidelity-oriented NNCF search centered on the same `attention_only` scope family?

Why this direction makes sense:

- the current winner came from a `gold_accuracy`-oriented branch
- the new research objective is fidelity to float
- so the next batch should align the search objective with the selection objective

Recommended batch:

- quantization mode:
  - `accuracy-control`
- metric:
  - `hf_agreement`
- ignored scope family:
  - `attention_only`
- bias correction:
  - `fast`
- subset sizes:
  - `128`
  - `300`
- `max_drop` values:
  - `0.005`
  - `0.002`

That creates four serious candidates:

- `nncf_fidelity_attention_only_n128_drop0p005`
- `nncf_fidelity_attention_only_n128_drop0p002`
- `nncf_fidelity_attention_only_n300_drop0p005`
- `nncf_fidelity_attention_only_n300_drop0p002`

This is intentionally small.

Do not add:

- `accurate` bias correction
- broad `none` / `attention_proj_only` matrix expansion
- static QDQ reruns
- QAT

The batch should answer one question only:

- can an objective-aligned NNCF refinement produce a new middle frontier point better than `nncf_accuracy_attention_only`?

### 3.3. Promotion rule for the new CPU batch

Run order:

1. materialize each candidate
2. run `smoke`
3. run `fidelity_validation_plan1`
4. lock the CPU frontier
5. run `fidelity_test_plan1` only on the locked points

Selection rule:

- primary metric: float-label agreement
- tie-break: lower mean absolute logit delta
- frontier is still computed on size versus fidelity

Success condition:

- at least one new candidate either:
  - stays below the current `nncf_accuracy_attention_only` size and improves fidelity
  - or stays at similar fidelity and reduces size further

Failure condition:

- no new candidate displaces `nncf_accuracy_attention_only`

If the batch fails, stop the CPU search rather than reopening a broad grid.

## 4. Secondary Direction: Resolve The `nncf_fidelity_attention_proj_only` Branch

`plan0` left one awkward gap:

- `nncf_fidelity_attention_proj_only` was in the carry-forward set
- but it never materialized in the new scratchpad run

That should be resolved explicitly before the repo keeps citing it as a live direction.

Required work:

1. try to recreate the exact artifact from its stored generator provenance
2. if recreation succeeds, benchmark it only as a baseline on `fidelity_validation_plan1`
3. if recreation fails again, record the failure mode and retire the branch from the active shortlist

Important limit:

- do not launch another broad `attention_proj_only` refinement batch unless the exact baseline first becomes reproducible

Reason:

- an irreproducible branch is not a research frontier

## 5. CoreML Should Become An Infrastructure Question First

`plan0` did not show that CoreML quantization is bad.

It showed something earlier:

- the current evaluation path cannot yet produce a trustworthy CoreML float reference on this machine

So `plan1` should not run a new CoreML quantization search.

Instead it should answer one binary question:

- can this repo obtain a valid, non-fallback CoreML reference for the float model?

Recommended CoreML work:

1. verify the exact fallback behavior for the float reference with explicit provider/session logging
2. confirm whether the blocker is:
   - unsupported ops in the current ONNX export
   - an ONNX Runtime CoreML provider limitation
   - or an environment/configuration issue
3. if ONNX Runtime CoreML cannot run the float reference without fallback, stop the ONNX/CoreML quantization lane
4. only then consider a separate CoreML-native baseline path:
   - float CoreML
   - or fp16 CoreML

Deliverable:

- either a valid CoreML reference baseline
- or a short closure note that the current ONNX/CoreML path is not a sound research lane in this environment

Until that is resolved, the correct status remains:

- no CoreML recommendation

## 6. Explicit Non-Goals For `plan1`

Do not spend the next batch on:

- performance benchmarking
- process memory benchmarking
- broad family-search reruns
- broad static-QDQ sweeps
- QAT rescue
- alternate runtimes

Reason:

- `result0` already identified the main CPU middle point
- the next useful work is to test whether that point can be improved narrowly
- everything else is lower priority until the CPU middle point is either displaced or confirmed

## 7. Recommended Execution Order

1. build fresh disjoint `plan1` validation/test splits
2. re-materialize and verify the CPU anchor artifacts
3. attempt exact recreation of `nncf_fidelity_attention_proj_only`
4. run the four-candidate `hf_agreement + attention_only` NNCF batch
5. compute the `plan1` CPU validation frontier
6. run untouched `fidelity_test_plan1` only on the locked frontier
7. update the CPU recommendation or explicitly confirm `nncf_accuracy_attention_only` as still best
8. in parallel or afterward, run the CoreML reference-validity investigation

## 8. Bottom Line

The repo should not treat `result0` as a reason to reopen every branch.

It should treat `result0` as evidence that:

- the CPU question is now narrow
- the CoreML question is now infrastructural
- and the next serious experiment is a small fidelity-oriented NNCF follow-up around `attention_only`

If that narrow CPU batch fails to beat `nncf_accuracy_attention_only`, the most likely conclusion will be:

- `nncf_accuracy_attention_only` remains the CPU recommendation
- the small dynamic tier remains too lossy
- the `attention_only` family remains too large
- and CoreML needs a different baseline/toolchain story before it can re-enter the mainline research plan
