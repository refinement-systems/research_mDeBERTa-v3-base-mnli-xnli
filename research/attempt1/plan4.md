# Attempt 1 Plan 4

Scope: updated research plan after the negative `result3.md` micro-batch.

This document supersedes `plan3.md`. The repo now has enough evidence to stop treating additional PTQ parameter tuning as the main attempt1 path.

## 1. Current Position

Attempt1 now has five strong conclusions:

1. The attempt1 research tooling is operational.
   The repo can generate disjoint calibration and validation slices, build structured zh-sensitive held-out packs, run NNCF accuracy-control experiments, and screen candidates through probe-gated evaluation.

2. Broad static QDQ did not move the frontier.
   It produced some interesting Chinese-preserving survivors, but no promotion candidate.

3. Narrow NNCF fidelity refinement also did not move the frontier.
   The plan3 micro-batch produced valid candidates, but both were screened out by the fidelity probe gate.

4. `accurate` bias correction is not usable on the current NNCF/ONNX stack.
   This is now an infrastructure limitation, not just a slow option.

5. The current recommendation frontier is unchanged.
   - `attention_only` remains the best quantized accuracy artifact
   - `attention_proj_only` remains the best quantized fidelity baseline
   - `nncf_fidelity_attention_proj_only` remains the best attempt1 NNCF artifact

## 2. Revised Main Hypothesis

The updated mainline hypothesis is:

- further blind PTQ tuning is now low-yield

Corollaries:

- attempt1 should stop opening new PTQ grids by default
- the next meaningful model-quality step is QAT rescue on a near-frontier seed
- the next meaningful evaluation step is building a **harder** Chinese stress pack, not just another generic zh held-out slice

## 3. Updated Research Priorities

### 3.1. First Priority: QAT rescue on the best near-frontier seed

If the repo wants to continue attempt1 model work, this is now the main path.

Best seed:

- `nncf_fidelity_attention_proj_only`

Why this seed:

- it remains the strongest attempt1 NNCF result
- it is already close to the fidelity frontier
- it is a more credible starting point for recovery than the newly screened-out plan3 candidates

QAT pilot:

- `1` epoch
- learning rate `1e-5`
- reduced fine-tune slice
- export back to ONNX before taking the result seriously
- benchmark in the usual order:
  - core probe
  - hard probe
  - full suite only if probes justify it

Do not start with multiple seeds.

Only add a second QAT seed if the first pilot completes cleanly and still leaves a plausible gap:

- optional second seed:
  - `attention_only`

### 3.2. Second Priority: Harder Chinese stress-pack construction

The current zh-sensitive pack is useful, but not discriminative enough.

The repo should build a more selective Chinese stress pack from:

- held-out zh disagreement cases
- high logit-drift cases
- examples where candidate labels stay equal but confidence structure changes materially
- known translated-stress patterns already observed in attempt0 / attempt1 logs

Important rule:

- this stress pack should be a reporting / diagnostic slice, not a recommendation gate by itself

Deliverable:

- a compact zh stress TSV that is harder than the generic held-out zh validation slice
- a short note showing whether current frontier models actually differ on it

### 3.3. Third Priority: Tooling fix branch only if explicitly desired

There is now a clear tooling problem:

- NNCF `accurate` bias correction fails on this stack

This is not the default next step, but it is a legitimate technical branch if the repo wants to invest in quantization infrastructure.

Possible focus:

- isolate a minimal reproduction of the backend failure
- determine whether the failure is caused by:
  - SmoothQuant retry interaction
  - graph preprocessing interaction
  - ONNX graph naming / duplication behavior
- only then decide whether `accurate` bias correction should be reopened

This branch should be treated as:

- infrastructure work
- not as a main research path for improving model quality directly

### 3.4. Fourth Priority: Freeze further PTQ sweeps by default

Unless a new, sharply motivated idea appears, do not spend attempt1 time on:

- broader NNCF parameter sweeps
- more static QDQ matrices
- more `subset_size` / `max_drop` tuning around the same plan3 region
- `accurate` bias-correction search variants

The current evidence is strong enough that these are low-return uses of time and machine budget.

## 4. Evaluation Rules

### 4.1. Keep probe-gated screening

This remains mandatory.

Order:

1. core probe
2. hard probe
3. full suite only for promoted candidates
4. runtime and RSS only for genuine finalists

### 4.2. Keep recommendation bars unchanged

Accuracy promotion:

- full accuracy strictly above `86.67%`
- full HF agreement not materially worse than `98.77%`
- full `xnli-zh` at least `85.33%`

Fidelity promotion:

- full HF agreement strictly above `98.82%`
- full accuracy at least `86.51%`
- full `xnli-zh` at least `85.33%`

Tie-break promotion:

- ties the best relevant quality metric and materially improves runtime or memory without worsening `xnli-zh`

### 4.3. Keep memory as a first-class constraint

After `result2.md` and `result3.md`, attempt1 should treat machine cost as part of experiment selection.

Rules:

- avoid large matrices that have already shown poor return
- prefer narrow, restartable runs
- only run high-memory evaluation or runtime measurements for genuine finalists

## 5. Concrete Next Batches

### Batch A: QAT pilot on `nncf_fidelity_attention_proj_only`

Run this first if continuing model work.

Goal:

- see whether a small training-based recovery step can close the remaining fidelity / `xnli-zh` gap without abandoning the deployment path

Deliverable:

- one exported ONNX artifact
- one benchmark table against:
  - `attention_proj_only`
  - `nncf_fidelity_attention_proj_only`
  - `attention_only`

### Batch B: Hard Chinese stress-pack build

Run this next if the repo wants a better diagnostic axis regardless of model changes.

Goal:

- produce a smaller, more informative zh slice that actually separates current candidates

Deliverable:

- one TSV
- one short model comparison note on that TSV

### Batch C: Optional infrastructure branch

Run this only if the repo wants to debug the quantization stack itself.

Goal:

- understand and isolate the `accurate` bias-correction failure

Deliverable:

- either a reproducible bug note
- or a confirmed working configuration that justifies reopening that branch

## 6. Recommended Execution Order

1. record the plan3 negative result from `result3.md`
2. stop default PTQ expansion
3. decide whether attempt1 continues as:
   - QAT rescue
   - evaluation-only stress-pack work
   - tooling investigation
4. if continuing model work, run the single-seed QAT pilot first
5. benchmark the QAT artifact with probe-gated screening
6. only if the QAT pilot is promising, consider:
   - a second seed
   - a longer fine-tune
7. independently, build the harder zh stress pack when better diagnostics are needed

## 7. Bottom Line

Attempt1 no longer needs another PTQ search plan.

The updated question is:

- is the repo willing to do training-based recovery or stack-level tooling work, or is the correct conclusion that the current attempt0 frontier already captures the best practical quantized options?

That is the right decision point after `result3.md`.
