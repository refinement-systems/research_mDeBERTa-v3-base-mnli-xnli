# Final Research Summary

Scope: cross-attempt summary of the quantization and backend work from `attempt0` through `attempt5`.

This document summarizes what was tried, what changed the frontier, what did not, and what the current conclusions are.

## 1. Final State

The work ended with three stable conclusions.

1. Faithful reference model:
   - `models/mdeberta/onnx/model.onnx`

2. Default quantized CPU deployment recommendation:
   - `nncf_accuracy_attention_only`

3. Current CoreML conclusion on this machine:
   - do not recommend the current ORT CoreML path as a primary deployment backend

There is no longer an open broad search question in this line of work. CPU selection is closed for the bounded catalog, and the CoreML lane is effectively closed for the current ORT path on this machine.

## 2. Attempt-By-Attempt Summary

### 2.1. Attempt 0

This was the baseline-establishment and first-frontier attempt.

What was tried:

- validate float ONNX against HF
- benchmark the published quantized export
- build deterministic MNLI/XNLI benchmark slices
- add single-example drift debugging
- run dynamic quantization and family-scoped exclusion search
- benchmark runtime and memory

What mattered:

- the float ONNX export was faithful and became the local ground truth
- the published quantized export was the real outlier
- the key structural insight was that FFN dense quantization caused most of the damage, while attention-side quantization was comparatively safe

What frontier it created:

- `attention_only`: best quantized accuracy-oriented artifact
- `attention_proj_only`: best quantized fidelity-oriented artifact

End state of attempt0:

- float remained the default
- `attention_only` became the recommended experimental quantized artifact
- `attention_proj_only` became the fidelity baseline for later comparisons

Why float stayed default:

- quantized latency gains were only modest
- CPU memory gains were weak
- CoreML memory behavior was worse for the quantized finalists

### 2.2. Attempt 1

This was the systematic PTQ/static/QDQ/QAT clarification pass.

What was tried:

- disjoint calibration and validation split generation
- NNCF mixed-precision PTQ
- static QDQ sweeps
- Chinese-focused held-out evaluation
- a narrow QAT rescue pilot

What worked:

- the tooling and split discipline improved substantially
- family-scoped exclusions could be reused across methods
- `nncf_fidelity_attention_proj_only` became a valid benchmark-backed candidate
- the harder zh stress-pack became a useful diagnostic slice

What did not change the frontier:

- first NNCF wave did not beat attempt0 finalists
- static QDQ was viable but not competitive enough
- narrow PTQ refinement did not improve the frontier
- the QAT pilot did not yield a usable artifact

End state of attempt1:

- attempt0 frontier still stood
- `attention_only` remained best on accuracy
- `attention_proj_only` remained best on fidelity
- `nncf_fidelity_attention_proj_only` was credible, but not a new default

### 2.3. Attempt 2

This was the methodological course correction.

What changed:

- backend-specific float references became explicit
- candidate selection moved to fidelity-vs-size under locked validation/test discipline
- the SQLite-backed study workflow became the main evaluation framework

What mattered:

- `nncf_accuracy_attention_only` emerged as the CPU lane winner
- it had explicit catalog provenance and reproducible study-run state

CoreML outcome in attempt2:

- no valid CoreML winner
- backend-validity and fallback issues blocked a real CoreML recommendation

End state of attempt2:

- CPU winner: `nncf_accuracy_attention_only`
- CoreML: non-result

### 2.4. Attempt 3

This was the first narrow CoreML-specific execution.

What was tried:

- run the CoreML lane outside the sandbox
- compare `reference` vs `reference_fp16`
- keep the attempt fidelity-focused and deliberately skip runtime benchmarking

What mattered:

- both `reference` and `reference_fp16` ran successfully on CoreML
- `reference_fp16` matched the CoreML float reference on both validation and test under the smaller attempt3 pack
- `reference_fp16` cut on-disk size by about half

Why this did not settle the backend question:

- no runtime or memory benchmarks were included
- the run only established that `fp16` preserved model behavior on a smaller pack

End state of attempt3:

- provisional CoreML baseline recommendation: `reference_fp16`
- this remained provisional because operational performance had not yet been measured

### 2.5. Attempt 4

This was the full CPU deployment study and the most important closing attempt.

What changed:

- CPU-only focus
- broad attempt4 dataset pack
- gold-label accuracy restored as a first-class metric
- runtime, RSS, on-disk size, and locked-final reporting integrated into one study

What was tried:

- bounded CPU catalog including float, naive dynamic baselines, attempt0/attempt1 carry-forwards, and selected static QDQ survivors
- development gates on MNLI, ANLI, float-label agreement, and peak RSS
- persistent CPU benchmarking on validation-complete candidates
- locked final test only on development survivors
- HANS and WANLI stress reporting

What mattered:

- the small ~323 MiB dynamic tier was too lossy
- the near-float static tier was not competitive enough under the frozen gates
- the only viable quantized family was the mid-size NNCF tier around `465-485 MiB`

Locked frontier:

- `nncf_accuracy_attention_only`
- `nncf_fidelity_attention_only_n128_drop0p005`
- `nncf_fidelity_attention_proj_only`
- `reference`

Final CPU recommendation:

- `nncf_accuracy_attention_only`

Operational reading:

- wins on size, steady RSS, and cold load
- loses materially on persistent warm latency

So the real CPU conclusion is two-profile:

- storage-and-memory-sensitive CPU deployment:
  - `nncf_accuracy_attention_only`
- warm-latency-sensitive CPU deployment:
  - `reference`

Attempt4 closed the CPU search for this bounded catalog.

### 2.6. Attempt 5

This was the CoreML follow-up after CPU closure.

Planned question:

- can CoreML beat the closed CPU recommendation on this Apple-silicon machine?

What was actually run in the short preflight:

- `reference`
- `reference_fp16`
- broad attempt4 validation pack
- persistent CoreML runtime and RSS benchmark

What mattered:

- both candidates passed the development fidelity gates
- `reference_fp16` preserved model behavior almost perfectly
- but the operational readout was poor

Observed CoreML behavior:

- CoreML float warm latency was roughly similar to CPU float, but load time and memory were much worse
- CoreML fp16 reduced size and RSS versus CoreML float
- but CoreML fp16 warm latency regressed badly
- ORT CoreML partitioning was weak, especially for fp16

End state of attempt5:

- current ORT CoreML path is not competitive enough on this machine
- do not spend a full locked-final CoreML study on the current path
- keep the CPU recommendation as the deployment default

The only remaining optional follow-up is archival:

- run `nncf_accuracy_attention_only` once on CoreML to document how the closed CPU winner transfers

That is not needed to answer the main backend question.

## 3. What Changed The Frontier

The attempts that actually changed the frontier were:

- `attempt0`
  - established float ONNX as the faithful baseline
  - produced `attention_only` and `attention_proj_only`

- `attempt2`
  - reframed the study around backend-specific float fidelity and locked split discipline
  - elevated `nncf_accuracy_attention_only` as the CPU lane winner

- `attempt4`
  - closed the CPU deployment question under broader metrics
  - confirmed `nncf_accuracy_attention_only` as the default quantized CPU recommendation

- `attempt5`
  - closed the CoreML lane negatively for the current ORT path on this machine

## 4. What Was Tried But Did Not Change The Recommendation

These workstreams were useful, but did not produce a new default:

- naive dynamic re-quantization variants
- broad exclusion policies around late blocks or broad attention scopes
- static QDQ sweeps
- narrow NNCF fidelity micro-batches
- QAT rescue pilot
- broader Chinese-sensitive validation as a selector
- early CoreML fp16 fidelity-only validation without runtime measurement

Many of these were still valuable because they clarified what not to spend time on next.

## 5. Final Recommendations

### 5.1. Reference model

Use as the faithful baseline:

- `models/mdeberta/onnx/model.onnx`

### 5.2. CPU deployment

Default quantized CPU recommendation:

- `nncf_accuracy_attention_only`

Use CPU float instead when persistent warm latency is the main objective:

- `reference`

### 5.3. CoreML deployment

Current recommendation:

- do not recommend the current ORT CoreML path as the main deployment backend on this machine

Reason:

- fidelity is acceptable
- operational behavior is not
- the current partitioning/readout does not beat the closed CPU recommendation

## 6. What Should Not Be Reopened

Do not reopen without a new explicit objective:

- broad CPU quantization search
- broader static-QDQ matrices
- more blind NNCF parameter sweeps
- a large integer CoreML search on the current ORT path

Those questions are no longer the bottleneck.

## 7. What A Future Revisit Would Need

If this research is revisited later, it should be because one of these changed:

- deployment priorities shift strongly toward warm latency
- a materially smaller artifact is required
- a different CoreML/runtime path becomes available
- a new backend or execution provider changes the operational tradeoff

If that happens, it should be treated as a new attempt with a new objective, not as an extension of the already-closed search.

## 8. Bottom Line

Across all attempts, the work converged from broad exploratory quantization into a backend-aware deployment conclusion.

The final practical outcome is:

- faithful baseline:
  - `model.onnx`
- default quantized CPU deployment artifact:
  - `nncf_accuracy_attention_only`
- fastest CPU choice:
  - `reference`
- CoreML for the current ORT path on this machine:
  - not recommended as the primary deployment lane

That is the end state supported by the accumulated evidence in `attempt0` through `attempt5`.
