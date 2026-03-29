# Quantization Plan 3

Scope: future direction after `QUANTIZATION_REPORT_6.md`.

This document is intentionally forward-looking. For the historical record of what has already been tried, see:

- `QUANTIZATION_REPORT_1.md`
- `QUANTIZATION_REPORT_2.md`
- `QUANTIZATION_REPORT_3.md`
- `QUANTIZATION_REPORT_4.md`
- `QUANTIZATION_REPORT_5.md`
- `QUANTIZATION_REPORT_6.md`

## 1. Current Position

The repo now has a stable quality frontier and the first meaningful runtime results.

Current model positions:

- reference / most faithful overall:
  - `models/mdeberta/onnx/model.onnx`
- best quantized accuracy candidate:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`
- best quantized fidelity candidate:
  - `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

Current high-level conclusion from `QUANTIZATION_REPORT_6.md`:

- the quantized finalists are real and benchmarked
- they do improve some gold-label outcomes over float
- but they do not currently buy a meaningful CPU runtime advantage
- so float is still the most defensible default overall

That changes the next-step objective.

The next work should not be "find any better quantized model."

The next work should be:

- finish the operational evidence,
- decide whether quantization is still worth pursuing as a shipping path,
- only continue candidate search if the target is still realistic.

## 2. Primary Goal

The next round should answer this decision cleanly:

- can a quantized model beat float on something that matters operationally, not just on a few benchmark points?

That means future work must be judged on:

1. quality
2. fidelity
3. runtime
4. size

If a candidate cannot improve at least one of those materially without unacceptable regression in the others, it should not advance.

## 3. What To Build Next

### 3.1. Persistent-Session Runtime Benchmark

This is now the highest-priority task.

The current runtime benchmark is useful, but it still launches `builddir/nli` once per example.

That is enough to measure cold load and warm latency separately, but it makes large CoreML sweeps slow and operationally noisy.

Add a benchmark path that:

- loads a model once
- runs many examples through the same process
- reports:
  - cold load time
  - per-example warm latency
  - aggregate warm median and p95
  - optionally memory usage if practical

Why:

- it will make full CoreML runtime benchmarking feasible
- it will better model the common case where a loaded model serves multiple requests

Expected value:

- high

Risk:

- medium

### 3.2. Full CoreML Runtime Benchmark

After the persistent-session path exists, run the same benchmark tiers on CoreML:

- core probe
- optionally hard probe

Candidates:

- float
- `attention_only`
- `attention_proj_only`

Why:

- CPU no longer provides a strong argument for quantization
- CoreML is now the only obvious place where a meaningful runtime win might still exist

Expected value:

- high

Risk:

- low to medium

### 3.3. Combined Benchmark Dashboard

Add one compact output that merges:

- full-suite quality
- hard-probe quality
- core-probe quality
- HF agreement
- `xnli-zh` guardrail
- file size
- CPU runtime
- CoreML runtime

Possible formats:

- JSON
- CSV
- Markdown report

Why:

- the repo now has enough benchmark surfaces that a single summarized view is more useful than reading separate files

Expected value:

- medium

Risk:

- low

### 3.4. Shipping Recommendation Check

Once the full runtime picture is complete, write down one explicit recommendation:

- default model
- optional experimental model
- rationale

Why:

- the repo is close to a decision point, not just an experimentation point

Expected value:

- medium

Risk:

- low

## 4. What To Try Only If Quantization Still Looks Worthwhile

After the runtime picture is complete, only continue quantization search if at least one of these remains plausible:

- meaningful CoreML warm-latency win
- meaningful memory win
- meaningful size reduction that matters operationally

If that bar is met, the next search should be narrow and benchmark-aware.

### 4.1. Targeted `xnli-zh` Repair

The clearest remaining quality weakness is still Chinese sensitivity.

If quantization remains in scope, the next search should explicitly test:

- whether small exclusions around the current finalists improve `xnli-zh`
- whether that improvement preserves the rest of the benchmark ladder

This should be done against:

- core probe
- hard probe
- full suite

Not against anecdotal examples.

### 4.2. Revisit Static Quantization Only If Runtime Still Justifies It

Static quantization should only come back if:

- CoreML or persistent-session runtime still suggests a meaningful upside, and
- current dynamic candidates remain too behaviorally expensive

Otherwise, static quantization is more work without a clear reason to expect better end-to-end results.

### 4.3. Revisit Alternate Export Toolchains Only As An Escalation

Fresh exports or alternate toolchains should now be treated as an escalation path, not a default next step.

Only use them if:

- the current runtime picture still makes quantization attractive, and
- current ORT-based paths clearly plateau

## 5. What Not To Do

### 5.1. Do Not Resume Broad Quantization Sweeps Now

Reason:

- the repo already has a stable frontier
- more broad sweeps are unlikely to produce signal before the operational picture is complete

### 5.2. Do Not Treat Small Accuracy Gains Alone As Enough

Reason:

- `attention_only` already shows that a quantized model can beat float slightly on benchmark accuracy
- that is not enough if runtime and operational behavior do not improve materially

### 5.3. Do Not Ignore The Default-Model Question

Reason:

- the repo now has enough evidence to make a serious default-model decision
- continuing experiments without that decision would blur research and product choices

### 5.4. Do Not Overread The Current CoreML Snapshot

Reason:

- the current CoreML runtime result is only a one-example snapshot
- it is useful, but it is not enough to decide the CoreML story yet

## 6. Acceptance Criteria For Future Quantized Candidates

A future quantized candidate should only be considered better if it satisfies at least one of these:

- higher full-suite accuracy than `attention_only`
- better HF agreement than `attention_proj_only`
- meaningfully better runtime than float in the shipped path
- meaningfully smaller size with no meaningful runtime or quality regression

And it must also satisfy all of these:

- no obvious collapse on the hard probe
- no obvious collapse on the core probe
- no obvious regression on `xnli-zh`
- no strong evidence that float remains the better default overall

## 7. Recommended Execution Order

1. add a persistent-session runtime benchmark path
2. benchmark float vs `attention_only` vs `attention_proj_only` on CoreML using that path
3. add a combined benchmark dashboard
4. write an explicit default-model recommendation
5. only then decide whether further quantization search is justified

## 8. Bottom Line

The benchmarking phase has largely succeeded.

The repo now knows:

- what the best quantized candidates are
- what their quality tradeoffs are
- that CPU runtime does not currently justify preferring them over float

So the next phase should be operational confirmation and product decision-making, not more wide quantization research by default.
