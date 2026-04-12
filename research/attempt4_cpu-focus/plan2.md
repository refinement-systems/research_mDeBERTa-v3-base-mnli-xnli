# Attempt 4 CPU Focus Plan 2

Scope: next actions after `result1.md`.

`result1.md` closes the bounded CPU quantization search. The repo now has a justified CPU recommendation for this catalog, and the next work should reflect that instead of reopening the grid.

## 1. CPU Search Is Closed

Freeze these conclusions unless a new objective is declared.

- default quantized CPU recommendation:
  - `nncf_accuracy_attention_only`
- acceptable locked alternatives:
  - `nncf_fidelity_attention_only_n128_drop0p005`
  - `nncf_fidelity_attention_proj_only`
- fastest CPU choice:
  - `reference`

Do not start another broad CPU quantization sweep.

The current study already answered the useful CPU questions:

1. The dynamic ~323 MiB tier is too lossy.
2. The near-float static/attention anchors are not competitive under the frozen gates.
3. The only viable quantized family is the mid-size NNCF tier.
4. The default winner is `nncf_accuracy_attention_only`.

## 2. Deployment Reading

The result supports two operational profiles.

### 2.1. Storage-and-memory-sensitive CPU deployment

Use:

- `nncf_accuracy_attention_only`

Reason:

- `56.3%` smaller on disk than float
- `40.3%` lower steady RSS than float
- `24.4%` faster cold load than float
- only `0.076` locked-final gold-accuracy points below float

### 2.2. Warm-latency-sensitive CPU deployment

Use:

- `reference`

Reason:

- all quantized locked survivors were materially slower on persistent warm latency
- the winning quantized artifact is about `53.6%` slower on warm inference than float

That distinction matters. `plan2` should not pretend one artifact is best for every CPU operating mode.

## 3. Immediate Repo Tasks

The next concrete tasks should be narrow and mechanical.

1. Write the result into the research record.
   - done via `result1.md`

2. Promote the CPU recommendation into any tracked deployment notes or scripts that currently point at the old CPU default.

3. Preserve the exact winning artifact identity in a stable place.
   - keep the scratchpad evidence
   - add a stable pointer or note to `scratchpad/attempt4_cpu_focus/candidates/attempt1/nncf_accuracy_attention_only.onnx`

4. Preserve the result interpretation next to the artifact choice.
   - memory/storage win
   - not a warm-latency win

5. Keep the full attempt4 evidence package intact.
   - `attempt4-cpu-summary.*`
   - per-dataset / per-language outputs
   - validation/test/stress summaries
   - runtime benchmark CSVs
   - final manifest

## 4. What Not To Do Next

Do not spend more CPU quantization time on:

- more dynamic ORT variants
- more static-QDQ grid expansion
- more attention-scope family-search variants
- another NNCF parameter sweep

Reason:

- the search outcome is already structurally clear
- more CPU quantization variants are unlikely to change the core tradeoff that emerged:
  - quantization reduced size and RSS
  - quantization did not improve persistent warm latency

If CPU latency becomes the next real problem, that should be treated as a different workstream:

- runtime engineering
- ORT/session tuning
- threading and batch policy
- possibly alternate runtimes

It should not be disguised as another quantization search.

## 5. Next Research Direction

The original staging decision was:

- finish CPU first
- then return to CoreML

Now that CPU selection is closed, the next research lane should leave `attempt4` and move to a CoreML-specific follow-up.

Recommended starting point:

1. Keep CPU frozen at `nncf_accuracy_attention_only`.
2. Reuse the attempt3 lesson that `reference_fp16` is the most plausible CoreML baseline.
3. Evaluate CoreML under the same disciplined reporting model:
   - backend-specific reference
   - gold accuracy
   - per-language outputs
   - runtime and memory

The first CoreML question should be narrow:

- can `reference_fp16` or another minimal CoreML-native candidate beat the CPU recommendation on Apple-device operational objectives without materially degrading task accuracy?

Do not start CoreML with a large integer search.

Start with:

- `reference`
- `reference_fp16`

Only expand from there if the baseline lane is valid and worth pursuing.

## 6. If CPU Must Be Revisited Later

Only reopen CPU work if one of these becomes true.

1. Production constraints say warm latency matters more than size and RSS.
2. A deployment target requires a materially smaller artifact than the current winner.
3. A new runtime or backend changes the latency picture enough to justify a fresh study.

If any of those happen, treat the next CPU revisit as a new attempt with a new stated objective.

Do not retroactively reinterpret `attempt4`.

## 7. Bottom Line

`plan2` is simple:

- freeze the CPU winner
- propagate it into deployment-facing notes
- keep the full evidence package
- stop spending time on more CPU quantization search
- move the research focus to the next real unanswered question, which is CoreML after CPU closure
