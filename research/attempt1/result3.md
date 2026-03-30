# Attempt 1 Result 3

Scope: updated attempt1 analysis after the `plan3.md` NNCF fidelity micro-batch around `attention_proj_only`.

This note records the first narrow follow-up run after `result2.md` demoted broad static-QDQ search. The goal of this batch was to test whether a smaller, zh-weighted, fidelity-oriented NNCF search could improve on the current `nncf_fidelity_attention_proj_only` result without repeating the operational cost of the static sweep.

## 1. What Was Run

### 1.1. Chinese-aware validation work

Two reusable held-out Chinese evaluation assets were created:

- `benchmarks/nli/xnli-zh-validation-attempt1-zh-extra-validation-skip96-128-per-label.tsv`
- `benchmarks/nli/attempt1-zh-sensitive-validation.tsv`

The extra zh slice contains `384` rows from `facebook/xnli/zh/validation`, beyond the current calibration, search-validation, and fine-tune windows. It is disjoint from:

- the calibration slices
- the attempt1 search-validation slices
- the fine-tune slices
- the final suite `xnli-zh` test slice
- the hard probe
- the core probe

The combined zh-sensitive pack currently contains `480` rows and is intended as a reporting slice rather than a final recommendation gate.

### 1.2. NNCF fidelity micro-batch

The planned micro-batch was:

- `metric=hf_agreement`
- `ignored_scope_family=attention_proj_only`
- `subset_size=128`
- `max_drop`:
  - `0.005`
  - `0.002`

Two bias-correction branches were tested:

1. `accurate`
2. `fast`

Validation mix:

- the original multilingual attempt1 search-validation slices
- plus the new extra zh validation slice

Summaries:

- `benchmarks/nli/attempt1-plan3-nncf-fidelity-summary.csv`
- `benchmarks/nli/attempt1-plan3-nncf-fidelity-fast-summary.csv`

## 2. Results

### 2.1. Accurate bias correction failed operationally

The `accurate` bias-correction branch did not complete successfully on this stack.

Observed behavior:

- the first candidate (`max_drop=0.005`) failed during NNCF ONNX `accuracy-control`
- the failure occurred after the existing SmoothQuant retry path
- the backend errors included:
  - duplicate `nncf_smooth_quant_output` definitions
  - a `KeyError` on `/deberta/embeddings/word_embeddings/Gather`

This was not a quality miss. It was a backend failure in the attempted quantization path.

### 2.2. Fast bias correction completed, but both candidates were screened out

The salvage batch using `fast` bias correction produced two valid ONNX candidates:

| Candidate | Core Acc | Core HF | Hard Acc | Hard HF | Full Suite | Verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `nncf_fidelity_attention_proj_only_n128_drop0p005` | `48.00%` | `80.00%` | `45.90%` | `83.61%` | not run | Screened out |
| `nncf_fidelity_attention_proj_only_n128_drop0p002` | `56.00%` | `80.00%` | `49.18%` | `80.33%` | not run | Screened out |

Both failed the fidelity probe gate because probe HF agreement stayed below the required threshold.

### 2.3. Comparison against the current best attempt1 fidelity candidate

Current best attempt1 NNCF fidelity candidate from `result1.md`:

| Candidate | Core HF | Hard HF | Full Acc | Full HF | `xnli-zh` Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nncf_fidelity_attention_proj_only` | `88.00%` | `88.52%` | `86.36%` | `98.82%` | `84.67%` |

The new micro-batch candidates did **not** improve that direction.

Instead:

- they increased probe gold accuracy somewhat
- but they materially reduced probe HF agreement
- they were too weak to justify even a full-suite benchmark

So the original `nncf_fidelity_attention_proj_only` remains the best attempt1 fidelity-oriented NNCF result.

### 2.4. Chinese-aware comparison did not reveal a hidden gain

The larger disjoint zh validation slice was benchmarked against:

- `float`
- `attention_proj_only`
- `nncf_fidelity_attention_proj_only`
- `nncf_fidelity_attention_proj_only_n128_drop0p005`
- `nncf_fidelity_attention_proj_only_n128_drop0p002`

Result on the `384`-row extra zh slice:

| Candidate | Accuracy | HF Agreement |
| --- | ---: | ---: |
| `float` | `99.48%` | `100.00%` |
| `attention_proj_only` | `99.48%` | `100.00%` |
| `nncf_fidelity_attention_proj_only` | `99.48%` | `100.00%` |
| `nncf_fidelity_attention_proj_only_n128_drop0p005` | `99.48%` | `100.00%` |
| `nncf_fidelity_attention_proj_only_n128_drop0p002` | `99.48%` | `100.00%` |

That means the zh-weighted validation change did **not** buy any observable Chinese-specific improvement for these new candidates.

## 3. Analysis

### 3.1. The narrow plan3 micro-batch is a negative quality result

The intended hypothesis was:

- a tighter `max_drop`
- a smaller `subset_size`
- and extra zh weight in validation

might produce a more disciplined fidelity candidate than the earlier `subset_size=300` run.

The actual result was the opposite:

- the new candidates were weaker on the probe fidelity metrics
- neither was worth promoting to the full suite

So this specific refinement direction did not move the frontier forward.

### 3.2. Probe accuracy and probe fidelity are diverging here

The two screened-out candidates are not uniformly bad.

In particular:

- `drop0p002` reached `56.00%` core accuracy
- `drop0p005` still matched float on hard accuracy

But the batch objective was fidelity, not accuracy. On that objective, both candidates regressed too much.

That suggests this parameter region may be pushing the search toward:

- more aggressive label changes on the curated probes
- despite not visibly improving the Chinese-specific held-out slice

### 3.3. Accurate bias correction is now effectively ruled out on this stack

This is the most important operational conclusion from this batch.

`accurate` bias correction is not just slower. In this repo, on this machine, for this ONNX/NNCF path, it is currently not a dependable research tool.

That means future attempt1 planning should treat:

- `fast` bias correction as the workable NNCF branch
- `accurate` bias correction as a tooling bug / infrastructure issue, not a practical search option

unless the quantization implementation itself is changed.

### 3.4. The new zh-sensitive pack is useful, but not decisive

The Chinese-aware evaluation work was still valuable:

- the repo now has a reproducible disjoint zh-sensitive reporting slice
- the extra zh slice is large enough to be a legitimate held-out check

But this batch also exposed a limitation:

- the zh-sensitive slice is not very discriminative among the current top candidates

So generic zh held-out validation is not enough by itself. If the repo wants a more sensitive Chinese-oriented research axis, it needs a **harder** Chinese stress pack built from:

- disagreement cases
- high-drift cases
- or other explicitly difficult translated examples

### 3.5. Attempt1 PTQ search is starting to look exhausted

At this point attempt1 has tried:

- broad family-scoped dynamic quantization from attempt0
- systematic NNCF mixed-precision PTQ
- static QDQ datatype/calibration sweeps
- a narrow zh-weighted NNCF fidelity refinement

None of these changed the recommendation frontier.

That is strong evidence that further blind PTQ tuning is low-yield.

## 4. What Changed

This result strengthens, rather than weakens, the current frontier conclusion:

1. `attention_only` remains the best quantized accuracy-oriented artifact.
2. `attention_proj_only` remains the best quantized fidelity baseline.
3. `nncf_fidelity_attention_proj_only` remains the best attempt1 NNCF result.
4. The plan3 micro-batch does not create a new promotion candidate.
5. `accurate` bias correction should be treated as non-viable on this current stack.

## 5. Recommended Next Direction

The most defensible next step is no longer another PTQ parameter search.

If attempt1 continues, the next serious research direction should be one of:

1. QAT rescue on a near-frontier seed.
   Most natural seed:
   - `nncf_fidelity_attention_proj_only`

2. A harder Chinese stress-pack construction effort.
   This is evaluation work, not a new model recommendation path by itself.

3. A tooling investigation into the NNCF `accurate` bias-correction failure.
   This only makes sense if the repo explicitly wants to invest in fixing the quantization stack rather than continuing model search.

## 6. Bottom Line

The plan3 micro-batch delivered a clear answer:

- the zh-weighted NNCF fidelity refinement did not improve the attempt1 frontier
- the two new `fast`-bias candidates were screened out before the full suite
- `accurate` bias correction is currently unusable on this stack
- the existing `nncf_fidelity_attention_proj_only` result remains the best attempt1 NNCF candidate

This makes attempt1’s PTQ story much clearer:

- systematic PTQ search has been explored enough to justify diminishing-confidence assumptions
- the next meaningful step, if any, should be training-based recovery or a tooling fix, not another blind PTQ sweep
