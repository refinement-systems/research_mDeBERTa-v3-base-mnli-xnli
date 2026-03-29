# Attempt 1 Result 1

Scope: first benchmark-backed attempt1 results after the attempt1 data-prep and NNCF mixed-precision tooling became operational.

This note analyzes the first two serious NNCF/OpenVINO-style `accuracy-control` candidates run on the new disjoint attempt1 search data.

## 1. What Was Run

The repo now has a working attempt1 path for:

- disjoint calibration and search-validation TSV generation
- NNCF `accuracy-control` ONNX quantization with `mixed` preset
- family-scoped ignored scopes reused from attempt0
- full benchmark screening on core probe, hard probe, and the full suite

The two first real attempt1 candidates were:

| Candidate | Validation Metric | Ignored Scope Family | Artifact |
| --- | --- | --- | --- |
| `nncf_accuracy_attention_only` | `gold_accuracy` | `attention_only` | `models/mdeberta/onnx/candidates/attempt1/nncf_accuracy_attention_only.onnx` |
| `nncf_fidelity_attention_proj_only` | `hf_agreement` | `attention_proj_only` | `models/mdeberta/onnx/candidates/attempt1/nncf_fidelity_attention_proj_only.onnx` |

Shared settings:

- `mode=accuracy-control`
- `preset=mixed`
- `subset_size=300`
- `preprocess=true`
- `fast_bias_correction=true`
- `smooth_quant_disabled=false`
- `672` calibration examples
- `672` validation examples

## 2. Results

Attempt0 reference frontier:

| Candidate | Full Acc | Full HF | Hard Acc | Core Acc | `xnli-zh` Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| `float` | `86.36%` | `100.00%` | `45.90%` | `44.00%` | `86.00%` |
| `attention_only` | `86.67%` | `98.77%` | `55.74%` | `52.00%` | `85.33%` |
| `attention_proj_only` | `86.51%` | `98.82%` | `50.82%` | `40.00%` | `85.33%` |

Attempt1 NNCF results:

| Candidate | Full Acc | Full HF | Hard Acc | Hard HF | Core Acc | Core HF | `xnli-zh` Full | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `nncf_accuracy_attention_only` | `86.15%` | `98.62%` | `44.26%` | `81.97%` | `44.00%` | `80.00%` | `84.67%` | Not competitive |
| `nncf_fidelity_attention_proj_only` | `86.36%` | `98.82%` | `47.54%` | `88.52%` | `40.00%` | `88.00%` | `84.67%` | Interesting, but not a new best |

## 3. Analysis

### 3.1. The attempt1 NNCF path now works end to end

This is a real result, even before discussing quality:

- the disjoint attempt1 data flow works
- long-running NNCF `accuracy-control` runs complete on this repo
- the ONNX candidate artifacts can be benchmarked through the existing harness

That means attempt1 is no longer blocked on infrastructure. The repo can now compare systematic mixed-precision PTQ against the manual attempt0 family search using real benchmark evidence.

### 3.2. `nncf_accuracy_attention_only` is not good enough

This candidate does not beat either attempt0 finalist on the final benchmark that matters:

- full accuracy is below both attempt0 finalists
- full HF agreement is below both attempt0 finalists
- `xnli-zh` full accuracy is below both attempt0 finalists

The interesting wrinkle is that its probe HF agreement is much higher than the attempt0 dynamic finalists. That suggests the NNCF mixed-precision search is capable of preserving HF behavior on the curated stress probes better than the simple dynamic quantization path. But that benefit did not translate into a better balanced full-suite model.

Operationally, this candidate is a dead end as a promotion target.

### 3.3. `nncf_fidelity_attention_proj_only` is the better NNCF result

This is the strongest attempt1 NNCF candidate so far.

It matters because:

- it matches float on full-suite gold accuracy: `86.36%`
- it matches the attempt0 `attention_proj_only` full-suite HF agreement: `98.82%`
- it clearly improves over the first attempt1 NNCF candidate on both full accuracy and full HF agreement

But it still falls short of displacing the attempt0 frontier:

- it does not beat `attention_only` on full accuracy
- it does not beat `attention_proj_only` on full HF agreement; it only ties it
- it is still worse than both attempt0 finalists on `xnli-zh`
- its hard-probe gold accuracy is still below both attempt0 finalists

So the fairest interpretation is:

- this is a viable fidelity-oriented mixed-precision candidate
- it is not a new default quantized recommendation
- it narrows the gap enough that NNCF should not be abandoned outright

### 3.4. The Chinese weakness remains unresolved

This remains the most important quality signal from the first attempt1 results.

Both NNCF candidates ended at:

- `84.67%` on full-suite `xnli-zh`

That is below:

- `85.33%` for both attempt0 dynamic finalists
- `86.00%` for float

So the first attempt1 mixed-precision search did not repair the known Chinese weakness. That keeps the reviewer's suggestion relevant, but it also means the repo should stop expecting a generic mixed-precision search to fix the problem automatically.

## 4. What These Results Mean

The attempt0 conclusion still stands:

- `attention_only` remains the best quantized accuracy-oriented artifact
- `attention_proj_only` remains the best quantized fidelity baseline
- `float` remains the safest overall default

Attempt1 did produce one meaningful update:

- `nncf_fidelity_attention_proj_only` is now the best evidence that systematic mixed-precision search can get close to the attempt0 fidelity frontier without hand-built exclusion sweeps

That is a useful research result, but it is not yet a recommendation change.

## 5. Recommended Next Directions

From these results, the most defensible next steps are:

1. Run the static QDQ matrix next.
   The first NNCF mixed-precision wave is no longer untested, and it did not clearly beat the attempt0 frontier. Static datatype and calibration tuning is now the cheapest serious unexplored axis.

2. If staying on NNCF, tighten the fidelity-oriented path instead of broadening blindly.
   The only NNCF branch that currently looks worth another round is the fidelity-oriented one around `attention_proj_only`. Good follow-ups would be stricter `max_drop`, a narrower validation target, or a Chinese-aware validation slice.

3. Treat QAT as a recovery path, not the next blind search.
   If the repo wants to spend training time, the best current QAT seed is `nncf_fidelity_attention_proj_only` for fidelity recovery or `attention_only` for accuracy recovery.

4. Keep Chinese-specific evaluation explicit in every next round.
   The main unresolved problem is not average accuracy alone. It is failure concentration in `xnli-zh` and related translated stress cases.

## 6. Bottom Line

Attempt1 has now delivered a real benchmark-backed answer:

- systematic NNCF mixed-precision search is viable on this repo
- the first accuracy-oriented NNCF candidate is not competitive
- the first fidelity-oriented NNCF candidate is respectable but still not better than the attempt0 frontier
- the attempt0 recommendation does not change yet

The strongest next research path is no longer "prove NNCF works at all." That is done. The next question is whether static QDQ tuning or a targeted fidelity/QAT recovery pass can beat the current `attention_only` and `attention_proj_only` frontier without reintroducing the known Chinese weakness.
