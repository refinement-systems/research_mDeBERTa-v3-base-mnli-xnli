# Quantization Research Conclusion

This document synthesizes `QUANTIZATION_REPORT_1.md` through `QUANTIZATION_REPORT_7.md` and `QUANTIZATION_RECOMMENDATION_1.md` into one conclusion. It follows the documented end state of the research rather than the current checked-in runtime default. `QUANTIZATION_REPLICATION.md` remains the source of truth for reproduction details, and `QUANTIZATION_PLAN_1.md` through `QUANTIZATION_PLAN_3.md` are used here only to explain how the research direction shifted over time.

## Executive Summary

The quantization work succeeded in one important sense: it found two real, benchmarked quantized finalists instead of stopping at the poor published quantized export. But it did not show that quantization should replace the float model as the repo default.

The final documented lineup is:

- default model: `models/mdeberta/onnx/model.onnx`
- optional experimental quantized model: `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`
- quantized fidelity baseline: `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

At the end of the research, the strongest consolidated numbers were:

| Candidate | Role | Full-Suite Accuracy | HF Agreement | `xnli-zh` Full Accuracy | Most Important Reading |
| --- | --- | ---: | ---: | ---: | --- |
| `float` | default / faithful reference | `1684/1950 = 86.36%` | `1950/1950 = 100.00%` | `86.00%` | Best default overall because it is the faithful ONNX baseline, loads fastest, and has the best memory profile. |
| `attention_only` | accuracy-oriented quantized finalist | `1690/1950 = 86.67%` | `1926/1950 = 98.77%` | `85.33%` | Best quantized model for gold-label accuracy and the recommended experimental quantized path. |
| `attention_proj_only` | fidelity-oriented quantized finalist | `1687/1950 = 86.51%` | `1927/1950 = 98.82%` | `85.33%` | Slightly closer to HF than `attention_only`, but not enough to change the overall recommendation. |

The operational caveat is also clear in the final dashboard and recommendation:

- quantized finalists are only modestly faster in persistent-session warm latency
- `attention_only` is best on CPU persistent warm median: `502.312 ms` vs `518.880 ms` for float
- `attention_proj_only` is best on CoreML persistent warm median: `501.197 ms` vs `518.399 ms` for float
- those gains are real, but only about `2-3%`
- memory does not strengthen the quantization case; CPU RSS is effectively flat, and float is much leaner than either quantized finalist on CoreML

So the research ended with a role-based outcome, not a single universal winner:

- float remained the repo default
- `attention_only` became the accuracy-oriented experimental quantized finalist
- `attention_proj_only` became the fidelity-oriented quantized baseline for research comparisons

## 1. What Was Attempted

The work unfolded in distinct phases, and the plans show that the goal changed as better evidence arrived.

### Phase 1: Prove the reference before tuning quantization

The first step was not to search for better quantization, but to establish a trustworthy baseline:

- compare Hugging Face / PyTorch against the exported float ONNX model
- compare both against the published quantized ONNX artifact
- add tooling to inspect logits and tokenizer behavior so preprocessing drift would not be confused with quantization drift

This phase answered the most important foundational question early:

- `models/mdeberta/onnx/model.onnx` was faithful to HF
- `models/mdeberta/onnx/model_quantized.onnx` was the outlier

That changed the whole project. From that point on, float ONNX, not the published quantized export, became the local ground truth.

### Phase 2: Try naive dynamic quantization and simple candidate generation

Fresh dynamic candidates were generated from the faithful float export, including default, per-channel, `MatMul`-only, and `quint8` variants. These were important negative experiments because they established that naive re-quantization was not enough.

The result was consistent:

- the fresh naive candidates still drifted badly from HF
- they did not repair the core behavior problem exposed by the original failure case

### Phase 3: Use single-example drift debugging to find a first workable candidate

The repo then added activation-drift debugging and exclusion suggestions. This phase was centered on the original probe pair:

- premise: `Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU`
- hypothesis: `Emmanuel Macron is the President of France`

That pair was valuable because it clearly separated faithful from unfaithful artifacts:

- HF and float ONNX strongly preferred `contradiction`
- the published quantized model flipped into the wrong `neutral` regime

Single-example debugging produced the first meaningfully better candidate:

- `dynamic_qint8_matmul_exclude_suggested.onnx`

This mattered because it proved that selective exclusion could help. It also exposed a limitation that shaped all later plans:

- tuning around one sentence pair can find a better local fix
- it does not prove general quality

### Phase 4: Replace anecdotal checks with real benchmark slices

The next major step was methodological rather than architectural:

- build deterministic MNLI and XNLI benchmark slices
- move evaluation away from Topical Chat proxy behavior
- standardize on a non-overlapping suite of `1950` examples

That full suite became:

- `mnli-validation_matched-200-per-label.tsv`
- `mnli-validation_mismatched-200-per-label.tsv`
- `xnli-de-test-50-per-label.tsv`
- `xnli-en-test-50-per-label.tsv`
- `xnli-es-test-50-per-label.tsv`
- `xnli-fr-test-50-per-label.tsv`
- `xnli-zh-test-50-per-label.tsv`

This was the turning point from anecdotal debugging to benchmark-backed research.

### Phase 5: Try preprocessing, static quantization, and broad exclusion policies

Several follow-up ideas were tried after the first suggested candidate:

- preprocessing before quantization
- a static QDQ smoke test
- broad late-layer exclusions
- broad Q/K/V exclusions

These attempts were useful because they ruled out common but overly simple fixes:

- preprocessing did not materially improve fidelity
- the first static path did not outperform the better dynamic candidates
- broad late-layer float policies could repair the Angela/Macron pair but regress badly on the real benchmark slices
- broad attention-side exclusions were too crude to identify the real source of damage

### Phase 6: Run a structured exclusion-family search

After report 1, the first plan shifted from "fix the bad pair" to "search systematically and judge everything on the benchmark suite." That produced the first true breakthrough.

The family search tested structured policies such as:

- `attention_only`
- `ffn_only`
- progressively larger FFN-float families
- late-block float families
- attention-output float variants

This was the most important successful search phase in the whole project. It showed that:

- quantizing attention-side weight projections was comparatively safe
- quantizing FFN dense layers was the main source of damage

The standout result was:

- `attention_only`: `1689/1950 = 86.62%`, slightly above float on the structured-family run

That replaced `dynamic_qint8_matmul_exclude_suggested.onnx` as the working quantized baseline.

### Phase 7: Narrow the search around the new frontier

Once `attention_only` existed, the research question changed again. The goal was no longer "find any quantized model that works," but:

- can the repo improve on `attention_only`
- or at least identify a cleaner accuracy-versus-fidelity tradeoff

That led to the attention-focused follow-up sweep:

- `attention_only`
- `attention_proj_only`
- upper-layer float variants derived from both

This sweep did not find a strict winner over `attention_only`. Instead, it established the long-lived frontier:

- `attention_only` was still best on benchmark accuracy
- `attention_proj_only` was slightly better on agreement with the float/HF reference

### Phase 8: Benchmark directly against HF, not only against float

The next stage added full HF-vs-ONNX benchmarking so the repo could measure:

- gold-label accuracy
- HF agreement
- float agreement
- mean and max logit drift
- concrete disagreement examples

This mattered because the repo now had two credible finalists and needed a more direct answer to:

- which candidate behaves more like the original model

The answer was precise:

- `attention_only` kept the accuracy lead
- `attention_proj_only` kept a small but real fidelity lead
- float stayed effectively label-identical to HF over the whole `1950`-example suite

### Phase 9: Build fixed hard-case and core probes

After the full-suite behavior was understood, the project shifted again, matching the later plans:

- stop broad random sweeps
- freeze benchmark gates
- make future regressions easier to detect

That produced two stress assets:

- a `61`-row hard probe built from HF disagreements, top-drift examples, and finalist disagreements
- a `25`-row core probe built as a fast, deterministic screen that still preserved source diversity, language diversity, disagreement cases, and a Chinese tripwire

These probes did not replace the full suite. They made it harder to overfit future search work.

### Phase 10: Measure runtime, then memory, then write the repo-level recommendation

The last phase stopped asking "which quantized artifact is best?" and started asking the more important product question:

- should any quantized model be preferred at all

To answer that, the repo added:

- CPU and CoreML cold-start benchmarks
- CPU and CoreML persistent-session benchmarks
- in-process resident-memory and peak-RSS sampling
- a generated dashboard and recommendation

That final phase closed the operational picture:

- quantization modestly improved warm persistent latency
- quantization did not improve cold-load behavior
- size savings were small
- CPU memory was basically unchanged
- CoreML memory was materially worse for the quantized finalists

That is why the research ended with float as the default even after finding two credible quantized finalists.

## 2. What Worked

Several conclusions held up from the first reports all the way through the final recommendation.

### 2.1. Establishing the faithful baseline worked

The best early decision was to validate the reference before tuning candidates:

- `models/mdeberta/onnx/model.onnx` tracked HF closely from the beginning
- later full-suite benchmarking confirmed that float remained `100.00%` aligned with HF labels on all `1950` benchmark examples

Without that step, later "improvements" would have been judged against the wrong baseline.

### 2.2. The benchmark and probe tooling worked

The project became much more reliable once it switched from anecdotal examples to layered evaluation:

- the full non-overlapping MNLI/XNLI suite gave the broad ranking
- the hard probe preserved the known drift hotspots
- the core probe provided a cheap first-stage filter
- paired HF-vs-ONNX benchmarking separated label agreement from logit stability

This was the right evaluation framework for a project like this. It is one of the main research outputs, not just supporting tooling.

### 2.3. Leaving FFN dense matmuls in float worked

The structured-family search produced the single most important model insight in the repo:

- FFN dense quantization was the dominant source of behavioral damage
- attention-side quantization was comparatively safe

That insight directly led to both finalists:

- `attention_only`
- `attention_proj_only`

### 2.4. `attention_only` worked as the best accuracy-oriented quantized model

`attention_only` became the strongest quantized candidate when judged by gold-label quality:

- full-suite accuracy: `1690/1950 = 86.67%`
- net gain over float: `+6` correct examples
- hard-probe accuracy: `34/61 = 55.74%`
- core-probe accuracy: `13/25 = 52.00%`

It also delivered the best CPU persistent warm latency among the finalists:

- `502.312 ms` vs `518.880 ms` for float

That combination made it the correct experimental quantized recommendation.

### 2.5. `attention_proj_only` worked as the best fidelity-oriented quantized model

`attention_proj_only` never became the universal winner, but it did hold a stable role:

- best quantized full-suite HF agreement: `1927/1950 = 98.82%`
- slightly lower mean max logit drift than `attention_only`
- much closer behavior on the original Angela/Macron probe

It also had the best CoreML persistent warm latency among the quantized finalists:

- `501.197 ms` vs `518.399 ms` for float

That justified keeping it as the fidelity-oriented research baseline even though it did not become the preferred experimental quantized default.

### 2.6. Persistent-session latency gains were real, even if small

The runtime work did find a real operational upside:

- quantized finalists were modestly faster in warm persistent-session serving
- the gains were repeatable on both CPU and CoreML

What worked here was not a dramatic speedup, but the measurement discipline. The repo stopped guessing and quantified the tradeoff clearly enough to support a recommendation.

## 3. What Didn't

The negative findings are nearly as important as the positive ones because they explain why the search converged where it did.

### 3.1. The published quantized export did not work

The original `models/mdeberta/onnx/model_quantized.onnx` was the earliest and clearest failure:

- it was not faithful to HF
- it failed the Angela/Macron probe badly
- it never became part of the final recommended lineup

This is the reason the research could not stop at "there is already a quantized model in the repo."

### 3.2. Naive ORT dynamic quantization did not work

Freshly generated default, per-channel, and `MatMul`-only candidates were still far from HF. This mattered because it ruled out the easy explanation that the published artifact was simply stale or badly exported.

The issue was deeper:

- naive dynamic quantization alone was not good enough for this model

### 3.3. Single-example tuning did not generalize reliably

The Angela/Macron pair was extremely useful as a detector, but it was a bad target to optimize against in isolation. The research repeatedly found that:

- a candidate could repair that pair
- and still regress badly on MNLI/XNLI

The clearest example was the late-layer exclusion candidate that fixed the anecdotal pair but dropped sharply on `mnli-validation_matched`.

### 3.4. FFN quantization was the wrong direction

The family search made this conclusion hard to avoid:

- `ffn_only` collapsed
- progressively larger FFN-float policies helped less than the cleaner attention-side families
- the best candidates were exactly the ones that refused to quantize FFN dense matmuls broadly

In practical terms, the project spent effort proving that this model is far less tolerant of FFN quantization than of attention-side quantization.

### 3.5. Broad late-layer and blanket attention rules did not solve the problem

These policies were all tried in one form or another:

- leave whole upper layers float
- exclude all Q/K/V projections
- exclude whole late blocks
- add extra upper-layer float rules on top of the finalists

The evidence was consistent:

- broad policies were either clearly worse or too blunt to beat the simpler finalists
- more exclusions often increased agreement slightly while reducing benchmark accuracy

The repo did not find a useful "just leave the top layers float" rule that beat `attention_only` or `attention_proj_only`.

### 3.6. Preprocessing and the first static path did not deliver

Both were worth trying, but neither became competitive:

- preprocessing before dynamic quantization did not materially repair fidelity
- the initial static QDQ smoke path did not produce a better candidate than the strong dynamic finalists

That is why the later plans demoted static quantization from a default next step to a conditional follow-up only if runtime gains had turned out to be stronger.

### 3.7. Quantization did not deliver decisive operational wins

This is the biggest late-stage negative result.

The quantized finalists did not produce the kind of system-level improvement that would automatically justify lower fidelity:

- size savings were modest: about `28 MB` for `attention_only` and about `8 MB` for `attention_proj_only` relative to float
- float loaded faster than both quantized finalists in cold-start CPU and CoreML paths
- CPU memory was effectively unchanged across all three finalists
- CoreML memory was much worse for the quantized finalists than for float

So the final answer was not "quantization failed completely." It was:

- quantization improved some benchmark outcomes and some warm-serving latencies
- but it did not improve enough of the operational picture to replace float as the default

## 4. Which Data Sets And Inputs Produced Particularly Interesting Results

The interesting results came from both broad balanced datasets and deliberately biased stress sets.

### 4.1. The non-overlapping MNLI/XNLI full suite was the most important balanced benchmark

This `1950`-example suite became the final ranking surface because it was:

- multilingual
- balanced enough to be useful
- stable across later reports
- large enough that tiny wins and losses could be interpreted seriously

It produced the headline frontier:

- float: `86.36%`
- `attention_only`: `86.67%`
- `attention_proj_only`: `86.51%`

It also made the multilingual tradeoffs visible instead of hiding them behind aggregate scores.

### 4.2. `xnli-zh` was the recurring guardrail and the clearest unresolved weakness

Chinese appeared in almost every later conclusion because it consistently stressed the quantized finalists more than the float model.

The final pattern was:

- full suite `xnli-zh`:
  - float: `86.00%`
  - `attention_only`: `85.33%`
  - `attention_proj_only`: `85.33%`
- hard probe `xnli-zh`:
  - float: `50.00%`
  - both quantized finalists: `37.50%`
- core probe `xnli-zh`:
  - float: `100.00%`
  - both quantized finalists: `0.00%`

That made `xnli-zh` the best single language-level guardrail against overclaiming quantization quality.

### 4.3. The hard probe and core probe were the most informative stress datasets

The full suite answered "who wins overall." The probe sets answered "where do the finalists still behave badly."

The hard probe was interesting because it was enriched for:

- HF disagreements
- high-drift examples
- cases where the finalists predicted different labels

The core probe was interesting because it kept that stress signal in only `25` rows while preserving:

- source coverage
- language coverage
- finalist disagreements
- a Chinese tripwire

These datasets were especially valuable because they prevented future work from hiding behind easy cases.

### 4.4. The Angela/Macron pair was the best early failure detector

The original bilingual probe pair remained important throughout the research even after the project moved beyond anecdotal checks.

Why it mattered:

- it exposed the published quantized model immediately
- it showed that float ONNX was faithful
- it showed that the first exclusion-based candidate really could repair a broken behavior mode
- later, it distinguished `attention_only` from `attention_proj_only` on fidelity even when both were broadly competitive on the main suite

It was not enough to rank final candidates, but it was a very good canary.

### 4.5. Specific recurring high-signal examples

Several named rows kept appearing as either high-drift examples, finalist disagreement cases, or language-specific tripwires.

#### `facebook-xnli-es-test-000003`

This Spanish example was especially useful because it showed that label stability and logit stability are not the same thing:

- HF label: `neutral`
- both finalists preserved the label
- `attention_only` still produced the highest recorded finalist drift in its probe set: `2.068416`

This is a good example of a case that looks safe by label agreement alone but is still behaviorally unstable.

#### `facebook-xnli-fr-test-000051`

This French example became the clearest `attention_proj_only` drift hotspot:

- HF label: `entailment`
- the label stayed the same
- `attention_proj_only` reached its highest reported full-suite drift here: `2.714536`

That made it a strong demonstration that the fidelity-oriented finalist was still only relatively better, not genuinely faithful.

#### `facebook-xnli-zh-test-000132`

This Chinese example became the single most useful quantization tripwire in the later probes:

- HF label: `neutral`
- float label: `neutral`
- both quantized finalists predicted `contradiction`

It was important because it compressed several conclusions into one row:

- Chinese remained sensitive
- both finalists still had shared blind spots
- the core probe correctly preserved it as a required guardrail

#### `nyu-mll-multi_nli-default-validation_mismatched-000672`

This MNLI mismatched example was one of the best finalists-separating rows:

- HF and float label: `contradiction`
- `attention_only`: `contradiction`
- `attention_proj_only`: `neutral`
- `attention_proj_only` drift: `2.052989`

This is a good example of why `attention_proj_only` remained only the fidelity baseline and not the universal recommendation. It was slightly better on average fidelity, but it still had sharp local failures.

### 4.6. Repeated multilingual "same structure, different language" cases were revealing

The probe sets also surfaced a family of translated XNLI examples across German, Spanish, French, and Chinese that behaved similarly across many quantizations:

- "take the totals and try to solve it" style examples
- "radiation can be contained during a fire" style examples
- paraphrases about whether someone needed help or acted independently

These were useful because they showed that the problem was not just one unlucky row in one language. Some semantic patterns were consistently fragile under quantization, especially once translated.

## 5. Rationale For The Final Quantization Choices

The final selection was based on roles, not on the idea that one artifact dominated every dimension.

### 5.1. Why float stayed the default

Float remained the most defensible default for four reasons.

First, it was the faithful reference:

- `100.00%` HF agreement on the full suite
- `100.00%` HF agreement on both the hard probe and core probe

Second, it remained the best language-level guardrail on the most fragile slice:

- strongest `xnli-zh` result on the full suite
- no collapse on the Chinese core-probe tripwire

Third, it was operationally simpler:

- fastest cold load on CPU
- fastest cold load on CoreML

Fourth, the final memory data strengthened the default-float case:

- CPU steady RSS was effectively the same across all three candidates
- CoreML steady RSS was far lower for float than for either quantized finalist

That combination made float the right repo default even though it did not have the best full-suite accuracy.

### 5.2. Why `attention_only` became the experimental quantized choice

`attention_only` was the right experimental quantized choice because it won on the metric that mattered most for an opt-in quantized path:

- best full-suite gold-label accuracy among quantized models: `86.67%`
- best net gain over float: `+6` correct examples
- best hard-probe gold-label accuracy
- best core-probe gold-label accuracy

It also had a real if modest serving advantage:

- best CPU persistent warm median among the finalists

The key strategic insight behind it was also clean and explainable:

- keep FFN dense matmuls in float
- quantize attention-side weight projections

That made it a credible experimental path rather than just a lucky benchmark artifact.

### 5.3. Why `attention_proj_only` stayed the fidelity baseline instead of the main experimental pick

`attention_proj_only` earned a stable place in the final recommendation, but not the top quantized slot.

Its case was:

- slightly better full-suite HF agreement than `attention_only`
- slightly lower mean drift
- much better behavior on the original Angela/Macron fidelity probe

Its limitation was equally clear:

- the accuracy edge belonged to `attention_only`
- the fidelity edge was tiny in aggregate
- it did not come with a decisive size, runtime, or memory advantage

So it remained valuable as the best quantized research baseline for fidelity-sensitive comparisons, but not as the repo's preferred quantized default.

### 5.4. Final answer to the selection question

The final documented choice was not:

- "ship quantization everywhere"

It was:

- use float by default because it is faithful, operationally simpler, and safer on memory and Chinese behavior
- keep `attention_only` as the main experimental quantized option because it is the best quantized accuracy tradeoff
- keep `attention_proj_only` as the secondary quantized baseline because it is the closest quantized model to HF, even if only slightly

## Bottom Line

The quantization research did exactly what a good research platform should do. It replaced vague intuition with a stable benchmark ladder, identified the real source of quantization damage, produced two defensible quantized finalists, and then showed why neither finalist was strong enough to displace float as the repo default.

What the repo learned is clear:

- the published quantized model was not good enough
- naive quantization was not good enough
- selective attention-side quantization could work
- FFN quantization was the main trap
- `attention_only` was the best accuracy-oriented quantized model
- `attention_proj_only` was the best fidelity-oriented quantized model
- float remained the correct default once quality, runtime, and memory were considered together
