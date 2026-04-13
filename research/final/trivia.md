# Trivia And Odd Findings

This note collects the memorable edge cases and operational oddities that kept resurfacing across the research history. It is not the main recommendation document; it is the "things that were weird enough to matter later" document.

## 1. The Original Hugging Face Docs Pair

The first strong warning sign came from the sentence pair used in the Hugging Face docs and in the repo's early debugging tools:

- premise: `Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU`
- hypothesis: `Emmanuel Macron is the President of France`

On this pair:

- Hugging Face / PyTorch logits were `[-3.285156, -0.886230, 3.873047]`
- the float ONNX export stayed very close at `[-3.287450, -0.884211, 3.871160]`
- the official `model_quantized.onnx` moved to `[-1.611540, 1.460460, 0.140061]`

So:

- the float export preserved the original model correctly
- the published official quantized artifact flipped into the wrong `neutral` regime

This one pair drove a lot of the early investigation because it showed immediately that the official quantized artifact was not just "slightly noisy"; it could change the decision boundary in a visibly bad way.

One early local repair, `dynamic_qint8_matmul_exclude_suggested.onnx`, restored the correct `contradiction` label on this pair and reduced max absolute logit drift versus HF to `0.456847`, but later broader evaluation showed that fixing this one case was not enough to guarantee a good quantized model overall.

## 2. The Chinese Row That Refused To Behave

The most famous Chinese stress example was:

- row id: `facebook-xnli-zh-validation-000425`
- gold label: `entailment`
- premise: `哦，好的，有意思，你上课，呃，你学过怎么做吗？`
- hypothesis: `你从哪学到怎么做的？`

What made it unusual:

- `hf_label = neutral`
- `float_label = neutral`
- `attention_only_label = entailment`
- `attention_proj_only_label = neutral`
- `nncf_fidelity_attention_proj_only_label = neutral`

So this was one of the rare rows where a quantized candidate (`attention_only`) matched the gold label while both Hugging Face and the float ONNX reference stayed on the same wrong label.

That is why this row became a key piece of the attempt1 Chinese stress analysis. It was not a general argument that quantization improved the model; it was a reminder that "closer to float" and "closer to gold" are not always the same thing on individual examples.

## 3. The Bigger Chinese Slice Was Less Interesting Than The Tiny Stress Pack

One surprising result from attempt1 was that a larger disjoint Chinese-heavy validation slice turned out to be almost useless for separating the finalists.

On the `384`-row extra zh slice:

- `float`: `99.48%` accuracy, `100.00%` HF agreement
- `attention_proj_only`: `99.48%`, `100.00%`
- `nncf_fidelity_attention_proj_only`: `99.48%`, `100.00%`
- `nncf_fidelity_attention_proj_only_n128_drop0p005`: `99.48%`, `100.00%`
- `nncf_fidelity_attention_proj_only_n128_drop0p002`: `99.48%`, `100.00%`

The smaller `48`-row zh stress pack was more informative than the much larger held-out slice. That is a good example of why the repo eventually moved toward carefully curated probe and stress packs instead of assuming that "more rows" automatically meant "more signal."

## 4. Certain Story Families Kept Reappearing

The frozen probe packs ended up containing a few recurring multilingual "story families" because they repeatedly produced disagreements or unusually large drift:

- the U2 aircraft / pilot training example
- the Ramona / phone call examples
- the aircraft-fire / radiation example
- short Chinese rows where tiny wording changes changed the label boundary sharply

Examples from the frozen probe packs:

- `facebook-xnli-en-test-000128`
  - premise: `It's 30 or 40 U2 aircraft...`
  - hypothesis: `We trained with a lot of other soldiers.`
  - `hf_label` and `float_label` were `contradiction`, while `attention_proj_only` moved to `neutral`
- `facebook-xnli-es-test-000088`
  - premise: `Así que yo pensaba, Dios mío, y Ramona estaba ahí.`
  - hypothesis: `Ramona me estaba juzgando en silencio.`
  - `hf_label` and `float_label` were `contradiction`, while `attention_proj_only` moved to `neutral`
- `facebook-xnli-zh-test-000132`
  - premise: `他说他们已经北上了。`
  - hypothesis: `他说在路上停了几站。`
  - `hf_label` and `float_label` were `neutral`, while both `attention_only` and `attention_proj_only` moved to `contradiction`

These rows mattered because they kept showing which candidate families were stable and which families were prone to semantic boundary drift.

## 5. The Research Had A Real Evaluation-Leak Trap

One of the most important late corrections was methodological, not numerical.

The repo review for attempt4 called out that this exact model was fine-tuned using the XNLI development set. That means XNLI validation is not a clean untouched evaluation split for quantization selection on this model. Earlier work had naturally drifted toward XNLI-heavy checking because the model is multilingual, but the later CPU study corrected that by moving selection to MNLI development plus ANLI development and keeping XNLI for locked final testing.

This was one of the most important "unusual findings" because it changed the evaluation protocol, not just the ranking of candidates.

## 6. Two Benchmark Packs Looked Separate But Weren't

Another easy-to-miss issue was overlap between some generated MNLI benchmark packs.

The early `100-per-label` MNLI slices overlap with the later `200-per-label` MNLI slices. So they should not be counted together in a single aggregate as if they were independent evidence. This is documented in the early attempt0 write-up and later carried forward into the final dataset documentation.

It is a small detail, but it matters because sloppy overlap handling would have made later accuracy summaries look more robust than they really were.

## 7. Percentile Static Quantization Was Not Merely Bad, It Was Broken

Attempt1 answered the "percentile calibration" question more harshly than expected:

- all `12` percentile static candidates failed during static quantization
- the branch was treated as operationally broken for this model / ORT path / machine combination

That mattered because it was not a quality tradeoff. It was a search-space pruning event.

## 8. A Static Sweep Managed To Hit 50+ GB On A 16 GB Machine

The broad static sweep produced one of the more absurd operational findings in the repo:

- a Python process was observed peaking above `50 GB`
- the machine had `16 GB` of RAM

Even allowing for allocator and swap effects, the conclusion was straightforward: broad static sweeps were an unhealthy fit for this machine and not justified by the quality upside being observed.

## 9. The "Accurate" NNCF Branch Failed In A Very Specific Way

One NNCF line looked promising enough to revisit after a SmoothQuant retry, but then died with oddly concrete graph-export problems:

- duplicate `nncf_smooth_quant_output` definitions
- `KeyError` on `/deberta/embeddings/word_embeddings/Gather`

Those errors are memorable because they were not vague training instability. They were hard graph / export failures after the run had already looked worth salvaging.

## 10. The QAT Pilot Mostly Vanished

The repo did try a gated QAT rescue pilot. It was launched, partially debugged, and ran long enough to feel real, but it never produced a benchmarkable artifact. Worse, after the wrapper session ended, there was no final useful traceback left behind.

So the QAT lane ended up as an unusual negative result:

- it was not disproven on quality grounds
- it simply never delivered a usable artifact under the actual machine and workflow constraints

## 11. The Final CPU Winner Was Smaller, Leaner, And Slower

The final CPU study produced a result that is easy to summarize incorrectly.

`nncf_accuracy_attention_only` became the default quantized CPU recommendation because it won the locked CPU frontier on:

- on-disk size
- steady RSS
- cold load
- near-float locked-final accuracy

But it did **not** beat float on persistent warm latency:

- CPU float warm median: `483.620 ms`
- `nncf_accuracy_attention_only` warm median: `743.025 ms`

So the CPU story ended with a real deployment tradeoff, not a universal win. Quantization helped the storage and memory budget, but the float model remained the warm-latency winner.

## 12. CoreML Had Good Fidelity But Strange Operational Behavior

The short CoreML study produced another memorable mismatch between quality and usefulness.

`reference_fp16` under CoreML preserved quality almost perfectly:

- dev accuracy: `75.74%` versus `75.75%` for CoreML float
- float-label agreement: `99.99%`

But the runtime behavior was poor:

- CoreML float warm median: `503.142 ms`
- CoreML fp16 warm median: `1862.160 ms`

And CoreML float itself was already awkward operationally on this machine:

- load median: `19280.200 ms`
- steady RSS: `4175.0 MiB`
- peak RSS: `6322.5 MiB`

So the CoreML conclusion was not "the backend is inaccurate." It was "the backend behaves badly enough on this machine that the quality preservation does not rescue it."
