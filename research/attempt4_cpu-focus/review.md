<reviewer>
Verdict: the repo supports a narrow claim, not the broader one its conclusion language suggests. What it actually establishes is this: among the specific candidates tested in attempt2, under a deliberately narrowed objective of on-disk size versus agreement with a backend-specific float ONNX reference, `nncf_accuracy_attention_only` remained the best CPU point after the locked `plan1` untouched-test readout at 465.1 MiB and 98.741% float-label agreement. It does **not** establish that this is the best quantized deployment artifact in a general sense, because the study explicitly removed task accuracy, latency, and memory from the final objective. ([GitHub][1])

The biggest methodological problem is internal inconsistency. The repo’s own overfitting note says teacher-fidelity should be only a debugging or pruning aid and that the final winner should be chosen on validation task metrics, then reported once on untouched test. Attempt2 does the opposite: it explicitly changes the main objective to fidelity against the backend-specific float reference and says labels are not part of the main selection objective. That is not just a framing difference; it changes what “winner” means. ([GitHub][2])

The second major issue is objective misalignment inside the search itself. The final attempt objective is float-reference fidelity, but the incumbent winner was created with NNCF `accuracy-control` using `--metric=gold_accuracy`, not `hf_agreement`; only the later `plan1` follow-up searched more directly around the fidelity objective. That means the final winner benefited from an earlier search conducted under a different metric than the one used for the final decision. That is not invalid, but it is asymmetrical and weakens claims that the final frontier is the product of one consistent optimization criterion. ([GitHub][3])

The third major issue is overstatement in the conclusion language. The repo repeatedly uses “CPU recommendation” and “winner,” but the study design explicitly says it is ignoring runtime benchmarking, RSS, and peak memory, and treats on-disk file size as the only size metric. OpenVINO’s own accuracy-control docs also note that these methods keep impactful operations in higher precision and therefore can yield smaller performance gains than basic 8-bit quantization. A safer conclusion would be: “best tested CPU artifact for preserving CPU-float behavior under this size-versus-fidelity objective,” not “final CPU recommendation” without qualification. ([GitHub][1])

The fourth major problem is reproducibility of the final evidence package. `result1.md` links its primary JSON evidence files, but those links resolve to absolute local paths under `/Users/mjm/...` and return GitHub 404s. That means the report’s final supporting artifacts are not actually inspectable from the public repo page that cites them. For a study centered on reproducible cataloged provenance, this is a real defect. ([GitHub][4])

A fifth issue is weak external validity for a multilingual deployment claim. The model card says the model supports NLI across 100 languages and was fine-tuned on XNLI’s 15 languages plus English MNLI. Attempt2 calibrates and validates on MNLI plus only five XNLI languages—de, en, es, fr, zh—and its final test set also uses only those five XNLI languages. That is adequate for a scoped probe, but under-validates the model for its advertised multilingual envelope. ([Hugging Face][5])

A sixth issue is that the published metrics are too narrow for zero-shot/NLI deployment. The summarizer computes float-label agreement, mean absolute logit delta, max absolute logit delta, and disagreement count by comparing each candidate to reference predictions and logits. There is no report of gold-label accuracy/F1 for the final untouched test, no per-language breakdowns, no confidence calibration metrics, and no probability-space divergence metrics. That leaves a blind spot where argmax decisions can be preserved while confidence structure shifts in ways that matter for ranking, thresholding, or zero-shot classification. This is an inference from the repo’s metric definitions and published outputs, but it is a material one. ([GitHub][6])

A seventh issue is that several reported differences are small enough that uncertainty should have been quantified. The final `plan1` test difference between `nncf_accuracy_attention_only` and `nncf_fidelity_attention_only_n128_drop0p005` is 98.741% versus 98.519%; the validation difference between `nncf_accuracy_attention_only` and `nncf_fidelity_attention_proj_only` is 99.421% versus 99.653%. The repo gives no confidence intervals, no paired significance test, and no bootstrap analysis. Because the evaluation pack is finite, some frontier movements could be only a few examples. That does not negate the result, but it does make the presentation too categorical. ([GitHub][7])

An eighth issue is that the ORT transformer baselines are under-explored relative to ONNX Runtime guidance. ORT recommends dynamic quantization for transformers in general, says CPU S8S8 QDQ is the default first choice, suggests trying `U8U8` or `reduce_range` when x86 saturation causes large accuracy drops, and points to transformer-specific optimizations such as QAttention via the Transformer Model Optimization Tool before quantization. Attempt2’s carry-forward ORT-style baselines are mainly `dynamic_qint8_default` and `dynamic_qint8_per_channel`, and they perform very poorly, but the repo does not show a comparable attempt using those transformer-specific or datatype-specific alternatives. So the data support “our tried dynamic baselines were bad,” not “dynamic ORT quantization is bad for this model family.” ([GitHub][8])

A ninth issue is search-space truncation by machine budget. `plan1` states that one candidate was retired as too slow for that machine and the `n300` pair was never started after a machine-based pruning decision. `result1.md` repeats that the `n300` pair was not started. That makes the final frontier conditional on local compute patience and hardware practicality, not an exhaustive evaluation of the declared candidate family. This is acceptable if stated plainly; it is not acceptable if the wording implies a global winner. ([GitHub][9])

A tenth issue is incomplete run transparency around retry behavior. The NNCF wrapper can catch certain failures, retry quantization with SmoothQuant disabled, and record `smooth_quant_disabled` plus `retry_reason` in its JSON report. That is a useful recovery mechanism, but the result documents do not surface whether any published candidate was produced through this retry path. Given that such a retry changes the effective method, this should be disclosed in the final comparison table for every artifact. ([GitHub][10])

There are also several weaker nitpicks.

The frontier code groups rows by `(dataset, backend)` when computing Pareto flags, not by `(role, dataset, backend)`. In the published workflow this probably does not bite because reports are usually generated per role, but it is a latent footgun: if someone summarizes multiple roles together, frontier status could mix incomparable buckets. ([GitHub][6])

The repo is better on split hygiene than many ad hoc quantization studies. It explicitly uses calibration, fidelity-validation, fidelity-test, and smoke roles; only locked frontier points go to untouched test; and the plan1 dataset-prep script checks generated files against guarded legacy filenames to avoid overlap. This part is methodologically sound. ([GitHub][11])

The use of preprocessing is not a flaw. The wrapper calls `quant_pre_process()` and, unless `--skip-preprocess-optimization` is set, allows graph optimization during preprocessing. ONNX Runtime docs explicitly say optimization is better done during preprocessing than during quantization. So this part is aligned with ORT guidance, though the preprocessed intermediate should ideally be versioned or hashed for exact reproducibility. ([GitHub][10])

The use of `subset_size=300` is also not obviously wrong. NNCF exposes 300 as the default for `quantize_with_accuracy_control`, and OpenVINO’s basic PTQ docs use “for example, 300 samples” for calibration. The real issue is not the number by itself; it is whether the sample is representative of the intended workload. For a model advertised across 100 languages, five-language MNLI/XNLI slices are a limited proxy. ([GitHub][10])

The strongest rewrite of the conclusion would be this:

“Within the attempt2 search space and its explicit objective of on-disk size versus agreement with the backend-specific float ONNX reference, `nncf_accuracy_attention_only` is the best tested CPU artifact. This does not establish it as the best overall deployment model, because final selection excluded task accuracy, latency, resident memory, broader multilingual coverage, and several ORT transformer-specific baselines.” ([GitHub][1])

If I were tightening the study, I would make four concrete changes. First, keep the current fidelity analysis, but make final winner selection multi-objective again: gold accuracy or F1 on untouched task test, plus latency, plus memory, plus file size. Second, add per-language and per-dataset breakdowns instead of only one aggregate score. Third, add stronger ORT baselines: transformer optimizer/QAttention, `U8U8`, and `reduce_range` where relevant. Fourth, publish the actual JSON artifacts linked from `result1.md` and surface whether SmoothQuant retry was triggered for each candidate. Those changes would not require abandoning the current work; they would turn it from a careful fidelity study into a stronger deployment study. ([GitHub][2])

[1]: https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/plan0.md "https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/plan0.md"
[2]: https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/0init_overfitting.md "https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/0init_overfitting.md"
[3]: https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/conclusion.md "https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/conclusion.md"
[4]: https://github.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/blob/main/research/attempt2_course-correction/result1.md "https://github.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/blob/main/research/attempt2_course-correction/result1.md"
[5]: https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli "https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
[6]: https://github.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/blob/main/tools/summarize-study-db.py "https://github.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/blob/main/tools/summarize-study-db.py"
[7]: https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/result1.md "https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/result1.md"
[8]: https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/result0.md "https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/result0.md"
[9]: https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/plan1.md "https://raw.githubusercontent.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/main/research/attempt2_course-correction/plan1.md"
[10]: https://github.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/blob/main/tools/quantize-onnx-nncf.py "https://github.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/blob/main/tools/quantize-onnx-nncf.py"
[11]: https://github.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/blob/main/research/attempt2_course-correction/result0.md "https://github.com/refinement-systems/research_mDeBERTa-v3-base-mnli-xnli/blob/main/research/attempt2_course-correction/result0.md"
</reviewer>
<response>
Thanks! How to ensure that grading the results on accuracy won't cause overfitting? What datasets should be used for the broadest test of this model? The model being quantized is MoritzLaurer/mDeBERTa-v3-base-mnli-xnli which is a finetune of microsoft/mdeberta-v3-base I don't want to lose the finetuning either
</response>
<reviewer>
To keep accuracy-based quantization selection from overfitting, separate the process into four roles and never blur them.

1. Calibration set. This is only for choosing activation ranges or other PTQ statistics. It should be representative of the inference workload, but it does not need labels. OpenVINO PTQ explicitly treats calibration as separate from evaluation, and its basic PTQ flow does not require retraining or fine-tuning. ([docs.openvino.ai][1])

2. Development set. This is the only labeled set allowed to influence candidate choice. If you use OpenVINO/NNCF “accuracy control,” this is the validation dataset in their terminology. That flow repeatedly evaluates accuracy during quantization, so anything used there is, in effect, part of model selection and can be overfit. ([docs.openvino.ai][2])

3. Locked test set. This is opened once, after the candidate family and selection rule are frozen.

4. Locked stress suite. This is separate from the main test and is used only for robustness reporting, not candidate choice.

For this specific model, the most important leak to avoid is XNLI validation. The model card states that `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` was trained on the XNLI development set and the MNLI train set, and explicitly avoided the machine-translated XNLI train set. Therefore XNLI validation is not a valid evaluation set for quantization selection here, because it is part of the original supervised fine-tuning data. ([Hugging Face][3])

A safe protocol for this model is:

* Use PTQ, not QAT, if the goal is to preserve the existing fine-tuning as-is. OpenVINO PTQ does not require retraining or fine-tuning, while QAT is explicitly a training-time fine-tuning method used to recover lost accuracy. ONNX Runtime likewise describes QAT as retraining if PTQ is insufficient. ([docs.openvino.ai][1])
* Use a disjoint, unlabeled calibration pool.
* Choose candidates only on a fixed dev aggregate.
* Open the locked test once.
* Report robustness sets separately.
* Record every candidate tried. Do not silently keep searching until one wins.

For the broadest faithful test of this model, use these benchmark buckets.

The core English anchor should be MultiNLI validation matched and validation mismatched. MultiNLI is the English NLI dataset the model was fine-tuned on, but the validation splits are held out from training, and they cover both matched and cross-genre evaluation. The dataset card lists `validation_matched` and `validation_mismatched`, each around 9.8k examples. ([Hugging Face][3])

The core multilingual anchor should be XNLI test in all 15 languages. The model card reports evaluation on exactly that benchmark, and the XNLI dataset card shows 5,010 test examples per language across the 15-language set. This is the broadest official multilingual held-out test tied directly to this model. ([Hugging Face][3])

For hard English robustness, add ANLI and HANS. ANLI is adversarial NLI and is explicitly harder than MNLI, with three rounds and train/dev/test splits. HANS is a targeted evaluation set for invalid NLI heuristics and is formatted similarly to MNLI. These are useful because a quantized model can keep average accuracy while degrading disproportionately on brittle reasoning cases. ([Hugging Face][4])

For additional out-of-domain English robustness, add WANLI test. WANLI contains 108k English NLI pairs overall and has a 5k test split built from worker-AI collaboration around difficult reasoning patterns derived from MultiNLI neighborhoods. ([Hugging Face][5])

That yields this practical split plan:

Calibration:

* Unlabeled representative inputs only.
* Prefer a mixture close to your actual deployment traffic.
* Do not use any locked test examples here.

Development / candidate selection:

* MultiNLI `validation_matched`
* MultiNLI `validation_mismatched`
* ANLI `dev` rounds 1–3

Locked final test:

* XNLI `test` all 15 languages
* ANLI `test` rounds 1–3

Locked stress / robustness:

* HANS evaluation set
* WANLI `test`

This setup prevents the main mistake: using XNLI validation for selection even though the model was trained on it. It also keeps the broadest multilingual official benchmark fully untouched until the end. ([Hugging Face][3])

If you want the single broadest score that still respects the model’s original fine-tuning, define it before running the search. A good choice is macro-average of:

* MultiNLI matched accuracy
* MultiNLI mismatched accuracy
* XNLI 15-language macro-average accuracy

Then require non-inferiority constraints on the stress sets, such as “no more than X points drop on any ANLI round, HANS, or WANLI.” The important part is not the exact formula. The important part is freezing it before candidate search.

To reduce selection overfitting further, add three guardrails.

First, keep the candidate family small and predeclared. Overfitting can happen even without weight updates if you try enough quantization variants and pick the luckiest one on the dev set.

Second, use paired testing on the final comparison, not just raw averages. Quantized-versus-float differences on the same examples should be checked with a paired method such as bootstrap confidence intervals or McNemar-style analysis on correctness flips. That tells you whether a tiny apparent gain is noise.

Third, monitor more than top-1 accuracy. Since your model is used for NLI and zero-shot classification, logit and probability drift matter. A candidate that preserves argmax on MNLI may still distort confidence margins and behave worse when used as a zero-shot classifier with templated hypotheses.

On “not losing the fine-tuning,” the strict answer is this:

* PTQ preserves the original learned task weights and only changes inference representation. That is the safest route if you want the finetune preserved. ([docs.openvino.ai][1])
* QAT can preserve task performance better than PTQ, but it does so by further training the model, so the weights are no longer exactly the original fine-tuned checkpoint. ([docs.openvino.ai][6])
* If you ever move to QAT, start from the already fine-tuned checkpoint and use the same task family the model was fine-tuned on, not unrelated data. For this model, that means staying in the MNLI/XNLI NLI regime. The model card explicitly says the author avoided the machine-translated XNLI train set to avoid quality issues, overfitting to the 15 XNLI languages, and catastrophic forgetting of the other 85 pretraining languages. ([Hugging Face][3])

The shortest recommendation is:

Use PTQ. Select only on MultiNLI validation plus ANLI dev. Keep XNLI test fully locked until the end. Report XNLI test all 15 languages as the main multilingual result, and HANS plus WANLI as stress tests. Never use XNLI validation for evaluation, because for this exact model it is training data. ([Hugging Face][3])

[1]: https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html "Post-training Quantization — OpenVINO™  documentation"
[2]: https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/quantizing-with-accuracy-control.html "Quantizing with Accuracy Control — OpenVINO™  documentation"
[3]: https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli · Hugging Face"
[4]: https://huggingface.co/datasets/facebook/anli "facebook/anli · Datasets at Hugging Face"
[5]: https://huggingface.co/datasets/alisawuffles/WANLI "alisawuffles/WANLI · Datasets at Hugging Face"
[6]: https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/compressing-models-during-training/quantization-aware-training.html?utm_source=chatgpt.com "Quantization-aware Training (QAT)"
</response>
