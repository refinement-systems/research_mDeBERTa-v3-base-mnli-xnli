In practice, the safe protocol is to give each dataset split exactly one job, and never let the test set participate in any quantization decision.

The jobs are these. The calibration split is used to estimate quantization parameters such as activation ranges and zero-points for static quantization. The validation split is used to choose among quantization candidates. The test split is touched once, at the very end, only to report final performance. This separation matters because calibration data for PTQ must be representative of real inputs, and because any finite-sample criterion used to choose among candidates is susceptible to selection bias if you optimize against it repeatedly. ([PyTorch Documentation][1])

A practical workflow looks like this.

First, freeze the reference model and the evaluation protocol. “Freeze” here means: fix the float model checkpoint, the preprocessing, the tokenizer, the inference settings, the hardware/backend you care about, and the task metrics you will report. If any of those move during quantization search, you are no longer measuring only the effect of quantization. This is especially important in ONNX Runtime and TensorRT style flows, where preprocessing, graph optimization, and backend behavior can materially affect what the final quantized graph looks like. ([ONNX Runtime][2])

Second, define four data roles, even if some are small. Role 1 is the original training or fine-tuning data, if any further training is involved. Role 2 is a calibration set for PTQ static quantization. Role 3 is a validation set for candidate selection. Role 4 is a final test set. The calibration set should look like the deployment distribution, because quantization ranges are derived from observed activations; vendor documentation explicitly warns that random or unrepresentative calibration data will produce inaccurate quantized models. ([PyTorch Documentation][1])

Third, keep the calibration set and validation set separate. This is the most common place people cut corners. The calibration set is for fitting the quantizer. The validation set is for deciding which quantizer to keep. If you reuse the same split for both jobs, you have already started to adapt the quantization recipe to that data. Sometimes this is acceptable in a loose engineering sense, but it weakens the cleanliness of the evaluation. The safer version is: calibrate on one representative slice, select on another, test on a third untouched slice. The general statistical reason is the same as in any model-selection pipeline: once a sample influences selection, its estimate becomes optimistic. ([Journal of Machine Learning Research][3])

Fourth, decide what the candidate space is before you look at validation results. For example, you might enumerate only these knobs: dynamic versus static quantization, per-tensor versus per-channel where supported, symmetric versus asymmetric formats where applicable, a small set of observer or calibration methods, and perhaps a short list of layer-exclusion patterns for numerically sensitive blocks. This should be a bounded, predeclared search space. The reason is simple: the more candidates you try, the easier it is to win the validation set by accident. That is ordinary model-selection overfitting. ([Journal of Machine Learning Research][3])

Fifth, run calibration only on the calibration split. For static quantization, you pass the calibration data through the prepared model so observers or backend-specific calibrators can collect activation statistics and derive quantization parameters. PyTorch describes PTQ static exactly this way, and ONNX Runtime and TensorRT require representative calibration data for the same reason. If you are using dynamic quantization, there is less offline fitting because activation scales are computed at inference time, but you still need a clean validation/test protocol for candidate selection. ([PyTorch Documentation][1])

Sixth, if you want to use variant (a), treat teacher-fidelity only as a development-time filter, not as the final acceptance rule. In practice that means this: on a development-only split, compare the quantized model against the float model using activation or weight comparisons, SQNR, normalized error, cosine similarity, or backend debugging tools that align tensors between float and quantized graphs. PyTorch exposes numeric comparison utilities, and ONNX Runtime explicitly documents quantization debugging workflows for matching weights and activations. This is useful to detect catastrophic breakage and to identify layers that should stay in higher precision. It is not a trustworthy proxy for task quality by itself, because it can reward faithful reproduction of the float model’s mistakes. ([PyTorch][4])

A safe use of teacher fidelity looks like this: you generate candidates, calibrate them, then discard the obviously bad ones using only float-vs-quantized diagnostics on a development subset. For example, if a candidate shows very large activation mismatch in a few layers, you drop it or exempt those layers from quantization. After that, you still choose the winner using real task metrics on the validation set, not by “best match to float.” ONNX Runtime’s debugging guidance is aligned with this: compare tensors to locate where the largest differences occur, then avoid quantizing those tensors or change the method. ([ONNX Runtime][2])

Seventh, select the final candidate on the validation set using the real task objective. For a classifier that means accuracy, F1, calibration error, latency, memory, or a predeclared composite acceptance rule. For a retrieval or generation system it may mean NDCG, MRR, exact-match, BLEU, acceptance rate, latency, and memory. The key point is that the validation set is the only place where you are allowed to say “candidate B is better than candidate A.” If the quantized model slightly beats the float model on validation, that is fine. It does not prove it is intrinsically better; it means that on the held-out validation sample, the perturbation introduced by quantization happened to improve the metric. That is allowed, as long as the decision is still finalized before the test set is opened. The unsafe version is to search repeatedly until something beats the float model on the test set. ([Journal of Machine Learning Research][3])

Eighth, lock the choice before touching the test set. “Lock” means no further changes to the quantization configuration, excluded layers, calibration method, backend flags, or acceptance thresholds. Then run the final float-versus-quantized comparison once on the test set and report it. At that point the test set has served its purpose. If you inspect the test failures and then go back to tweak the quantizer, that test set is no longer a test set. It has become another validation set, and you need a fresh untouched test set for any credible final claim. ([Journal of Machine Learning Research][3])

Ninth, document the exact provenance of the quantized model. In practice you want the following recorded alongside the artifact: base model checksum, quantization tool version, backend, calibration split definition, candidate grid, selection rule, validation results for all candidates, and the single final test result. This is not just bookkeeping. In quantization flows, especially ONNX-based ones, graph rewrites and preprocessing can materially change the result, so reproducibility depends on capturing the exact pipeline. ONNX Runtime also notes that model optimization can make float/quantized debugging harder, which is why they recommend doing model optimization during preprocessing rather than inside quantization itself. ([ONNX Runtime][2])

Tenth, keep a small post-test “shadow” or “deployment replay” check if the deployment distribution is volatile. This is not a replacement for the untouched test set. It is an operational safeguard. You take recent, real, unlabeled or lightly labeled production-like inputs and verify that activation ranges, latency, and obvious failure modes still look sane. This follows directly from the requirement that calibration data be representative of actual workload. If your deployment traffic shifts, recalibration on a more representative sample may be necessary. NVIDIA explicitly recommends recalibrating with a more representative calibration dataset when INT8 accuracy is poor. ([NVIDIA Docs][5])

For your two variants, the protocol becomes very concrete.

For variant (a), the safe version is:

1. Split data into calibration, validation, test.
2. Generate quantization candidates.
3. Calibrate each candidate only on calibration data.
4. Optionally use float-vs-quantized fidelity diagnostics on calibration or another dev-only split to prune clearly broken candidates.
5. Choose the winner only on validation task metrics.
6. Evaluate once on the untouched test set. ([PyTorch Documentation][1])

For variant (b), the safe version is even simpler:

1. Split data into calibration, validation, test.
2. Generate and calibrate candidates.
3. Compare all candidates only on validation metrics.
4. Pick one candidate.
5. Run one final test evaluation.
6. Do not return to tuning after seeing the test result. ([PyTorch Documentation][1])

If data are scarce, the strict version is nested resampling: an inner loop for quantization candidate selection and an outer loop for unbiased performance estimation. That is the same general remedy recommended for model-selection bias more broadly. ([Journal of Machine Learning Research][3])

A compact rule set captures the whole protocol:

Calibration data fits the quantizer.
Validation data chooses the candidate.
Test data reports the result.
Teacher fidelity is a debugging aid, not the final objective.
Once the test set influences a choice, replace it. ([PyTorch Documentation][1])

A concrete example for an NLP model would be this. You export one fixed FP32 ONNX model. You prepare a representative calibration set that matches your expected language mix, sequence-length distribution, and input formatting. You create, for example, eight candidates differing in static versus dynamic quantization, weight format, and a small set of layer-exclusion rules. You calibrate static candidates on the calibration split. You use tensor-level diagnostics only to eliminate candidates with clear numerical breakage. You then score the remaining candidates on validation accuracy and latency. You pick one candidate according to a predeclared rule such as “highest validation F1 among candidates whose latency is at least 20% better than FP32.” Only after that do you run the untouched test set once and write down the final result. That is the safe protocol in practice. ([ONNX Runtime][2])

The main invalidating mistakes are also simple.

Using the test set to decide which candidate wins is invalid. ([Journal of Machine Learning Research][3])

Using teacher-fidelity on the test set to decide which candidate wins is also invalid. It is still selection on the test set, just with a different objective. ([Journal of Machine Learning Research][3])

Recalibrating after looking at test failures invalidates the test set. ([Journal of Machine Learning Research][3])

Trying many more candidates than you planned and only reporting the winner increases selection bias unless that search was confined to validation data and the final evaluation remained untouched. ([Journal of Machine Learning Research][3])

Using unrepresentative or random calibration inputs makes the quantized model unreliable, even if the test protocol itself is otherwise clean. ([ONNX Runtime][6])

If you want the protocol expressed as a literal checklist or CI pipeline, it can be written in that form as well.

[1]: https://docs.pytorch.org/docs/2.6/_sources/quantization.rst.txt?utm_source=chatgpt.com "quantization.rst.txt"
[2]: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html?utm_source=chatgpt.com "Quantize ONNX models | onnxruntime"
[3]: https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf?utm_source=chatgpt.com "On Over-fitting in Model Selection and Subsequent ..."
[4]: https://pytorch.org/blog/quantization-in-practice/?utm_source=chatgpt.com "Practical Quantization in PyTorch"
[5]: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html?utm_source=chatgpt.com "Best Practices — NVIDIA TensorRT"
[6]: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html?utm_source=chatgpt.com "QNN Execution Provider - Qualcomm"
