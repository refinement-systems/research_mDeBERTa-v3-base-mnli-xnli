The main missing category is not “another random quantizer.” It is more systematic accuracy-controlled mixed-precision tooling.

The strongest missed approach is accuracy-controlled PTQ in NNCF/OpenVINO. OpenVINO documents a transformer-specific quantization scheme, a `MIXED` preset for asymmetric activations such as GELU, bias-correction controls, larger calibration subsets, explicit ignored scopes, and an “accuracy control” flow that automatically keeps the most impactful operations in original precision based on a validation metric. That is very close in spirit to your manual family search, but it is a productized search procedure rather than a hand-built exclusion sweep. For an ONNX-derived encoder model, this is the first approach I would add. ([docs.openvino.ai][4])

The second missed approach is QAT. OpenVINO’s documentation states directly that QAT is the most accurate quantization method and is the fallback when PTQ cannot meet the accuracy target. Your report is almost entirely a PTQ study. Given how close your two finalists already are, a short fine-tuning pass starting from `attention_only` or `attention_proj_only` is the most credible way to try to recover the remaining fidelity gap without going back to full float. ([ONNX Runtime][1])

The third missed axis is ONNX Runtime datatype and format tuning. ONNX Runtime says that on CPU the first choice should be S8S8 with QDQ, that S8S8 with QOperator is generally slow on x86-64, and that U8U8 is worth trying when there is a large accuracy drop because U8S8 can suffer saturation on AVX2/AVX512 paths. If your search concentrated mainly on dynamic `MatMulInteger`-style candidates, this format/signedness axis looks underexplored. ([ONNX Runtime][1])

The fourth missed family is outlier-aware quantization outside the ONNX/CoreML path. SmoothQuant is calibration-based and explicitly targets activation outliers by migrating difficulty from activations to weights. LLM.int8() isolates outlier features into higher precision, and QLoRA introduces NF4 plus double quantization. This is relevant because your own findings point to localized sensitivity and outlier-like behavior rather than uniform fragility. That said, this is only useful if you are willing to use a different runtime stack. It is not a drop-in replacement for your ONNX/CoreML deployment path. ([arXiv][5])

A related caveat: one adjacent mDeBERTa-based paper, xCOMET-lite, reports that GPTQ was incompatible with the mDeBERTa architecture in their setting, while LLM.int8() and QLoRA were viable. So “try GPTQ” is not an obviously missed move here; “try LLM.int8/QLoRA in a different runtime” is more plausible. ([arXiv][6])

[4]: https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html "Basic Quantization Flow — OpenVINO™  documentation"
[5]: https://arxiv.org/pdf/2211.10438 "SmoothQuant: Accurate and Efficient  Post-Training Quantization for Large Language Models"
[6]: https://arxiv.org/html/2406.14553v2 "xCOMET-lite: Bridging the Gap Between Efficiency and Quality in Learned MT Evaluation Metrics"

