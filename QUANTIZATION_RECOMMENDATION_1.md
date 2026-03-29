# Quantization Recommendation 1

## Recommendation

- Default model: `models/mdeberta/onnx/model.onnx`
- Optional experimental quantized model: `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_only.onnx`
- Secondary quantized fidelity baseline: `models/mdeberta/onnx/candidates/family_search/dynamic_qint8_matmul_attention_proj_only.onnx`

## Rationale

- Float remains the best default because it keeps 100.00% HF agreement on the full suite, has the strongest `xnli-zh` result (86.00%), and initializes fastest on both CPU and CoreML.
- `attention_only` is the recommended experimental quantized path because it has the best full-suite accuracy among quantized models (86.67%) and the best net gain over float (+6 correct examples on the full suite).
- `attention_proj_only` is still useful as the research fidelity baseline because its full-suite HF agreement is slightly higher (98.82%), but that edge is too small to justify preferring it over the accuracy-oriented quantized candidate for a repo-level experimental default.
- Persistent-session runtime does favor quantization, but only modestly. The recommended experimental path saves about 10.18 ms on CPU warm median and about 12.89 ms on CoreML warm median.

## Caveat

- If deployment is clearly persistent-session oriented and exact HF fidelity matters less than warm steady-state latency, a quantized model is defensible as an opt-in path. That still does not justify changing the repo default away from float.
