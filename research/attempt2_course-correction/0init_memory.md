Quantization reduces model precision and often reduces the serialized model size, but it does not guarantee a lower process RSS. Your own code is measuring current resident memory and a peak/high-water mark. On Linux, `ru_maxrss` is explicitly the maximum resident set size, and on macOS you are reading `resident_size_max`, so any transient spike during model load, graph compilation, prepacking, or first inference can keep the reported peak unchanged or higher even if later steady-state memory is lower. ([Man7][1])

For the ONNX Runtime CPU backend, “same RSS after quantization” is common. ORT’s CPU allocator is arena-based by default; the docs state that it allocates large regions, keeps unused chunks, and does not return that memory to the system once allocated. ORT also documents that CPU memory arena pre-allocation is expected behavior. Separately, ORT’s dynamic quantization computes activation quantization parameters on the fly, which adds runtime overhead rather than simply shrinking all runtime memory. ORT also notes that pre-packing initializers can allocate new buffers and undo some peak-memory savings. ([ONNX Runtime][2])

The CoreML result being worse is also plausible. ONNX Runtime’s CoreML EP converts ONNX subgraphs into Core ML models and can cache those converted models. The CoreML provider also specializes graph segments for target devices, which affects load behavior. At the same time, the published CoreML EP supported-operator lists are dominated by float operators such as `MatMul`, `Gemm`, `Add`, and `Conv`; they do not list common ONNX quantized operators like `MatMulInteger`, `QuantizeLinear`, `DequantizeLinear`, or `QLinear*`. That makes a typical ONNX int8 graph a poor native match for CoreML EP, so partial placement or CPU fallback is a realistic outcome, and ORT explicitly says fallback can hurt performance. ([GitHub][3])

Apple’s Core ML quantization guidance also points in the same direction. Their docs say quantization reduces weight size and can help memory-bottlenecked models, but activation quantization can slow execution on CPU and sometimes GPU because weights may be decompressed at load time while activations must be decompressed at runtime. They recommend activation quantization mainly when the model runs fully or mostly on the Neural Engine. So if your CoreML path is not mostly NE-backed, a quantized model using the CoreML route can easily show no RSS win, or even a worse one. ([Apple GitHub][4])

The practical conclusion is:

1. CPU EP: unchanged RSS is normal.
2. CoreML EP: increased RSS is also normal, especially for ONNX-native int8/QDQ/QOperator graphs.
3. This is not, by itself, evidence of a bad re-quantization.

To verify more precisely, measure in a fresh process and separate these phases: session creation, first inference, and later warmed-up inferences. For CPU EP, repeat the test with CPU memory arena disabled and with prepacking disabled, because both can mask weight-memory savings in RSS measurements. For CoreML EP, enable verbose logging and inspect placement/fallback, because unsupported quantized nodes are the most likely explanation for the regression. ORT exposes the relevant knobs and explicitly recommends verbose logs for seeing provider placement. ([ONNX Runtime][5])

If the target is Apple hardware specifically, the more relevant comparison is often not float32 ONNX vs int8 ONNX on CoreML EP, but float32/fp16 ONNX vs a Core ML-native compression strategy aligned with the intended compute unit. For CoreML EP, ONNX int8 quantization is not automatically the memory-optimal path. ([Apple GitHub][4])

[1]: https://man7.org/linux/man-pages/man2/getrusage.2.html "getrusage(2) - Linux manual page"
[2]: https://onnxruntime.ai/docs/get-started/with-c.html "C | onnxruntime"
[3]: https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h "onnxruntime/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h at main · microsoft/onnxruntime · GitHub"
[4]: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html "Performance — Guide to Core ML Tools"
[5]: https://onnxruntime.ai/docs/api/c/struct_ort_api.html?utm_source=chatgpt.com "OrtApi Struct Reference"
