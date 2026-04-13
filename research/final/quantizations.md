Scope: all model-format variants used in `attempt0` through `attempt5`, with emphasis on how each artifact was obtained, what part of the graph was quantized, and to what precision.

This document is about model artifacts, not datasets or benchmark roles. It includes the integer quantization families, the CoreML `fp16` deployment conversion, and the failed QAT branch for completeness.

## 1. Precision Legend

- `fp32`: regular float32 ONNX weights and activations.
- `fp16`: float16 weights/internal tensors, but integer text inputs and float logits can still be preserved.
- dynamic int8: ONNX Runtime `quantize_dynamic` style. Weights are stored quantized, activations are quantized at runtime before integer matmuls.
- static QDQ: ONNX Runtime `quantize_static` style using `QuantizeLinear` / `DequantizeLinear` nodes around selected subgraphs.
- NNCF PTQ: NNCF post-training quantization, exported back to ONNX as a QDQ graph.
- QAT: quantization-aware training. This was attempted, but no usable ONNX artifact was produced.

## 2. Non-Integer Baselines

These are not integer quantizations, but they were part of the study and are useful reference points.

### 2.1. `reference`

- Artifact: `models/mdeberta/onnx/model.onnx`
- Obtained by: downloading the float ONNX export from `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- Created or staged by:
  - `tools/download-mdeberta-v3-base.sh`
  - later staged into study scratchpads with `tools/stage-study-artifact.py`
- What is quantized: nothing
- Precision:
  - weights: `FLOAT`
  - internal numerics: `FLOAT`
  - inputs: `INT64`
  - logits: `FLOAT`

### 2.2. `reference_fp16`

- Artifact: `candidates/reference_fp16.onnx` in the CoreML scratchpads
- Created by: `tools/convert-onnx-to-fp16.py`
- How it works:
  - uses `onnxruntime.transformers.float16.convert_float_to_float16`
  - keeps model I/O types by default
- What is quantized: nothing in the integer-quantization sense; this is a float16 conversion
- Precision:
  - weights and internal float tensors: `FLOAT16`
  - inputs: still `INT64`
  - logits: still `FLOAT`

## 3. Official Hugging Face Quantized Export

### 3.1. `model_quantized`

- Artifact: `models/mdeberta/onnx/model_quantized.onnx`
- Obtained by: downloading the official `onnx/model_quantized.onnx` file from `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- Created upstream by: ONNX Runtime quantization
  - local graph metadata shows `producer = onnx.quantize 0.1.0`
  - metadata includes `onnx.infer = onnxruntime.quant`
- Graph style:
  - `DynamicQuantizeLinear`: `50`
  - `MatMulInteger`: `74`
  - `DequantizeLinear`: `1`
- What was quantized:
  - the word embedding table
  - `74` learned matrix weights:
    - `12` attention query projections
    - `12` attention key projections
    - `12` attention value projections
    - `12` attention output dense layers
    - `12` FFN intermediate dense layers
    - `12` FFN output dense layers
    - `1` pooler dense
    - `1` classifier
- Precision:
  - embedding table: `UINT8`, asymmetric, per-tensor
  - learned matmul weights: `INT8`, effectively symmetric, per-channel
  - activations entering `MatMulInteger`: dynamically quantized at runtime
  - accumulator path: integer matmul followed by rescaling back to `FLOAT`
  - inputs: `INT64`
  - logits: `FLOAT`
- Important wrinkle:
  - this file still keeps `24` float copies of some query/key projection weights for secondary reuse paths, so it is not a perfectly clean "all selected weights only exist in int8" export

## 4. ONNX Runtime Dynamic Quantization

### 4.1. Full-model ORT baselines

These were generated with `tools/quantize-onnx-model.py`, which wraps ONNX Runtime `quantize_dynamic`.

#### `dynamic_qint8_default`

- Artifact: `models/mdeberta/onnx/candidates/dynamic_qint8_default.onnx`
- Created by:
  - `tools/quantize-onnx-model.py`
  - preset: `mdeberta-study`
  - weight type: `qint8`
  - per-channel: `false`
- What was quantized:
  - same effective scope as the official Hugging Face quantized export:
    - quantized word embedding table
    - the same `74` learned matrix weights listed above
- Precision:
  - embedding table: `UINT8`
  - learned matmul weights: `INT8`
  - weight scales and zero points: scalar per tensor for the local default artifact
  - activations for quantized matmuls: dynamic runtime quantization
- Notes:
  - same quantization family as the official Hugging Face export
  - not byte-identical to the downloaded file, but structurally the same style

#### `dynamic_qint8_per_channel`

- Artifact: `models/mdeberta/onnx/candidates/dynamic_qint8_per_channel.onnx`
- Created by:
  - `tools/quantize-onnx-model.py`
  - weight type: `qint8`
  - per-channel: `true`
- What was quantized:
  - same scope as `dynamic_qint8_default`
- Precision:
  - embedding table: `UINT8`
  - learned matmul weights: `INT8`
  - key difference from `dynamic_qint8_default`:
    - matmul weight scales and zero points are vectors, not scalars
    - example: `onnx::MatMul_4664_scale` has shape `[768]` instead of `[]`

#### `dynamic_qint8_matmul`

- Artifact family root: generated by `tools/quantize-onnx-model.py --op-type MatMul`
- What was quantized:
  - only constant-B `MatMul` weights
  - no embedding-table quantization
- Precision:
  - learned matmul weights: `INT8`
  - activations: dynamic runtime quantization before integer matmuls

#### `dynamic_qint8_matmul_per_channel`

- Same as `dynamic_qint8_matmul`, but with per-channel weight quantization.

#### `dynamic_quint8_matmul`

- Same as `dynamic_qint8_matmul`, but with `quint8` weights instead of `qint8`.
- This was explored as an ORT datatype baseline, not a promoted frontier artifact.

### 4.2. Historical single-example exclusion artifact

#### `dynamic_qint8_matmul_exclude_suggested`

- Artifact: `models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx`
- Created by:
  - `tools/quantize-onnx-model.py --preset single`
  - `--op-type MatMul`
  - a fixed hand-built `--nodes-to-exclude` list taken from the early single-example drift debugging
- What was quantized:
  - dynamic int8 matmul quantization, but with a specific small exclusion set left float
- Precision:
  - same precision style as the other ORT dynamic matmul-only artifacts:
    - quantized weights in `INT8`
    - dynamic runtime activation quantization

### 4.3. Family-search dynamic artifacts

These were created by `tools/search-onnx-quantization-families.py`.

Shared method:

- base quantizer: `tools/quantize-onnx-model.py`
- quantization style: ORT dynamic `MatMul` quantization
- weight type: `qint8`
- per-channel: `false`
- reduce-range: `false`
- embedding table: not quantized

The difference between these artifacts is which named encoder `MatMul` nodes were kept in float.

#### Candidate list

- `baseline_dynamic_matmul`
  - no exclusions
- `current_best_reference`
  - historical carry-forward from the early exclusion search
- `ffn_layers_8_11_float`
  - keep FFN intermediate and FFN output dense matmuls float in layers `8-11`
- `ffn_layers_6_11_float`
  - keep FFN intermediate and FFN output dense matmuls float in layers `6-11`
- `ffn_layers_4_11_float`
  - keep FFN intermediate and FFN output dense matmuls float in layers `4-11`
- `attention_output_layers_8_11_float`
  - keep attention-output dense matmuls float in layers `8-11`
- `layer_11_block_float`
  - keep the full quantizable block float in layer `11`
- `layers_10_11_block_float`
  - keep the full quantizable block float in layers `10-11`
- `attention_only`
  - quantize attention-side learned matmuls only
  - keep FFN dense matmuls float
  - actual quantized learned weights:
    - query, key, value, attention-output dense in all `12` layers
    - pooler
    - classifier
  - local graph count: `50` `MatMulInteger` nodes
- `ffn_only`
  - quantize FFN and output-dense matmuls only
  - keep attention-side learned matmuls float
- `attention_proj_only`
  - quantize only attention query/key/value projections
  - keep attention-output dense and all FFN dense matmuls float
  - pooler and classifier remained quantized
  - local graph count: `38` `MatMulInteger` nodes
- `attention_only_layer_11_float`
  - `attention_only`, but layer `11` attention-side matmuls left float
- `attention_only_layers_10_11_float`
  - `attention_only`, but layers `10-11` attention-side matmuls left float
- `attention_only_attention_output_layers_8_11_float`
  - `attention_only`, but upper-layer attention-output dense matmuls left float
- `attention_proj_only_layer_11_float`
  - `attention_proj_only`, but layer `11` attention projections left float
- `attention_proj_only_layers_10_11_float`
  - `attention_proj_only`, but layers `10-11` attention projections left float

## 5. ONNX Runtime Static QDQ

These artifacts were generated with `tools/quantize-onnx-static.py`.

Shared method:

- base quantizer: ONNX Runtime `quantize_static`
- graph format: `QDQ` only
- quantized op family: `MatMul`
- extra option: `MatMulConstBOnly = True`
- calibration source: TSV NLI slices encoded through the repo tokenizer
- calibration methods tried:
  - `minmax`
  - `percentile`
- subset sizes tried:
  - `128`
  - `300`
- per-channel: `false`

Naming scheme used in the sweep:

- `static_<family>_<dtype>_<method>_n<subset>`

Families used:

- `attention_only`
- `attention_proj_only`

Datatype combinations used:

- `s8s8`
  - activations: `INT8`
  - weights: `INT8`
- `u8u8`
  - activations: `UINT8`
  - weights: `UINT8`
- `u8s8_rr`
  - activations: `UINT8`
  - weights: `INT8`
  - `reduce_range = true`

What was quantized:

- the selected `MatMul` families plus the surrounding activation tensors needed by the QDQ graph
- unlike the official Hugging Face export and the ORT full-model dynamic baselines, this branch did not quantize the word-embedding table itself

Important operational result:

- all `percentile` variants failed on this model / machine stack
- only `minmax` survivors remained relevant

### 5.1. Named static artifacts that actually mattered later

- `static_attention_only_u8u8_minmax_n128`
  - attention-side family
  - activations `UINT8`, weights `UINT8`
- `static_attention_only_s8s8_minmax_n300`
  - attention-side family
  - activations `INT8`, weights `INT8`
- `static_attention_only_u8u8_minmax_n300`
  - attention-side family
  - activations `UINT8`, weights `UINT8`
- `static_attention_proj_only_u8s8_rr_minmax_n128`
  - attention-projection family
  - activations `UINT8`, weights `INT8`
  - reduced range enabled
- `static_attention_proj_only_s8s8_minmax_n300`
  - attention-projection family
  - activations `INT8`, weights `INT8`
- `static_attention_proj_only_u8u8_minmax_n300`
  - attention-projection family
  - activations `UINT8`, weights `UINT8`

### 5.2. Static survivors carried into the final CPU study

The bounded `attempt4` CPU deployment study only carried forward these static controls:

- `static_attention_only_u8u8_minmax_n128`
- `static_attention_proj_only_s8s8_minmax_n300`
- `static_attention_proj_only_u8s8_rr_minmax_n128`

## 6. NNCF Post-Training Quantization

These artifacts were created with `tools/quantize-onnx-nncf.py`.

Shared method:

- base tool: NNCF ONNX PTQ / accuracy-control
- model type: `TRANSFORMER`
- preset: `mixed`
- preprocessing: enabled
- bias correction:
  - usually `fast`
  - `accurate` was tried later and failed operationally
- SmoothQuant:
  - enabled by default
  - the tool includes a retry path that disables SmoothQuant if the first attempt fails

Observed exported graph style:

- QDQ graph, not `MatMulInteger`
- dominant precision on this model: signed `INT8`
- local graph inspection shows:
  - `QuantizeLinear` / `DequantizeLinear` pairs throughout the quantized subgraph
  - quantized embedding table stored as `INT8`
  - `nncf_smooth_quant_scale` tensors in `FLOAT`
- Unlike the ORT family-search dynamic artifacts, the NNCF exports did quantize the embedding table.

### 6.1. Stable benchmarked NNCF artifacts

#### `nncf_accuracy_attention_only`

- Created by:
  - `mode = accuracy-control`
  - `metric = gold_accuracy`
  - `ignored_scope_family = attention_only`
  - `subset_size = 300`
  - `max_drop = 0.01`
  - `fast_bias_correction = true`
- What was quantized:
  - embeddings
  - attention projections
  - attention-output dense branch
  - pooler and classifier
  - FFN dense modules were kept float by the ignored scope
- Precision:
  - dominant exported quantized tensors: `INT8`
  - some auxiliary zero-point tensors appear as `UINT8`

#### `nncf_fidelity_attention_proj_only`

- Created by:
  - `mode = accuracy-control`
  - `metric = hf_agreement`
  - `ignored_scope_family = attention_proj_only`
  - `subset_size = 300`
  - `max_drop = 0.01`
  - `fast_bias_correction = true`
- What was quantized:
  - embeddings
  - attention projections
  - pooler and classifier
  - attention-output dense and FFN dense modules were kept float
- Precision:
  - same general exported style as `nncf_accuracy_attention_only`:
    - QDQ graph
    - mostly `INT8`

#### `nncf_fidelity_attention_only_n128_drop0p005`

- Created by the narrow attempt2 plan1 refinement
- Configuration:
  - `mode = accuracy-control`
  - `metric = hf_agreement`
  - `ignored_scope_family = attention_only`
  - `subset_size = 128`
  - `max_drop = 0.005`
  - `fast_bias_correction = true`
- What was quantized:
  - same intended scope as `nncf_accuracy_attention_only`, but optimized for float-label fidelity instead of gold accuracy
- Precision:
  - same exported NNCF style as the other stable NNCF artifacts:
    - QDQ graph
    - mostly `INT8`

### 6.2. Narrow NNCF variants that were tried but not promoted

#### `nncf_fidelity_attention_proj_only_n128_drop0p005`

- Created by the `attempt1` plan3 micro-batch
- Configuration:
  - `metric = hf_agreement`
  - `ignored_scope_family = attention_proj_only`
  - `subset_size = 128`
  - `max_drop = 0.005`
- Outcome:
  - valid artifact
  - screened out before full-suite promotion
- Precision and scope:
  - same NNCF `INT8` QDQ style as the other attention-projection NNCF artifacts

#### `nncf_fidelity_attention_proj_only_n128_drop0p002`

- Same family as the previous artifact, but `max_drop = 0.002`
- Outcome:
  - valid artifact
  - screened out before full-suite promotion
- Precision and scope:
  - same NNCF `INT8` QDQ style as the other attention-projection NNCF artifacts

### 6.3. Planned or partially run NNCF configurations

The following objective-aligned attempt2 refinement configurations were explicitly part of the work, but only `nncf_fidelity_attention_only_n128_drop0p005` became a stable carry-forward artifact:

- `nncf_fidelity_attention_only_n128_drop0p002`
- `nncf_fidelity_attention_only_n300_drop0p005`
- `nncf_fidelity_attention_only_n300_drop0p002`

These were meant to use the same NNCF QDQ `INT8` style as the other attention-only NNCF artifacts, but they were retired as too slow or not useful enough to keep on the mainline path.

## 7. Quantization-Aware Training (QAT) Pilot

### 7.1. Intended artifact

- Planned output: `models/mdeberta/onnx/candidates/qat/attention_only_qat_pilot.onnx`
- Tool: `tools/recover-mdeberta-qat.py`

### 7.2. How it was supposed to work

- load the HF sequence-classification model
- build an NNCF mixed-precision PTQ seed
- optionally ignore the same family scopes used elsewhere:
  - `attention_only`
  - `attention_proj_only`
- fine-tune for a small number of epochs
- export back to ONNX
- strip quantization wrappers with `nncf.strip(..., strip_format=nncf.StripFormat.DQ)` when possible

### 7.3. Intended precision

- same general family as the NNCF PTQ exports:
  - quantized transformer-aware mixed-precision graph
  - expected ONNX export in a QDQ / DQ-stripped int8 style

### 7.4. Actual outcome

- this branch never produced a benchmarkable ONNX artifact
- so there is no final on-disk QAT artifact whose exact precision layout can be inspected

## 8. Practical End State

By the end of the work, the important artifacts were:

- faithful float baseline:
  - `reference`
- official vendor quantized baseline:
  - `model_quantized`
- best hand-built ORT dynamic accuracy artifact:
  - `attention_only`
- best hand-built ORT dynamic fidelity artifact:
  - `attention_proj_only`
- final CPU recommendation:
  - `nncf_accuracy_attention_only`
- CoreML deployment format baseline:
  - `reference_fp16`

That means the repo ended with four distinct quantization stories:

1. the official Hugging Face ORT dynamic export (`model_quantized`)
2. local ORT dynamic integer quantization with family exclusions
3. local ORT static QDQ quantization
4. local NNCF transformer-aware PTQ

The final winning CPU artifact came from the fourth group.
