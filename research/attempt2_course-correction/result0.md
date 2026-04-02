# Attempt 2 Result 0

Scope: first execution of the scratchpad-backed `attempt2` workflow from `plan0.md`.

This run replaced the old mixed objective with a backend-separated study of fidelity versus serialized size on disk. The selection rule was:

- pick candidates on `fidelity_validation`
- lock the backend-specific frontier
- run `fidelity_test` only on the locked frontier points
- treat `smoke` as diagnostic only

## 1. What Was Run

Runtime workspace:

- `scratchpad/models`
- `scratchpad/datasets`
- `scratchpad/candidates`
- `scratchpad/db.sqlite3`

Reference definition for this run:

- source artifact: `scratchpad/models/mdeberta/onnx/model.onnx`
- CPU reference behavior: float ONNX on CPU
- CoreML reference behavior: float ONNX on CoreML

Carry-forward catalog seeded for the run:

- `reference`
- `model_quantized`
- `dynamic_qint8_default`
- `dynamic_qint8_per_channel`
- `nncf_accuracy_attention_only`
- `nncf_fidelity_attention_proj_only`
- `attention_only`
- `attention_proj_only`

Operational notes:

- the frozen smoke assets were copied into `scratchpad/datasets` from the tracked probe TSVs
- `nncf_fidelity_attention_proj_only` remained `missing` in the artifact table and did not enter the evaluation matrix
- CoreML fallback was treated as a hard failure; fallback CPU results were not recorded as CoreML

## 2. Lane Outcomes

### 2.1. CPU lane

The CPU lane completed successfully.

Validation frontier points locked for test:

- `dynamic_qint8_default`
- `model_quantized`
- `nncf_accuracy_attention_only`
- `attention_only`
- `reference`

Validation-only candidates screened out before test:

- `dynamic_qint8_per_channel`
- `attention_proj_only`

Non-starting carry-forward candidate:

- `nncf_fidelity_attention_proj_only`

### 2.2. CoreML lane

The CoreML lane did not produce a valid baseline in this environment.

Observed behavior:

- `reference` on `hf-core-probe.tsv` opened with CoreML requested
- ONNX Runtime fell back to CPU execution
- the runner rejected that session and recorded a failed CoreML reference run with zero evaluation rows

Result:

- no CoreML validation frontier
- no CoreML test frontier
- no CoreML recommendation from this run

## 3. CPU Validation Frontier

Aggregated over the full `fidelity_validation` split:

| Quantization | Size (MiB) | Float-label agreement | Mean abs logit delta |
| --- | ---: | ---: | ---: |
| `dynamic_qint8_default` | 322.6 | 69.196% | 1.803702 |
| `model_quantized` | 323.0 | 69.345% | 1.818640 |
| `nncf_accuracy_attention_only` | 465.1 | 99.554% | 0.070974 |
| `attention_only` | 1036.1 | 99.851% | 0.092471 |
| `reference` | 1064.4 | 100.000% | 0.000000 |

Reading of the validation frontier:

- the two ~323 MiB dynamic baselines stayed on the frontier only because they are the smallest points; their fidelity is weak
- `nncf_accuracy_attention_only` is the first clearly strong middle point
- `attention_only` is slightly more faithful than `nncf_accuracy_attention_only`, but nearly float-sized
- `attention_proj_only` did not survive the size-versus-fidelity tie-break against `attention_only`

## 4. CPU Test Frontier

Aggregated over the locked `fidelity_test` split:

| Quantization | Size (MiB) | Float-label agreement | Mean abs logit delta |
| --- | ---: | ---: | ---: |
| `dynamic_qint8_default` | 322.6 | 65.897% | 1.712398 |
| `model_quantized` | 323.0 | 66.205% | 1.709938 |
| `nncf_accuracy_attention_only` | 465.1 | 98.359% | 0.105864 |
| `attention_only` | 1036.1 | 98.718% | 0.113791 |
| `reference` | 1064.4 | 100.000% | 0.000000 |

Test readout:

- the small dynamic tier stayed small, but fidelity dropped further on the untouched test split
- `nncf_accuracy_attention_only` kept a strong fidelity profile while remaining about `56.3%` smaller than float
- `attention_only` stayed somewhat more faithful, but only reduced disk size by about `2.7%` relative to float

## 5. Recommendation

CPU recommendation:

- primary quantized recommendation: `nncf_accuracy_attention_only`

Reason:

- it is the best size-versus-fidelity compromise in this run
- it cuts on-disk size from `1064.4 MiB` to `465.1 MiB`
- it keeps `98.359%` float-label agreement on the untouched test split
- the larger `attention_only` point buys only a small fidelity gain for almost no additional size savings versus float

Not recommended as default CPU picks:

- `dynamic_qint8_default`
- `model_quantized`

Reason:

- they are much smaller, but their fidelity loss is too large to make them defensible default recommendations for this model family

CoreML recommendation:

- no recommendation from this run

Reason:

- the CoreML reference itself fell back to CPU, so the lane never obtained a valid backend-specific float baseline

## 6. Dataset Inventory Used In This Run

`calibration`

- `mnli-train-calibration-64-per-label.tsv`
- `xnli-de-validation-calibration-32-per-label.tsv`
- `xnli-en-validation-calibration-32-per-label.tsv`
- `xnli-es-validation-calibration-32-per-label.tsv`
- `xnli-fr-validation-calibration-32-per-label.tsv`
- `xnli-zh-validation-calibration-32-per-label.tsv`

`fidelity_validation`

- `mnli-train-search-validation-skip64-64-per-label.tsv`
- `xnli-de-validation-search-validation-skip32-32-per-label.tsv`
- `xnli-en-validation-search-validation-skip32-32-per-label.tsv`
- `xnli-es-validation-search-validation-skip32-32-per-label.tsv`
- `xnli-fr-validation-search-validation-skip32-32-per-label.tsv`
- `xnli-zh-validation-search-validation-skip32-32-per-label.tsv`

`fidelity_test`

- `mnli-validation_matched-200-per-label.tsv`
- `mnli-validation_mismatched-200-per-label.tsv`
- `xnli-de-test-50-per-label.tsv`
- `xnli-en-test-50-per-label.tsv`
- `xnli-es-test-50-per-label.tsv`
- `xnli-fr-test-50-per-label.tsv`
- `xnli-zh-test-50-per-label.tsv`

`smoke`

- `hf-core-probe.tsv`
- `hf-probe-set.tsv`

## 7. Report Artifacts

Per-dataset summaries:

- `scratchpad/reports/validation-summary.csv`
- `scratchpad/reports/validation-summary.json`
- `scratchpad/reports/test-summary.csv`
- `scratchpad/reports/test-summary.json`

Aggregated frontier summaries:

- `scratchpad/reports/validation-frontier-aggregated.json`
- `scratchpad/reports/test-frontier-aggregated.json`
