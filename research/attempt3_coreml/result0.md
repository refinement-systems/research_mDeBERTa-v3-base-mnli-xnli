# Attempt 3 CoreML Result 0

Scope: first completed CoreML-specific execution from `plan0.md`.

This run was intentionally narrow. It asked whether the CoreML lane could be exercised outside the sandbox and, if so, whether `fp16` should be the default CoreML target on this `M1` machine under the current objective:

- backend-specific fidelity to a backend-specific float reference
- size on disk
- no performance benchmarking yet

## 1. What Was Run

Scratchpad workspace:

- `scratchpad/attempt3_coreml/models`
- `scratchpad/attempt3_coreml/datasets`
- `scratchpad/attempt3_coreml/candidates`
- `scratchpad/attempt3_coreml/db.sqlite3`

Catalog entries available for the attempt:

- `reference`
- `reference_fp16`
- `model_quantized`
- `dynamic_qint8_default`

What actually entered the evaluation matrix:

- `reference`
- `reference_fp16`

What did not run in this execution:

- `model_quantized`
- `dynamic_qint8_default`

Reason:

- the run was executed without `--include-int8-controls`
- this kept the attempt focused on the primary `fp16` question first

## 2. Lane Outcome

The CoreML lane completed successfully outside the sandbox.

Backend integrity:

- completed evaluation logs record `Using ONNX Runtime backend: CoreML`
- no fallback CPU runs were recorded as CoreML
- smoke, validation, and test all completed for both `reference` and `reference_fp16`

Operational caveat:

- the run emitted the expected CoreML EP partial-partition warnings during execution
- this means not all graph nodes were placed on CoreML
- that does not invalidate the fidelity result, but it means this run should not be read as a speed claim

## 3. Validation Frontier

Aggregated over the locked `fidelity_validation` pack:

| quantization | size on disk | float-label agreement | mean abs logit delta | max abs logit delta | status |
| --- | ---: | ---: | ---: | ---: | --- |
| `reference_fp16` | 532.9 MiB | 100.000% | 0.000684 | 0.008212 | frontier |
| `reference` | 1064.4 MiB | 100.000% | 0.000000 | 0.000000 | dominated by smaller point |

Validation readout:

- `reference_fp16` matched `reference` on float-label agreement across all `672` validation examples
- it introduced zero label disagreements
- it cut serialized size by about `49.9%` relative to the float reference

## 4. Untouched Test Frontier

Only the locked validation points were allowed onto `fidelity_test`.

| quantization | size on disk | float-label agreement | mean abs logit delta | max abs logit delta | status |
| --- | ---: | ---: | ---: | ---: | --- |
| `reference_fp16` | 532.9 MiB | 100.000% | 0.000904 | 0.027337 | frontier |
| `reference` | 1064.4 MiB | 100.000% | 0.000000 | 0.000000 | dominated by smaller point |

Test readout:

- `reference_fp16` again matched `reference` on float-label agreement across all `1050` untouched test examples
- it again produced zero disagreements
- the largest observed absolute logit drift on the test pack was `0.027337`

## 5. Recommendation

CoreML recommendation for this `M1` run:

- primary recommendation: `reference_fp16`

Reason:

- it is the sole non-reference frontier point on both validation and test
- it preserves `100.000%` float-label agreement against the CoreML float reference on both splits
- it reduces on-disk size from `1064.4 MiB` to `532.9 MiB`

What this does and does not mean:

- it supports the `fp16`-first CoreML plan
- it does not yet say that `fp16` is faster, because this attempt deliberately omitted runtime benchmarks
- it does not yet say that integer ONNX controls are uninteresting, because they were not executed in this run

## 6. Integer-Control Status

This result is not an `int8` verdict.

- `model_quantized` was present as a downloaded control artifact but was not evaluated
- `dynamic_qint8_default` remained unevaluated
- the next CoreML follow-up can still test integer controls as secondary, measurement-driven candidates

The important point from `result0` is narrower:

- for the current ONNX Runtime CoreML EP path on this `M1`, `fp16` is already good enough to become the baseline recommendation

## 7. Dataset Inventory Used In This Run

`smoke`

- `hf-probe-set.tsv`
- `hf-core-probe.tsv`

`fidelity_validation`

- `mnli-train-attempt3-coreml-search-validation-skip256-64-per-label.tsv`
- `xnli-en-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`
- `xnli-de-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`
- `xnli-es-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`
- `xnli-fr-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`
- `xnli-zh-validation-attempt3-coreml-search-validation-skip96-32-per-label.tsv`

`fidelity_test`

- `mnli-validation_matched-attempt3-coreml-test-skip300-50-per-label.tsv`
- `mnli-validation_mismatched-attempt3-coreml-test-skip300-50-per-label.tsv`
- `xnli-en-test-attempt3-coreml-test-skip100-50-per-label.tsv`
- `xnli-de-test-attempt3-coreml-test-skip100-50-per-label.tsv`
- `xnli-es-test-attempt3-coreml-test-skip100-50-per-label.tsv`
- `xnli-fr-test-attempt3-coreml-test-skip100-50-per-label.tsv`
- `xnli-zh-test-attempt3-coreml-test-skip100-50-per-label.tsv`

Calibration note:

- no calibration datasets were used in this execution because integer controls were not run

## 8. Evidence

Primary report artifacts:

- `scratchpad/attempt3_coreml/reports/validation-summary.json`
- `scratchpad/attempt3_coreml/reports/validation-frontier-aggregated.json`
- `scratchpad/attempt3_coreml/reports/test-summary.json`
- `scratchpad/attempt3_coreml/reports/test-frontier-aggregated.json`
- `scratchpad/attempt3_coreml/reports/attempt3-coreml-plan0-manifest.json`
