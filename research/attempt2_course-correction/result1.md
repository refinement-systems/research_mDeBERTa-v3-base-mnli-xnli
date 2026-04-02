# Attempt 2 Result 1

Scope: `plan1.md` follow-up after `result0.md`.

This note records the current `plan1` validation state. It is still a CPU-only result, because the CoreML lane remains blocked at backend validation. It is also still a validation-only result: no untouched `fidelity_test_plan1` run has been executed yet.

## 1. Current Validated CPU Frontier

These numbers come from the aggregated `fidelity_validation` report in `scratchpad/plan1`.

| quantization | size on disk | float-label agreement | mean abs logit delta | status |
| --- | ---: | ---: | ---: | --- |
| `model_quantized` | 323.0 MiB | 68.171% | 1.779537 | frontier |
| `nncf_accuracy_attention_only` | 465.1 MiB | 99.421% | 0.071612 | frontier |
| `nncf_fidelity_attention_only_n128_drop0p005` | 465.1 MiB | 99.537% | 0.144470 | frontier |
| `nncf_fidelity_attention_proj_only` | 485.3 MiB | 99.653% | 0.065814 | frontier |
| `reference` | 1064.4 MiB | 100.000% | 0.000000 | frontier |

The most important non-frontier anchor is:

| quantization | size on disk | float-label agreement | mean abs logit delta | status |
| --- | ---: | ---: | ---: | --- |
| `attention_only` | 1036.1 MiB | 99.537% | 0.084663 | dominated |

## 2. Reading

- `nncf_fidelity_attention_proj_only` is currently the strongest validated quantized CPU point in `plan1`.
- `nncf_fidelity_attention_only_n128_drop0p005` is a real frontier point, but it does not beat `nncf_fidelity_attention_proj_only` on the combined size-versus-fidelity reading.
- `nncf_fidelity_attention_only_n128_drop0p005` does beat the old float-sized `attention_only` anchor on size while matching it on aggregated float-label agreement.
- `nncf_accuracy_attention_only` remains a strong middle point because it is slightly smaller than the new `n128_drop0p005` artifact and has a materially lower mean absolute logit delta.

If a recommendation had to be made from validation only, the current best quantized CPU point would be `nncf_fidelity_attention_proj_only`.

## 3. Machine-Limited `attention_only` Follow-Up

The `attention_only` fidelity-oriented refinement batch was pruned by what this machine could realistically sustain.

- `nncf_fidelity_attention_only_n128_drop0p005`
  - materialized successfully
  - smoke completed on `hf-probe-set.tsv` and `hf-core-probe.tsv`
  - full `fidelity_validation` completed
  - classification: slow but somewhat applicable on this machine
- `nncf_fidelity_attention_only_n128_drop0p002`
  - did not complete
  - artifact status was set to `failed`
  - classification: too slow for this machine
- the `n300` pair was not started after the machine-based pruning decision

## 4. Current Recommendation Status

- CoreML: still no valid recommendation from `plan1`, because the backend-validity problem remains unresolved.
- CPU: `nncf_fidelity_attention_proj_only` is the current validation leader.
- Final `plan1` recommendation: not locked yet, because no untouched `fidelity_test_plan1` run has been executed.

## 5. Evidence

Primary report artifacts:

- [validation-summary.json](/Users/mjm/repo/nli/scratchpad/plan1/reports/validation-summary.json)
- [validation-frontier-aggregated.json](/Users/mjm/repo/nli/scratchpad/plan1/reports/validation-frontier-aggregated.json)

Supporting artifact:

- [nncf_fidelity_attention_only_n128_drop0p005.onnx](/Users/mjm/repo/nli/scratchpad/plan1/candidates/plan1/nncf_fidelity_attention_only_n128_drop0p005.onnx)
