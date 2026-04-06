# Attempt 2 Result 1

Scope: `plan1.md` follow-up after `result0.md`.

This note records the completed CPU outcome of `plan1`. CoreML was explicitly deferred for later once the CPU lane became the priority, so this is a CPU-only result.

## 1. Validation Frontier

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

## 2. Untouched Test Frontier

Only the locked validation frontier points were allowed onto `fidelity_test_plan1`, which keeps the split discipline intact.

| quantization | size on disk | float-label agreement | mean abs logit delta | status |
| --- | ---: | ---: | ---: | --- |
| `model_quantized` | 323.0 MiB | 64.148% | 1.745576 | frontier |
| `nncf_accuracy_attention_only` | 465.1 MiB | 98.741% | 0.111492 | frontier |
| `nncf_fidelity_attention_only_n128_drop0p005` | 465.1 MiB | 98.519% | 0.178286 | dominated |
| `nncf_fidelity_attention_proj_only` | 485.3 MiB | 98.741% | 0.100908 | dominated |
| `reference` | 1064.4 MiB | 100.000% | 0.000000 | frontier |

## 3. Reading

- `nncf_fidelity_attention_proj_only` won validation, but it did not preserve a unique edge on untouched test. On test it matched `nncf_accuracy_attention_only` on float-label agreement while remaining larger, so it is dominated on the stated objective.
- `nncf_fidelity_attention_only_n128_drop0p005` was a real validation frontier point, but it regressed on untouched test and did not displace `nncf_accuracy_attention_only`.
- `nncf_accuracy_attention_only` therefore remains the best practical CPU recommendation after `plan1`.
- `attention_only` never re-entered contention: the smaller `n128_drop0p005` candidate matched it on validation agreement and removed the case for carrying the almost-float-sized anchor into test.

## 4. CPU Recommendation

- CPU recommendation after untouched `plan1` test: `nncf_accuracy_attention_only`
- Rationale: it remains on the final test frontier at 465.1 MiB and 98.741% float-label agreement.
- CoreML recommendation: deferred. This run intentionally stopped at the CPU conclusion.

## 5. Machine-Limited `attention_only` Follow-Up

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

## 6. Evidence

Primary report artifacts:

- [validation-summary.json](scratchpad/plan1/reports/validation-summary.json)
- [validation-frontier-aggregated.json](scratchpad/plan1/reports/validation-frontier-aggregated.json)
- [test-summary.json](scratchpad/plan1/reports/test-summary.json)
- [test-frontier-aggregated.json](scratchpad/plan1/reports/test-frontier-aggregated.json)

Supporting artifact:

- [nncf_fidelity_attention_only_n128_drop0p005.onnx](scratchpad/plan1/candidates/plan1/nncf_fidelity_attention_only_n128_drop0p005.onnx)
