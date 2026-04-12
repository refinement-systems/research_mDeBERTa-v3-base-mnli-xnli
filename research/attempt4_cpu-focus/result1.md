# Attempt 4 CPU Focus Result 1

Scope: first completed full execution of the bounded `attempt4` CPU deployment study from `plan1.md`.

This run completed the intended CPU-only study:

- development selection on gold accuracy plus float-reference fidelity
- persistent CPU runtime and RSS benchmarking on validation-complete candidates
- locked final test on the surviving set only
- separate stress reporting on HANS and WANLI

The result is clear enough to close CPU selection for this search space.

## 1. What Was Run

Scratchpad workspace:

- `scratchpad/attempt4_cpu_focus/models`
- `scratchpad/attempt4_cpu_focus/datasets`
- `scratchpad/attempt4_cpu_focus/candidates`
- `scratchpad/attempt4_cpu_focus/db.sqlite3`

Bounded CPU catalog evaluated in development:

- `reference`
- `model_quantized`
- `dynamic_qint8_default`
- `dynamic_qint8_per_channel`
- `attention_only`
- `nncf_accuracy_attention_only`
- `nncf_fidelity_attention_proj_only`
- `nncf_fidelity_attention_only_n128_drop0p005`
- `static_attention_only_u8u8_minmax_n128`
- `static_attention_proj_only_s8s8_minmax_n300`
- `static_attention_proj_only_u8s8_rr_minmax_n128`

Dataset roles used in this run:

`calibration`

- `mnli-train-calibration-64-per-label.tsv`
- `xnli-de-validation-calibration-32-per-label.tsv`
- `xnli-en-validation-calibration-32-per-label.tsv`
- `xnli-es-validation-calibration-32-per-label.tsv`
- `xnli-fr-validation-calibration-32-per-label.tsv`
- `xnli-zh-validation-calibration-32-per-label.tsv`

`smoke`

- `hf-probe-set.tsv`
- `hf-core-probe.tsv`

`fidelity_validation`

- `mnli-validation_matched-attempt4-dev.tsv`
- `mnli-validation_mismatched-attempt4-dev.tsv`
- `anli-r1-dev-attempt4-dev.tsv`
- `anli-r2-dev-attempt4-dev.tsv`
- `anli-r3-dev-attempt4-dev.tsv`

`fidelity_test`

- `xnli-{ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh}-test-attempt4-test.tsv`
- `anli-r1-test-attempt4-test.tsv`
- `anli-r2-test-attempt4-test.tsv`
- `anli-r3-test-attempt4-test.tsv`

`stress_test`

- `hans-evaluation-attempt4-stress-test.tsv`
- `wanli-test-attempt4-stress-test.tsv`

## 2. Development Gate Outcome

Locked after development gates:

- `nncf_accuracy_attention_only`
- `nncf_fidelity_attention_only_n128_drop0p005`
- `nncf_fidelity_attention_proj_only`
- `reference`

Screened out before locked final test:

- `attention_only`
- `dynamic_qint8_default`
- `dynamic_qint8_per_channel`
- `model_quantized`
- `static_attention_only_u8u8_minmax_n128`
- `static_attention_proj_only_s8s8_minmax_n300`
- `static_attention_proj_only_u8s8_rr_minmax_n128`

The gate pattern is important.

1. The tiny ~323 MiB baselines failed hard on task accuracy and float-label agreement.
2. The near-float static-QDQ controls failed mainly on fidelity and, for several variants, ANLI development drop.
3. `attention_only` came closest to surviving outside the final three quantized NNCF artifacts.
   - It kept development task accuracy near reference.
   - It still failed the frozen 98.0% float-label-agreement gate at `97.815%`.

So the development gates did what they were supposed to do:

- they rejected both the very small lossy tier
- and the large near-float tier that did not preserve float behavior well enough

## 3. Candidate Summary

Development gate readout from `attempt4-cpu-summary.md`:

| Candidate | Gate | Dev Acc | Dev Float Agree | CPU Warm | CPU Steady RSS | CPU Peak RSS | Cold Load |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention_only` | fail | 77.63% | 97.81% | 473.929 ms | 2864.2 MiB | 2865.7 MiB | - |
| `dynamic_qint8_default` | fail | 60.15% | 65.24% | 482.109 ms | 1025.2 MiB | 1031.3 MiB | - |
| `dynamic_qint8_per_channel` | fail | 60.50% | 65.57% | 478.272 ms | 1026.3 MiB | 1032.5 MiB | - |
| `model_quantized` | fail | 59.91% | 64.91% | 486.618 ms | 1026.4 MiB | 1034.0 MiB | - |
| `nncf_accuracy_attention_only` | pass | 77.63% | 98.19% | 743.025 ms | 1707.5 MiB | 1717.8 MiB | 3230.930 ms |
| `nncf_fidelity_attention_only_n128_drop0p005` | pass | 77.61% | 98.15% | 743.842 ms | 1707.1 MiB | 1714.1 MiB | 3224.450 ms |
| `nncf_fidelity_attention_proj_only` | pass | 77.62% | 98.27% | 736.538 ms | 1716.5 MiB | 1726.9 MiB | 3233.470 ms |
| `reference` | pass | 77.74% | 100.00% | 483.620 ms | 2859.2 MiB | 2860.7 MiB | 4273.770 ms |
| `static_attention_only_u8u8_minmax_n128` | fail | 77.14% | 93.05% | 682.902 ms | 2642.3 MiB | 2653.5 MiB | - |
| `static_attention_proj_only_s8s8_minmax_n300` | fail | 76.99% | 92.92% | 671.952 ms | 2656.2 MiB | 2667.6 MiB | - |
| `static_attention_proj_only_u8s8_rr_minmax_n128` | fail | 76.95% | 92.66% | 674.365 ms | 2656.2 MiB | 2667.6 MiB | - |

Two things stand out.

1. The only viable quantized CPU lane in this study is the mid-size NNCF tier around `465-485 MiB`.
2. None of the quantized survivors beat float on persistent warm latency.
   - The win is on disk size, cold load, and steady RSS.
   - The loss is on warm inference time.

## 4. Locked Final Frontier

Locked final frontier from `attempt4-cpu-summary.md`:

| Candidate | Size | Final Acc | Final Float Agree | CPU Warm | CPU Steady RSS | Frontier |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `nncf_accuracy_attention_only` | 465.1 MiB | 78.66% | 97.71% | 743.025 ms | 1707.5 MiB | frontier |
| `nncf_fidelity_attention_only_n128_drop0p005` | 465.1 MiB | 78.63% | 97.44% | 743.842 ms | 1707.1 MiB | frontier |
| `nncf_fidelity_attention_proj_only` | 485.3 MiB | 78.67% | 97.77% | 736.538 ms | 1716.5 MiB | frontier |
| `reference` | 1064.4 MiB | 78.74% | 100.00% | 483.620 ms | 2859.2 MiB | frontier |

Interpretation:

- `reference` remains on the frontier because it is much faster on persistent warm latency.
- `nncf_fidelity_attention_proj_only` remains on the frontier because it trades about `20 MiB` more size for slightly better final fidelity and slightly lower warm latency than the default winner.
- `nncf_fidelity_attention_only_n128_drop0p005` remains on the frontier because it ties the winner on size and is marginally lower on steady RSS.
- `nncf_accuracy_attention_only` remains on the frontier because it is the smallest locked point and wins the frozen tie-break among the `465 MiB` candidates.

## 5. CPU Recommendation

Default CPU recommendation for this search space:

- `nncf_accuracy_attention_only`

Reason:

- it is the smallest nondominated locked-final survivor
- it matches the size of the `n128_drop0p005` variant
- it is smaller than `nncf_fidelity_attention_proj_only`
- it keeps stronger final-test accuracy than the equal-size `n128_drop0p005` point
- it stays within a very small locked-final gold-accuracy delta versus float

Operational reading of the recommendation against `reference`:

- size on disk reduced by `56.3%`
- steady RSS reduced by `40.3%`
- cold load median improved by `24.4%`
- persistent warm latency worsened by `53.6%`
- locked-final gold accuracy drop versus float: `0.076` percentage points

So the correct reading is:

- `nncf_accuracy_attention_only` is the best tested CPU artifact when the objective is disk size + steady RSS + locked-final accuracy, with warm latency included as a frontier axis rather than the sole objective
- it is not the fastest CPU choice
- if persistent warm latency is the only operational priority, the float reference still wins

## 6. Multilingual And Robustness Readout

### 6.1. XNLI all-language readout

Reference XNLI test envelope:

- best language: English `88.32%`
- worst language: Urdu `74.03%`

`nncf_accuracy_attention_only` tracks that envelope closely:

- best language: English `88.20%`
- worst language: Urdu `73.87%`

Largest language-specific gold-accuracy drop versus reference for the default winner:

- Spanish: `0.56` points

Largest language-specific gain versus reference for the default winner:

- Swahili: `0.34` points

This is consistent with the aggregate locked-final result:

- the winner preserves the model’s multilingual fine-tuning behavior closely enough to remain the smallest surviving frontier point

### 6.2. ANLI locked-final readout

ANLI test remains hard for the whole model family, including float.

Per-round gold accuracy:

- `anli-r1`
  - `reference`: `32.2%`
  - `nncf_accuracy_attention_only`: `31.3%`
  - `nncf_fidelity_attention_only_n128_drop0p005`: `31.2%`
  - `nncf_fidelity_attention_proj_only`: `31.0%`
- `anli-r2`
  - `reference`: `28.5%`
  - `nncf_accuracy_attention_only`: `28.3%`
  - `nncf_fidelity_attention_only_n128_drop0p005`: `28.7%`
  - `nncf_fidelity_attention_proj_only`: `28.8%`
- `anli-r3`
  - `reference`: `31.67%`
  - `nncf_accuracy_attention_only`: `31.92%`
  - `nncf_fidelity_attention_only_n128_drop0p005`: `31.17%`
  - `nncf_fidelity_attention_proj_only`: `32.0%`

So the locked survivors did not collapse on ANLI. They stayed in the same difficult-performance regime as the float model.

### 6.3. Stress sets

Stress reporting is descriptive only and did not affect promotion.

HANS gold accuracy:

- `reference`: `71.97%`
- `nncf_accuracy_attention_only`: `72.10%`
- `nncf_fidelity_attention_only_n128_drop0p005`: `72.14%`
- `nncf_fidelity_attention_proj_only`: `72.20%`

WANLI gold accuracy:

- `reference`: `60.26%`
- `nncf_accuracy_attention_only`: `59.82%`
- `nncf_fidelity_attention_only_n128_drop0p005`: `59.80%`
- `nncf_fidelity_attention_proj_only`: `59.88%`

These stress results support the same broad reading:

- the locked NNCF tier stays close to float
- WANLI is slightly lower than float
- HANS does not show a new quantization-specific failure mode

## 7. Retry Metadata

The attempt4 reporting stack now surfaces generation metadata.

Locked frontier metadata:

- `nncf_accuracy_attention_only`
  - `smooth_quant_disabled=false`
  - `retry_reason=""`
- `nncf_fidelity_attention_proj_only`
  - `smooth_quant_disabled=false`
  - `retry_reason=""`
- `nncf_fidelity_attention_only_n128_drop0p005`
  - no non-empty retry metadata recorded in the staged artifact summary

No locked candidate carries a non-empty `retry_reason`.

## 8. Conclusion

`attempt4` answers the CPU deployment question for this bounded catalog.

Main result:

- the viable CPU quantized tier is the mid-size NNCF family around `465-485 MiB`
- the small dynamic baselines are too lossy
- the large static and attention-only anchors do not justify their size under the frozen gates
- `nncf_accuracy_attention_only` is the default CPU recommendation for this study

Most important caveat:

- CPU quantization here buys disk size, cold load, and resident memory
- it does not buy persistent warm latency

That means the repo now has a real CPU recommendation, but it is a storage-and-memory recommendation first, not a latency recommendation.

## 9. Evidence

Primary result artifacts:

- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary.md`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary.csv`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary-per-dataset.csv`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary-per-dataset.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary-per-language.csv`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary-per-language.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-validation-summary.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-test-summary.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-stress-summary.json`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-validation-cpu-persistent.csv`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-test-cpu-cold.csv`
- `scratchpad/attempt4_cpu_focus/reports/attempt4-manifest.json`
