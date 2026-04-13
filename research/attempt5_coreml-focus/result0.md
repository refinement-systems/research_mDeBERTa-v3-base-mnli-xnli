# Attempt 5 CoreML Focus Result 0

Scope: short `plan1` preflight run with `--skip-controls --skip-test`.

This run was meant to answer one question before spending more time on CoreML:

- is the current ORT CoreML path on this machine competitive enough to justify a full locked-final run?

The answer is no.

## 1. What Was Run

Scratchpad workspace:

- `scratchpad/attempt5_coreml_focus/models`
- `scratchpad/attempt5_coreml_focus/datasets`
- `scratchpad/attempt5_coreml_focus/db.sqlite3`

Candidates evaluated:

- `reference`
- `reference_fp16`

Not run in this preflight:

- `nncf_accuracy_attention_only`
- locked final test
- stress test

Datasets used:

- `hf-probe-set.tsv`
- `hf-core-probe.tsv`
- `mnli-validation_matched-attempt4-dev.tsv`
- `mnli-validation_mismatched-attempt4-dev.tsv`
- `anli-r1-dev-attempt4-dev.tsv`
- `anli-r2-dev-attempt4-dev.tsv`
- `anli-r3-dev-attempt4-dev.tsv`

## 2. Validation Outcome

Both CoreML primary candidates passed the frozen development gates.

Validation summary from `attempt5-coreml-summary.md`:

| Candidate | Gate | Dev Acc | Dev Float Agree | CoreML Warm | CoreML Steady RSS | CoreML Peak RSS | Load |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reference` | pass | 75.75% | 100.00% | 503.142 ms | 4175.0 MiB | 6322.5 MiB | 19280.200 ms |
| `reference_fp16` | pass | 75.74% | 99.99% | 1862.160 ms | 1444.7 MiB | 1746.0 MiB | 5829.820 ms |

Important fidelity readout:

- `reference_fp16` lost only `0.0067` percentage points of aggregate dev gold accuracy versus CoreML float
- aggregate float-label agreement stayed at `99.993%`
- MNLI dev macro drop was `0.0`
- ANLI dev macro drop was `0.033` points

So model quality is not the problem.

## 3. Operational Readout

The backend story is the problem.

### 3.1. CoreML float reference

`reference` under CoreML is not operationally attractive versus the closed CPU baselines.

Compared with CPU float from `attempt4`:

- warm latency is roughly the same
  - CoreML float: `503.142 ms`
  - CPU float: `483.620 ms`
- load time is much worse
  - CoreML float: `19.28 s`
  - CPU float: `4.27 s`
- steady RSS is much worse
  - CoreML float: `4175.0 MiB`
  - CPU float: `2859.2 MiB`
- peak RSS is dramatically worse
  - CoreML float: `6322.5 MiB`
  - CPU float: `2860.7 MiB`

So the current float CoreML path is already weaker than CPU on this machine.

### 3.2. CoreML fp16

`reference_fp16` improves some deployment metrics versus CoreML float:

- size is cut from `1064.4 MiB` to `532.9 MiB`
- load median drops from `19.28 s` to `5.83 s`
- steady RSS drops from `4175.0 MiB` to `1444.7 MiB`

But the warm path collapses:

- CoreML float warm median: `503.142 ms`
- CoreML fp16 warm median: `1862.160 ms`

That is about `3.7x` slower than CoreML float.

Compared with the closed CPU default `nncf_accuracy_attention_only` from `attempt4`:

- CoreML fp16 uses less steady RSS
  - `1444.7 MiB` vs `1707.5 MiB`
- but load is still worse
  - `5.83 s` vs `3.23 s`
- and warm latency is much worse
  - `1862.160 ms` vs `743.025 ms`

So even the promising fp16 lane does not beat the closed CPU winner on the metrics that matter for deployment.

## 4. Provider Partitioning Readout

The ORT CoreML warnings were informative.

Observed support during session creation:

- `reference`
  - `636 / 1941` nodes supported by CoreML
  - `99` CoreML partitions
- `reference_fp16`
  - `62 / 2061` nodes supported by CoreML
  - `24` CoreML partitions

Interpretation:

- the float model gets a meaningful but still partial CoreML split
- the fp16 model gets almost no meaningful CoreML coverage
- that support collapse is the most likely explanation for the fp16 warm-latency regression

This is still a real CoreML-partitioned run, not a silent full fallback to CPU, because the study code refuses to record a full backend fallback as CoreML.

## 5. Conclusion

This short preflight is already enough to change the decision.

- Do not treat CoreML as a leading deployment backend on this machine under the current ORT path.
- Do not spend a full locked-final run on `reference` and `reference_fp16` just to confirm what the operational metrics already show.
- The closed CPU recommendation remains the stronger deployment default:
  - `nncf_accuracy_attention_only`

The only remaining reason to continue `attempt5` would be documentation completeness:

- run the single CPU-winning control on CoreML as a last negative-control check

That is optional. The primary backend question is already answered.
