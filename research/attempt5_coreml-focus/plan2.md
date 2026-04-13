# Attempt 5 CoreML Focus Plan 2

Scope: optional cleanup after the negative `result0.md` preflight.

`result0.md` already answered the main backend question:

- the current ORT CoreML path is not competitive enough on this machine to justify a full locked-final CoreML study

So there is no default recommendation to continue `attempt5` as originally scoped.

## 1. Default Action

Default action:

- stop `attempt5` here
- keep the closed CPU recommendation as the deployment default
- treat `result0.md` as the CoreML backend readout for the current implementation path

This is the recommended path.

## 2. Optional Follow-Up

There is only one remaining follow-up worth considering:

- run `nncf_accuracy_attention_only` on CoreML through the same validation + persistent benchmark path

Reason:

- it is the closed CPU winner
- it is the only missing control that could still change the narrative from “CoreML is not competitive” to “the CPU-winning artifact transfers unusually well to CoreML”

Do not run:

- locked final test
- stress test
- the discarded extra CPU frontier controls

## 3. How To Run The Optional Follow-Up

Reuse the current attempt5 scratchpad and add only the missing control:

```bash
python3 tools/run-attempt5-coreml-study.py --skip-test
```

Expected behavior:

- existing `reference` and `reference_fp16` validation rows are reused from the current scratchpad
- the runner adds `nncf_accuracy_attention_only`
- the persistent CoreML benchmark is recomputed over the validation-complete set
- no locked-final or stress evaluation is run

## 4. Decision Rule For The Optional Follow-Up

If you run the control, the bar should be strict.

`nncf_accuracy_attention_only` would need to show something operationally meaningful that `reference_fp16` did not:

- better warm latency than `reference_fp16`
- while keeping a real size or memory advantage

If it does not do that, close `attempt5` immediately.

## 5. Bottom Line

`plan2` is optional.

The recommended decision is already:

- close the CoreML lane for the current ORT path on this machine

Only run the single missing control if you want one extra archival data point for the CPU-winner transfer question.
