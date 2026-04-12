# Attempt 5 CoreML Focus Plan 1

Scope: narrowed CoreML follow-up after the CPU lane was closed in `attempt4_cpu-focus/result1.md`.

`attempt4` already established the CPU default:

- `nncf_accuracy_attention_only`

The practical CoreML question is now smaller than `plan0`.

- Do `reference` or `reference_fp16` beat the closed CPU winner on this Apple-silicon machine?
- Does the single CPU-winning artifact remain interesting when pushed through the CoreML path?

Because the full CPU run was already very expensive on this machine, `plan1` drops the extra carry-forward CPU finalists and keeps only the winner as the CoreML control.

## 1. Goal

The goal of `attempt5` in `plan1` is:

1. verify that CoreML EP runs cleanly outside the sandbox on the broad attempt4 pack
2. retest `reference_fp16` as the primary CoreML candidate
3. compare it against:
   - `reference`
   - `nncf_accuracy_attention_only`
4. compute a CoreML frontier on:
   - on-disk size
   - persistent warm latency
   - resident-after-warmup RSS
5. determine the default CoreML recommendation for this machine

This is still a backend study, not a reopened quantization search.

## 2. Candidate Set

Primary CoreML candidates:

- `reference`
- `reference_fp16`

Single carry-forward CoreML control:

- `nncf_accuracy_attention_only`

Reason for the narrower control set:

- `nncf_accuracy_attention_only` is the closed CPU recommendation
- it is the only control that must be tested to answer whether the CPU winner transfers to CoreML well enough to matter
- the other locked CPU frontier points are not necessary for the next decision and would increase runtime substantially

Do not add in `plan1`:

- the other two locked CPU frontier points
- the tiny dynamic ORT tier
- more static-QDQ candidates
- a new NNCF search
- Core ML native compression work
- Neural-Engine-specific `W8A8` or `int4` claims

## 3. Dataset Pack

Reuse the attempt4 pack structure under:

- `scratchpad/attempt5_coreml_focus`

using the same roles:

- `calibration`
- `smoke`
- `fidelity_validation`
- `fidelity_test`
- `stress_test`

This keeps the CoreML result directly comparable to the closed CPU result.

## 4. Selection Rule

Use the same development gates as `attempt4`, but against the CoreML float reference.

Development gates:

- MNLI dev macro accuracy drop must be `<= 0.5` points vs CoreML float
- ANLI dev macro accuracy drop must be `<= 1.0` point vs CoreML float
- no individual MNLI split or ANLI dev round may drop by more than `1.5` points vs CoreML float
- aggregate float-label agreement must be `>= 98.0%`
- peak RSS must be recorded and must not exceed CoreML float by more than `25%`

Locked-final frontier axes:

- on-disk size
- persistent warm median latency
- resident-after-warmup RSS

Peak RSS remains a guardrail, not a frontier axis.

Recommendation tie-break:

1. smallest nondominated locked-final survivor
2. lower persistent warm latency
3. lower steady RSS
4. higher locked-final aggregate accuracy

## 5. Expected Outcome

The most likely result is still:

- `reference_fp16` remains the default CoreML recommendation

The important negative control is now narrower:

- if `nncf_accuracy_attention_only` does not transfer well to CoreML, that is enough to close the question for the current carry-forward CPU artifact lane

If it does transfer well, the later result should compare it directly against both:

- CoreML `reference_fp16`
- the closed CPU default on CPU

## 6. Entry Point

Prepared entry point for local terminal execution:

- `tools/run-attempt5-coreml-study.py`

Prepared catalog:

- `research/attempt5_coreml-focus/study_quantization_catalog.json`

Prepared report builder:

- `tools/build-attempt5-coreml-report.py`

Suggested full run command:

```bash
python3 tools/run-attempt5-coreml-study.py --force
```

Suggested narrow preflight command:

```bash
python3 tools/run-attempt5-coreml-study.py --skip-controls --skip-test --force
```

Run it in your own terminal outside the Codex sandbox.

## 7. Bottom Line

`plan1` asks one bounded question:

- is CoreML `fp16` better than float and better than the single closed CPU winner on this machine?

That is the next decision point. Everything else can wait until after this run finishes.
