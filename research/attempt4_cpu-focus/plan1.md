# Attempt 4 CPU Focus Plan 1

Scope: next execution after `result0.md`.

`result0.md` established that the redesigned CPU study is implemented and runnable, but that a complete bounded execution is too long to finish interactively on this machine. So `plan1` is not another methodology change. It is an execution and decision plan for finishing the study cleanly and then closing CPU selection.

## 1. Objective

Complete one full bounded `attempt4` CPU run and use its locked outputs to answer exactly these questions:

1. Which quantized CPU artifacts survive the development gates?
2. What is the final locked CPU Pareto frontier on:
   - on-disk size
   - persistent warm latency
   - resident-after-warmup RSS
3. What is the default CPU recommendation under the frozen tie-break:
   - smallest nondominated survivor
   - then lower persistent warm latency
   - then lower steady RSS
   - then higher locked-final aggregate accuracy

No new search-space expansion should happen before this full run is completed.

## 2. Candidate Set

The study should stay bounded to these exact CPU candidates:

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

Do not add:

- CoreML candidates
- new NNCF search variants
- new static-QDQ sweeps
- QAT
- QAttention

The point of `plan1` is to finish the frozen CPU study, not to reopen search.

## 3. Recommended Execution

### 3.1. Preflight

Run these first:

```bash
python3 -m unittest tests.python.test_summarize_study_db tests.python.test_study_catalog
python3 tools/prepare-attempt4-cpu-datasets.py --scratchpad-root scratchpad/attempt4_cpu_focus
```

Expected result:

- tests pass
- dataset pack validates
- disjointness check reports `31` attempt4 slices

Use `--force-datasets` only if you intentionally want to rebuild the pack or suspect local dataset corruption.

### 3.2. Full run

Run the full study from a normal local terminal:

```bash
python3 tools/run-attempt4-cpu-study.py --force
```

Notes:

- `--force` recreates the study DB and scratchpad-backed study state
- it does not force dataset regeneration unless `--force-datasets` is also supplied
- on this machine, the run should be treated as multi-hour and likely overnight

If you only want a development-only checkpoint first:

```bash
python3 tools/run-attempt4-cpu-study.py --skip-test --force
```

### 3.3. Sandbox note

If this is run under Codex or another restricted sandbox, the RSS benchmark step may need unsandboxed execution because the macOS memory wrapper uses `/usr/bin/time -l`.

If this is run in a normal local shell, no special handling should be needed.

## 4. Expected Workflow Stages

The run should proceed in this order:

1. Prepare datasets
2. Stage runtime assets into `scratchpad/attempt4_cpu_focus/models`
3. `nli-study init`
4. Verify dataset-role assignments
5. Run `smoke` + `fidelity_validation` on the full bounded CPU catalog
6. Summarize validation
7. Benchmark persistent CPU runtime and RSS for validation-complete candidates
8. Build intermediate report and freeze locked quantizations
9. Run locked `fidelity_test`
10. Run locked `stress_test`
11. Run cold benchmark on the locked set
12. Build the final report package

If the run stops before step 8, there is still no CPU selection result.

## 5. Success Criteria

`plan1` is successful only if all of the following are true.

- final manifest written:
  - `scratchpad/attempt4_cpu_focus/reports/attempt4-manifest.json`
- final report written:
  - `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary.md`
- candidate summary JSON written:
  - `scratchpad/attempt4_cpu_focus/reports/attempt4-cpu-summary.json`
- per-dataset and per-language tables written
- locked quantizations are explicitly recorded
- final frontier is explicitly recorded
- recommendation is non-null unless only `reference` survives

The study is also expected to preserve these reporting properties:

- gold accuracy included
- float-label agreement included
- runtime and RSS included
- `smooth_quant_disabled` and `retry_reason` surfaced
- HANS scored as binary entailment vs non-entailment only

## 6. Decision Rules After The Run

### 6.1. If one or more quantized artifacts survive and a frontier exists

Then do this:

- write `research/attempt4_cpu-focus/result1.md`
- declare the default CPU recommendation from the final report
- close CPU search unless there is a concrete reason to challenge the winner

### 6.2. If only `reference` survives the development gates

Then do not immediately loosen the gates.

First write down:

- which gate each quantized candidate failed
- whether failures are mostly:
  - task accuracy
  - float-label agreement
  - peak RSS
  - missing/failed execution

Only after that should a narrower follow-up be considered.

The most defensible fallback options, in order, are:

1. accept float CPU as the current operational default
2. rerun only the strongest near-miss candidates after fixing execution problems
3. consider one tightly scoped CPU follow-up if failures cluster on one correctable issue

### 6.3. If a quantized candidate survives but offers weak operational gains

For example:

- size drops materially but RSS barely moves
- or latency worsens too much for the saved size

Then the decision should still follow the frozen frontier rule, but the write-up should say plainly that the artifact is a storage win more than a runtime-memory win.

## 7. What Not To Change Mid-Run

Do not change any of the following after starting the decisive full run:

- candidate set
- dataset roles
- development gates
- final frontier axes
- recommendation tie-break

If any of those need to change, abort the run, record why, and treat the next attempt as a new plan rather than quietly mutating `plan1`.

## 8. Deliverables After Completion

Once the full run finishes, the next write-up should include:

- final locked quantization list
- final CPU frontier table
- default CPU recommendation
- per-language XNLI readout
- ANLI and WANLI/HANS readout
- explicit note on any SmoothQuant retry metadata
- exact evidence file list under `scratchpad/attempt4_cpu_focus/reports/`

## 9. Bottom Line

`plan1` is intentionally simple:

- do not redesign the study again
- do not expand the search
- run the bounded CPU study to completion
- document the locked result
- only then decide whether any further CPU work is justified
