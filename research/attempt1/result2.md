# Attempt 1 Result 2

Scope: updated attempt1 analysis after the corrected static QDQ frontier sweep from `plan2.md`, plus the operational memory findings from that sweep.

This note updates the attempt1 picture after the first serious static-QDQ batch was run end to end on the corrected pipeline.

## 1. What Was Run

The corrected Batch A static sweep covered:

- ignored scope families:
  - `attention_only`
  - `attention_proj_only`
- datatype / format variants:
  - `S8S8 + QDQ`
  - `U8U8 + QDQ`
  - `U8S8 + reduce_range + QDQ`
- calibration methods:
  - `minmax`
  - `percentile`
- calibration caps:
  - `128`
  - `300`

Shared static settings:

- `preprocess=true`
- `max_examples_per_source=0`
- probe-gated screening through:
  - core probe
  - hard probe
  - full suite only for promoted candidates

The corrected summary is in:

- `benchmarks/nli/attempt1-static-frontier-summary-v2.csv`
- `benchmarks/nli/attempt1-static-frontier-summary-v2.json`

## 2. Results

Attempt0 reference frontier:

| Candidate | Full Acc | Full HF | Hard Acc | Core Acc | `xnli-zh` Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| `float` | `86.36%` | `100.00%` | `45.90%` | `44.00%` | `86.00%` |
| `attention_only` | `86.67%` | `98.77%` | `55.74%` | `52.00%` | `85.33%` |
| `attention_proj_only` | `86.51%` | `98.82%` | `50.82%` | `40.00%` | `85.33%` |

Promoted static-QDQ survivors:

| Candidate | Full Acc | Full HF | Hard Acc | Core Acc | `xnli-zh` Full | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `static_attention_only_u8u8_minmax_n128` | `86.21%` | `96.62%` | `57.38%` | `52.00%` | `86.00%` | Best static accuracy / Chinese preservation |
| `static_attention_only_s8s8_minmax_n300` | `85.74%` | `96.87%` | `47.54%` | `44.00%` | `85.33%` | Probe survivor, not frontier-level |
| `static_attention_only_u8u8_minmax_n300` | `85.74%` | `96.15%` | `50.82%` | `44.00%` | `84.00%` | Inferior to `n128` |
| `static_attention_proj_only_u8s8_rr_minmax_n128` | `85.85%` | `96.72%` | `52.46%` | `48.00%` | `86.00%` | Good probes, still too much fidelity loss |
| `static_attention_proj_only_s8s8_minmax_n300` | `86.15%` | `96.92%` | `49.18%` | `44.00%` | `84.67%` | Best static fidelity result |
| `static_attention_proj_only_u8u8_minmax_n300` | `86.05%` | `96.56%` | `50.82%` | `44.00%` | `86.00%` | Good Chinese preservation, weak fidelity |

Negative results:

- All `12` `percentile` candidates failed during static quantization.
- Multiple minmax candidates were screened out before the full suite.
- No static candidate beat the attempt0 frontier on its target metric.

## 3. Analysis

### 3.1. Static QDQ is viable, but not frontier-changing

This batch did produce real survivors. Static QDQ is not a dead path in the sense of "everything immediately breaks."

The strongest positive signal is:

- several static candidates preserved `xnli-zh` at `86.00%`, matching float and beating both attempt0 dynamic finalists on that slice

The strongest negative signal is:

- every surviving static candidate gave up too much full-suite HF agreement to challenge either the attempt0 fidelity frontier or the best attempt1 NNCF result

So the fair conclusion is:

- static QDQ can preserve some localized behavior
- static QDQ did not produce a new balanced best model

### 3.2. The best static candidate still misses the real bars

`static_attention_only_u8u8_minmax_n128` is the best static accuracy-oriented result:

- full accuracy: `86.21%`
- `xnli-zh`: `86.00%`

That is interesting because it keeps the Chinese slice at float level and beats float on the hard probe.

But it is still not promotable:

- it is below float on full accuracy
- it is below `attention_only` on full accuracy
- its full HF agreement (`96.62%`) is far below the quantized frontier

`static_attention_proj_only_s8s8_minmax_n300` is the best static fidelity-oriented result:

- full accuracy: `86.15%`
- full HF agreement: `96.92%`

That is still far below:

- attempt0 `attention_proj_only` at `98.82%`
- attempt1 `nncf_fidelity_attention_proj_only` at `98.82%`

So static did not beat the best dynamic or mixed-precision candidates on the metric where it needed to win.

### 3.3. Histogram-style static calibration is not workable here

This batch answers the percentile question clearly enough for attempt1:

- the `percentile` branch is not just weak
- it is operationally broken on this model / ORT path / machine combination

That means histogram-based static calibration should not remain in the main attempt1 search space unless the quantization implementation changes materially.

### 3.4. Broad static sweeps are too expensive for this machine

This is the operationally important update.

During the static sweep, a Python process was observed peaking above `50 GB` on a machine with `16 GB` of RAM. Even if that peak mixes allocator behavior and swap-backed virtual memory, it is still a strong sign that the broad static matrix is an unhealthy fit for this machine.

The conclusion is not just "the run was slow." It is:

- broad static sweeps can create severe memory pressure
- that pressure is not justified by the quality upside observed so far
- static QDQ should stop being treated as the mainline attempt1 path

### 3.5. `plan2` was useful, but its main hypothesis does not survive the evidence

`plan2.md` correctly prioritized the static-QDQ axis as the cheapest serious unexplored path at the time.

After the corrected static sweep, that claim no longer holds:

- the path is not cheap in practice on this machine
- it did not beat the attempt0 frontier
- it did not beat the best attempt1 NNCF candidate

So the repo should now pivot away from "broad static frontier search" and back toward narrower high-signal work.

## 4. What Changed

The attempt1 ranking now looks like this:

1. `attention_only` remains the best quantized accuracy-oriented artifact.
2. `attention_proj_only` remains the best quantized fidelity baseline.
3. `nncf_fidelity_attention_proj_only` remains the best attempt1 direction.
4. Static QDQ is now a secondary branch for targeted Chinese-preservation experiments, not the mainline search path.

## 5. Recommended Next Step

The strongest next attempt1 path is now:

- a very small NNCF fidelity refinement batch centered on `attention_proj_only`

Recommended first grid:

- `metric=hf_agreement`
- `ignored_scope_family=attention_proj_only`
- `subset_size=128`
- `max_drop`:
  - `0.005`
  - `0.002`
- bias correction:
  - `accurate`

Operationally, this is a better fit because:

- the search space is much smaller
- the best prior attempt1 evidence already points in this direction
- it avoids repeating the highest-memory low-upside static matrix

## 6. Bottom Line

The corrected static sweep delivered a clear answer:

- static QDQ can produce valid survivors
- some static candidates preserve `xnli-zh` unusually well
- the fidelity cost is still too large
- the percentile branch is broken for attempt1 purposes
- the broad static matrix is too memory-expensive for this machine

So the attempt1 mainline should now pivot to narrow NNCF fidelity refinement, with static retained only as a targeted side branch.
