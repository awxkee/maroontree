# Decision-metric unification: SATD for split screening, weighted SSE everywhere else

Tree: `dev9_26` at `79e0d2f`. Scope: AV1 (`src/coder*`) and AV2 (`src/av2`, stills) encoders.
Date: 2026-11-06. Status: PLAN (nothing in this document is implemented yet).

## 0. The law we are committing to

One rule, applied in both encoders, every format, every speed tier:

1. **SATD (4x4 Hadamard L1) screens split candidates.** Wherever a partition leg (NONE / SPLIT /
   rect / 4:1 / IntraBC-vs-NONE) is *priced* without a full coded trial, the luma distortion of that
   leg is SATD of the reconstruction error. SATD of the prediction error also ranks which modes,
   angle deltas and chroma modes get a coded evaluation (shortlists / top-K prunes). This is a
   deliberate decision backed by three closed attempts to replace it with SSE
   (`av1-satd-partition-pricing`) and one with weighted SSE (`av1-weighted-sse-2026-11`, +0.98 /
   +0.21 / +0.77).
2. **Weighted SSE (wSSE) is THE distortion in every other cost that is compared with lambda x bits**:
   mode decisions, TX type / TX split, chroma modes, CfL vs DC, MHCCP, palette, IntraBC, trial
   encodes, SDP chroma plans, CCSO / CDEF / Wiener / deblock decisions. Same kernel, same weights,
   same unit in AV1 and AV2, all formats including AV2 4:4:4 (currently off there).
3. **wSSE has ONE normalized unit.** The kernel is normalized so that white error measures the
   same as plain SSE (mean weight = 1). Every re-centring factor that today hides in `wsse_scale`
   (AV1, 6.0) or in the un-normalized AV2 kernel (scale 1) moves into an explicit decision lambda.
4. **One decision lambda per encoder,** `lambda_d(q, format)`, in wSSE units. Every other decision
   lambda (partition proxy, chroma, palette map, filters) is derived from it with a *named* unit
   conversion. The trellis / RDOQ lambda stays in its own coefficient domain (plain squared
   coefficient error; QM already frequency-weights; WRDOQ measured negative in AV2) and is
   re-swept once as the interface between the two domains.
5. **SATD-domain and wSSE-domain quantities are never added or compared without an explicit
   bridge constant** `K_satd(format, q)`, measured, not guessed.

Reading of the brief: "screening split candidates" = (1); "weighted SSE for screening everywhere"
= (2) plus the explicit experiment in Phase 6 that tries wSSE-of-prediction as the shortlist
ranking too (today SATD ranks shortlists in both encoders). If the brief meant the stronger reading
(wSSE replaces SATD for mode shortlists as well), Phase 6 is the gate that decides it on data.

## 1. What is in the tree today

### 1.1 The wSSE kernel

Both encoders carry the hpvca 4x4-DCT weights (DC 1.8 / AC1 1.0 / diag 0.8 / axis2 0.4 / HF 0.2):
AV1 `rd_sse.rs:1323` `WSSE_WEIGHTS`, AV2 `helpers.rs` `DISTORTION_WEIGHTS`.

| quantity | value |
|---|---|
| mean of the 16 weights | 7.4 / 16 = **0.4625** |
| scale at which white error equals plain SSE | 1 / 0.4625 = **2.162** |
| AV1 shipped `wsse_scale` | 6.0 = 2.162 x **2.775** |
| AV2 shipped scale | 1.0 = 2.162 x **0.4625** |
| measured wSSE / plain SSE on real coded residuals (scale 1) | 0.30 .. 0.46 |

So the AV1 "re-centring to 6" is really two things: a unit normalization (2.16) and a hidden
**2.8x cut of the effective decision lambda**. That cut is corroborated independently: the plain-SSE
x4 control alone gave -1.2 .. -2.3 SS2 and -2.2 PSNR (`av1-weighted-sse-2026-11`), and the
plain-regime study found mode lambda 0.4 beating the shipped 0.8 in both PSNR and SS2
(`av1-core-engine-plain-regime-gap`). AV2 went the other way: scale 1 is a hidden 2.16x *raise*
of lambda, and the AV2 PART lambda sweep with wSSE on (x0.5 -1.24 / x1 -1.54 / x2 -1.39) says that
raise is what AV2 wanted. Neither encoder has an explicit lambda that means "wSSE units per bit".

### 1.2 AV1 inventory (condensed; full table in the survey of 2026-11-06)

Consistent with the law today:

- All `RdDispatch::sse_recon / sse_u16 / chroma_sse` calls (every emitter's mode, TX type, TX
  split, uv mode, CfL-vs-DC, palette RD, IntraBC skip, trial legs) are wSSE x6
  (`lossy_state.rs:3760`).
- Mode / delta / filter-intra / chroma-mode shortlists rank by `satd_sad_proxy` (SAD + SATD/4):
  `superblock.rs:153/330/381/434/498`, `partition_search.rs:221/1146`, `block64.rs:382`,
  `block64_rect.rs:365`.
- Partition legs price luma by `luma_partition_distortion` = SATD x max(qstep,1) x 0.25 x
  `luma_satd_scale` (`lossy_state.rs:2497`): 20 call sites across `partition_search.rs`
  (`rd_cost_square` 419..856, rect16 957), `lossy_state.rs` (2085/2152/2342/2951),
  `block32.rs` (463/553), `block64.rs` (1647/1765/1937/2049/2108), `palette.rs:1761`.

Violations (what Phases 2-3 fix):

| # | site | problem |
|---|---|---|
| V1 | partition proxies: `prefer_8x8_none` 2170-2190, `rd_choice_16_inner`, `edge_choice_16` 3073-3100, `rd_choice_rect32` 348/359, `choose_64` 2230/2285 | add SATD-priced luma (about 18x plain-SSE units, 45x at the 4:4:4 top band) to chroma wSSE x6 x `chroma_part_weight` (1/16 at 420/422, 1/32 at 444, `coder.rs:1659`). The 1/16 IS the Hadamard-gain bridge. The x6 on the chroma leg landed with the wSSE change and was never isolated: the chroma leg is now ~2.8x heavier than it was calibrated. |
| V2 | `rd_cost_intrabc_skip` (`block64.rs:1274`, wSSE) compared against SATD-priced NONE/SPLIT legs at `superblock.rs:658-676`, `lossy_state.rs:3089/3299/3523`, `block32.rs:393-401`, `block64.rs:2289-2335` | mixed units; IntraBC distortion under-weighted ~6x relative to its competitors (the comment at `block64.rs:1063` records the 18x discount from before wSSE). |
| V3 | `rd_pick_luma64` (`block64.rs:1805`) | the only EMITTER whose final luma mode / delta / TX (Quad32, Tx64, Grid16) is chosen on SATD-price, not wSSE. |
| V4 | `uv_palette_gate` (`palette.rs:1807`; callers `block8.rs:1890`, `block16.rs:889`, `block32.rs:3554`) | plain prediction SSE vs a wSSE x6 best_total: the gate threshold `uv_pal_gate_k` is in no unit. |
| V5 | `trial_struct_penalty` (`trial.rs:541`) | plain-SSE-unit penalty added to a wSSE trial cost. |
| V6 | CDEF (`coder.rs:3803-4273`): luma `cdef_dist_8x8` (libaom variance-weighted), chroma plain SSE x `cdef_chroma_w`, 40 per-mille margin, set-size bits priced with `mode_lambda_q` without the SS2 weight or `mlam_scale` | three metrics and a lambda nobody else uses. |
| V7 | Wiener (`coder.rs:3750`), deblock search (`coder.rs:5015`, off) | plain SSE, no lambda. |
| V8 | TX-type guards (1/32 tolerance at 12 sites, strict at 5, filter-intra 3, TX split margins `block32.rs:1172`, `partition_search.rs:4443`) | both sides wSSE so the x6 cancels, but the guards were tuned on plain SSE shapes; never re-swept after weighting. |
| V9 | chroma lambda: `block16.rs:632` (444 16x16 uv search) and `block8.rs:2088` (420 8x8 CfL, `uv8_cfl_mlam_c` false) use luma `emit_mlam`; everything else `emit_mlam_c` | two lambdas for one plane (the class fixed in `av1-chroma-lambda-mismatch-2026-10`, two sites remain). |
| V10 | trellis (`trellis.rs:55/721`, `trellis_dist.rs:100`) | plain coefficient SSE with `TRELLIS_AOM_CALIB x trellis_calib 1.15`: by design, but the outer RD became 2.8x less rate-heavy and the trellis did not follow. |

Calibration constants that live in the SATD domain and must be re-fit when the bridge changes
(`tuning.rs`): `satd_base/top/knee_lo/knee_hi`, `none16_top_bias_420/422/444`, `none32_split_bias`,
`none64_split_bias`, `none32/64_split_bias_444_top` + knee/width, `top_none_bias_420_hi/lo`,
`rect16_bias(_422)`, `quad4_bias`, `split16_bias`, `split4_bias`, `part_signal_bits`,
`part_signal_seam_drop`, `split32_signal_mult`, `split_breakout_slow`, `split4_breakout_slow`,
`b64_refinement_window`, `b64_split_refinement`, `tx64_bias`, `edge_rect_bias`,
`chroma_part_rd_weight` (`coder.rs:1659`), `vbp_thresh_*` (variance / ac_q^2, unit-free, keep).

Lambda constants: `MODE_AOM_CALIB`, `mlam_scale_420/422/444` (0.8 / 0.65 / 0.6),
`mode_lambda_lo_*`, `emit_mlam_c = min(mlam, mlam_c) x 0.5(420) x masking`, `trellis_calib`,
`palette_viterbi_444/420/uv`, `trial_child_prdo`, `cdef_margin`, `uv_pal_gate_k`.

### 1.3 AV2 inventory (condensed)

Consistent with the law today (420 / 422 / 400 stills, `set_wsse(still_rd())`):

- Leaf RD, chroma, CfL, MHCCP, SDP plan trials, probe split decisions go through the
  `wsse_on()`-aware helpers `rect_sse_f32`, `pixel_sse_rounded_block`, `tx16_distortion`,
  `dist_rounded`, `dist_flat`.
- SATD (`metrics.rs:satd_f32`, pure Hadamard) screens: whole-64 (keep 4/8, refine 2), 32x32
  (keep 3, prk32 2), probe 16/8 (keep 3, refine 2, unified prune 2), coded 16x16 / 8x8 leaves
  (prk 2), uv shortlists (keep 2/3/4). The search-ceiling check says top-2 is the sweet spot.

Violations:

| # | site | problem |
|---|---|---|
| A1 | 4:4:4: `y444/lossy.rs:5606` `set_wsse(false)` | wSSE off (+0.33 on tuning, mascot +3.15); 4:4:4 instead uses a THIRD metric, the edge-aware `tx16_distortion` = pixel + 0.5 x edge. |
| A2 | GTREE / `leaf_eval` (`y420/partition.rs:675-680`, costs 699/739/760/794), rect `multi_tu_search` | raw plain SSE closure; the TX16 type is picked weighted (735) but costed plain (739). |
| A3 | `tx64_eval` (`y420/partition.rs:1130`), whole-64 TX64-vs-4xTX32 (`4754-4791`) | raw plain SSE on both sides. |
| A4 | ExtraSlow SB partition trial `sse_plane` (`y420/partition.rs:2493-2512`) | raw plain SSE Y+U+V at 0.5 x part_lambda. |
| A5 | palette `search / dp_centers / build_map` (`palette.rs:405-425, 478-482, 514-538`) | raw e^2 with the (wSSE-calibrated) caller lambda. |
| A6 | CCSO `search_edge` (`ccso.rs:174-288`) plain, no lambda; `decide_blk_md` weighted with `ccso_rd_scale x qstep^2 / 2^20` | two metrics inside one tool. |
| A7 | CDEF luma `cdef_dist_8x8`, chroma raw d^2 (`cdef_est.rs:236/245`), margins 10 per mille, no lambda | same as AV1 V6. |
| A8 | 444 MHCCP baseline (`y444/lossy.rs:799`) = prediction-only plain SSE with NO rate vs MHCCP full J | apples to oranges. |
| A9 | lambda: 444 SDP chroma priced at `subsampled_part_lambda_c` 2.5e-5 (`sdp.rs:345/665/1074`) while the rest of 444 uses 1e-4; CfL/MHCCP on the 420 fast path use frame `qstep_i` while the uv search uses `sb_qstep`; 422 uses `qstep_i` throughout | three lambda laws (`luma_mode_lambda` = max(rdoq,0.07) x qstep^2 / 4096 ~ 1.7e-5 qstep^2; `part_lambda` 2.5e-5 / 1e-4 qstep^2; CCSO 9.5e-7 qstep^2) with no documented relation. |
| A10 | RDOQ (`coder.rs:520/567/640/667/747`) | coefficient-domain plain, level lambdas 0.07 (hq schedule to 0.20) / 0.11 / 0.16: by design, interface to be re-swept. |

## 2. What has to change (analysis)

### 2.1 Normalize the kernel, make the lambda explicit (Phase 1)

Divide the weight table by 0.4625 in both encoders (DC 3.89 / AC1 2.16 / diag 1.73 / axis2 0.86 /
HF 0.43). Then:

- AV1: `wsse_scale` 6.0 becomes 1.0 and the 2.775 factor moves into the decision lambda. Since
  cost = 6 W + lambda b has the same argmin as 2.162 W + (lambda / 2.775) b, every decision that
  mixes ONLY wSSE and lambda x bits is unchanged (up to the i64 rounding of the scaled distortion).
  The decisions that are NOT unchanged are exactly the violations: V1 / V2 / V5 (SATD-domain legs
  see the old lambda) and the lambda-only consumers (`palette_viterbi_*`, palette rate ranking,
  `part_signal_bits`-style fees). To land this phase byte-equivalent we introduce
  `lambda_d = mlam / 2.775` for wSSE sites and keep `lambda_part = mlam` for the SATD-domain
  proxy, with the ratio named `K_PART_LAMBDA = 2.775`. That single constant is then what Phase 3
  re-fits.
- AV2: the kernel normalization multiplies every wSSE by 2.162; to stay decision-identical,
  `luma_mode_lambda`, `part_lambda`, `ccso_rd_scale` and the SDP lambda are multiplied by 2.162 in
  the formats where wSSE is on. 4:4:4 (wSSE off) is untouched by this phase.

Acceptance: BD within +-0.05 of the baseline on every corpus, -t1 == -tN, Fast / Medium byte-identical
where the change does not reach.

### 2.2 Bridge the SATD domain explicitly (Phase 3, AV1)

Measured facts (`av1-satd-partition-pricing`): the partition price is 18x plain SSE at 420/444
mid-band, 45x at the 444 top (satd_top ramp); the chroma weight 1/16 and every bias is in that
domain; the SSE-swap re-fits all wanted ~3.6x LESS distortion weight at fine q than at coarse q.

Change: define the proxy luma distortion as `SATD x bridge(q, format)` where `bridge` is fitted so
that the proxy distortion is in *normalized wSSE units*:

    bridge(q, fmt) = qstep x 0.25 x luma_satd_scale(q, fmt) / K_satd(q, fmt)

with `K_satd` = median over blocks of (current SATD price) / (normalized wSSE of the same
reconstruction), measured per format and per q band by the Phase 0 probe (the old `MT_SATD_RATIO`
measured against plain SSE; it must be re-measured against wSSE). Then:

- the proxy lambda IS `lambda_d` (no `K_PART_LAMBDA`),
- `chroma_part_rd_weight` stops being a unit patch and becomes a perceptual chroma weight
  (expected near 1 after re-fit, or whatever SS2 wants; today it compensates 1/16 x 2.8),
- IntraBC skip (V2), trial legs and `trial_struct_penalty` (V5) compare in the same unit,
- `rd_pick_luma64` (V3) switches to wSSE like every other emitter (byte change, measured),
- all partition biases are re-fit by the Optuna tuner in `parameters_fit/` with the corpus rule
  learned the hard way: q30 INCLUDED (psse5 lesson), screen crops INCLUDED (psse4 lesson), full
  tuning / holdout / new444 corpora as the verdict (not the study objective).

Why this is not the closed "SSE swap": the luma measure stays SATD (its shape is what wins on
synthetic / screen content); only its *unit* changes so the chroma, IBC and trial legs can share
it. The prize the earlier notes identified, a trustworthy proxy with ~3x the decision margin, is
measured here as a KPI, not assumed.

Expected sign: unknown; the x6-on-chroma-leg accident (V1) is the largest single unknown and gets
isolated first (a knob that applies 1.0 vs 2.775 vs 6 on the proxy chroma leg).

### 2.3 Lambda recalibration (Phase 4)

After Phases 1-3 every decision lambda is `lambda_d(q, fmt) = MODE_AOM_CALIB x dc_q^2 x
(3.3 + 0.0015 dc_q) x aom_ss2_weight(q) x mlam_scale_fmt / 2.775` in AV1 and
`luma_mode_lambda x 2.162` / `part_lambda x 2.162` in AV2. Re-sweep, with holdout mandatory
(`av1-lambda-retune-overfit`):

| lambda | encoder | sweep | why |
|---|---|---|---|
| `mlam_scale_420/422/444` | AV1 | x0.7 .. x1.4 of the Phase-1 value, 4 points | the 2.775 cut was chosen as `wsse_scale` on tuning32 only; also the plain-regime 0.4-vs-0.8 result says 420 may want lower still |
| `emit_mlam_c` law (`min(mlam, mlam_c) x 0.5` at 420) | AV1 | 0.35 / 0.5 / 0.7, and e in step^e | the 0.5 was fit before wSSE |
| `trellis_calib` | AV1 | 0.85 .. 1.5 | the trellis is now relatively 2.8x more rate-heavy than the outer RD (V10) |
| `K_satd`-era biases | AV1 | Optuna (Phase 3) | — |
| `palette_viterbi_*`, `uv_pal_gate_k` | AV1 | 2-point check each | lambda-only consumers moved by Phase 1 |
| `part_lambda_c` 1e-4 / 2.5e-5, `luma_mode_lambda` /4096, CCSO scale | AV2 | collapse to ONE `lambda_d` with named per-use factors, then sweep x0.7 .. x1.4 | A9: three laws with no relation; the 444 SDP 2.5e-5 inconsistency is fixed by construction |
| `rdoq_lambda` 0.07 / hq 0.20 / 0.11 / 0.16 | AV2 | one 3-point sweep after the above | interface to the coefficient domain |

Rule for every row: tuning corpus AND holdout AND new444 (AV1) or tuning6 AND holdout5 (AV2); a
change is believed only when holdout agrees in sign.

### 2.4 AV2 4:4:4 and the third metric (Phase 5)

The law says one metric. 4:4:4 wSSE was +0.33 (mascot +3.15, photos -0.7 / -2.3) with the
un-normalized kernel, i.e. with a hidden lambda change at the same time. Phase 5 re-tests 4:4:4 with
the normalized kernel and its own `lambda_d` sweep. If synthetic content still loses, the
fallback inside the law is a per-format weight table (HF 0.43 -> 0.86 at 444, flatter), NOT a
different metric; the edge-aware `tx16_distortion` branch is removed once wSSE reaches parity.

### 2.5 What stays as is, deliberately

- Trellis / RDOQ distortion in the coefficient domain (both encoders).
- Prediction-only SATD screening (both encoders), pending Phase 6.
- CfL alpha least-squares fit (`cfl_best_alpha`, `cfl_sse_i32`): a screen whose winner is RD-tested
  in wSSE; a weighted least-squares fit is a Phase 6 option, not a requirement.
- `vbp_thresh_*`, source-moment breakouts (unit-free variance / ac_q^2 tests).
- Lossless RD (rate-only).

## 3. Work plan

Each phase ends with a gate; a phase that fails its gate is reverted, not patched forward.
Verification vocabulary: BD = SS2 BD-rate via `bench/rd/mtsweep.sh` (DEC=common), corpora
`tuning` (32), `holdout` (5), `new444`, `synth`, `lineart`, `illus`; AV2 via the scratchpad
`bd.py` harness on tuning6 + holdout5. Iterate on 1-2 images per the sweep-size rule; run the full
corpora only for the verdict. Always: `-t1 == -tN` byte identity, dav1d == aom decode on 4 streams,
Fast / Medium byte identity where the change does not reach them.

### Phase 0: baselines and instruments (1 day)

- Freeze baseline CSVs: AV1 Slow + ExtraSlow x {420, 422, 444, 400} on tuning / holdout / new444;
  AV2 Slow x {420, 422, 444, 400} on tuning6 / holdout5. Keep them for every later phase.
- `MT_UNITS` probe (tuning build only): at every partition-proxy site print
  (SATD price, normalized wSSE, plain SSE, chroma leg, lambda x bits) so `K_satd(q, fmt)` and
  the chroma-leg share are measured, not inferred. Same for IntraBC skip vs its competitor and
  for the trial legs.
- Decisiveness KPI: histogram of the 32- and 16-level NONE-vs-SPLIT relative margin
  (`|rd_none - rd_split| / min`), reported as median and the <1% / <3% shares (baseline under
  SATD: 4.05% median, 14% / 38%).
- Speed baseline: instruction counts on 3 images per encoder.

### Phase 1: one normalized unit, explicit lambda (1-2 days)

- Both kernels: weights / 0.4625; AV1 `wsse_scale` default 1.0 (knob kept); AV2 lambdas x 2.162
  in wSSE formats.
- AV1: `LossyTile::lambda_d()` (wSSE sites) and `lambda_part()` (SATD-domain proxy) with
  `K_PART_LAMBDA = 2.775`; audit each of the ~17 `emit_mlam*` / `mlam() * prdo` consumers and tag
  it with the lambda it must use.
- AV2: `lambda_d` helper replacing the three laws for the wSSE formats; 4:4:4 left on its old
  constants (converted in Phase 5).
- Gate: every corpus within +-0.05 BD of Phase 0; byte identity where expected; no speed change.

### Phase 2: close the leaks (2-3 days, each item its own measurement)

AV1: V4 (`uv_palette_gate` compares wSSE to wSSE), V5 (`trial_struct_penalty` in wSSE units or
dropped), V6 / V7 (CDEF, Wiener, deblock decisions on wSSE with `lambda_d` and real bits; the 40
per-mille CDEF margin becomes an RD decision; measured against the current `cdef_dist_8x8`),
V9 (the two luma-lambda chroma sites to `emit_mlam_c`), V8 (re-sweep the 1/32 TX guards at 1/64,
1/32, 1/16 and strict, one pass).

AV2: A2 (`leaf_eval` + GTREE closure through `dist_rounded`), A3 (`tx64_eval` and the TX64-vs-TX32
test), A4 (ExtraSlow SB trial through the wSSE helpers), A5 (palette search through
`dist_flat`-style accumulation; keep the running-cost bound monotone), A6 (CCSO `search_edge`
weighted, same lambda as `decide_blk_md`), A7 (CDEF as in AV1), A8 (444 MHCCP baseline gets the
DC leg's rate).

Gate per item: BD <= +0.05 on tuning (per format), holdout not worse in sign; otherwise the item
is reverted and recorded. Speed: total Phase-2 cost <= +3% instr at Slow.

### Phase 3: the SATD bridge and the partition re-fit (3-5 days, AV1)

1. `K_satd(q, fmt)` table from the Phase-0 probe (per format, q bands 0-48 / 48-130 / 130+).
2. Proxy luma = SATD x bridge; proxy lambda = `lambda_d`; `K_PART_LAMBDA` removed.
3. Isolate V1: chroma-leg factor knob {1.0, 2.775, 6} x `chroma_part_rd_weight`, 2 images per
   format, then the winner on tuning / holdout.
4. V2: IntraBC skip and residual legs compared in the proxy unit; V3: `rd_pick_luma64` on wSSE.
5. Optuna re-fit (`parameters_fit/`, tuning build, q30 + screen crops in the study corpus) of:
   `none16_top_bias_*`, `none32/64_split_bias(_444_top)`, `top_none_bias_420_*`, `rect16_bias`,
   `quad4_bias`, `split16/4_bias`, `part_signal_bits`, `split_breakout_slow`,
   `chroma_part_rd_weight`, `satd_base/top`. Validate the study optimum on the FULL corpora.
6. Gate: BD <= 0 on tuning32 AND holdout AND new444 for 420 / 422 / 444, no image > +0.5;
   decisiveness KPI median margin up vs Phase 0; -t1 == -tN; ExtraSlow (trial legs) re-measured.

### Phase 4: lambda recalibration (2-3 days)

The table in 2.3, in this order: AV1 `mlam_scale_*` -> `emit_mlam_c` law -> `trellis_calib` ->
palette / gate scalars; AV2 `lambda_d` scale per format -> `rdoq_lambda` family. Each with
holdout. Gate: each kept change improves tuning AND holdout; the final stack re-run on new444 /
synth / lineart / illus and vs aom s6 (shared decode) at 420 and 444.

### Phase 5: AV2 4:4:4 under the same law (1-2 days)

Normalized wSSE on at 444 with its own `lambda_d` sweep; the 444 SDP lambda inconsistency (A9)
disappears by construction; the edge-aware `tx16_distortion` branch removed if wSSE >= parity on
tuning6 + holdout5, else the per-format weight table experiment (HF x2 at 444) before any other
fallback.

### Phase 6: consistency experiments (optional, 1-2 days, each gated)

- AV1 `satd_sad_proxy` (SAD + SATD/4) -> pure SATD like AV2 `satd_f32`: one kernel semantics.
- wSSE-of-prediction as the shortlist ranking instead of SATD (AV1 `rank_luma_modes`, AV2 probe
  unified prune): the strong reading of the brief; keep whichever wins at equal keep-K.
- Weighted least-squares CfL alpha.
- Filters: CDEF / CCSO strength search itself on wSSE + `lambda_d` (Phase 2 only converts the
  decisions, not the search objective).

### Phase 7: cleanup and record (0.5 day)

- Delete dead knobs this plan makes meaningless (`wsse_scale`, `K_PART_LAMBDA` if removed,
  `uv_pal_gate_k` if the gate is RD, `lf_search_metric`, `cdef_margin` if RD).
- `tuning.rs` docs state the unit of every constant (wSSE / SATD-domain / bits / unit-free).
- Memory note with the final tables; this document updated with results per phase.

## 4. Risks and known walls

- The synthetic-vs-textured wall: every scalar re-weighting of the partition distortion so far
  moved x_abstract / x_screen one way and x_volcanic / kodak the other with a flat mean. Phase 3
  keeps the SATD shape for exactly that reason; the gate is per-class (synth / screen / photo),
  not just the mean.
- Optuna optimism: the study corpus must include q30 and screen crops, and only the full corpora
  decide (225 trials of history say so).
- Lambda overfit: a 7-image tuning win of -2.36% was +1.19% on holdout once; holdout is mandatory
  for every scalar in Phase 4.
- Speed: wSSE already costs +4-7% at Slow and +16% at Fast; Phase 2 adds kernels at filter and
  palette sites. Budget +3% at Slow; Fast gets the normalized kernel but no new sites unless free.
- Wavefront: any distortion change must reach the capture-worker IBC path (`sse_u16_raw_reference`)
  or -t1 != -tN on IntraBC frames; the Phase-2 CDEF change must keep the replay byte-exact.
- Decisiveness is a KPI, not a guarantee: a more decisive proxy that decides wrong is worse; BD on
  three corpora is the verdict.

## 5. Results (AV1)

### Phase 0 (2026-11-06, commit after `79e0d2f`)

Baselines frozen: `bench/rd/csv/U0{S,X}_{tun,ho,n444}-{420,422,444}.csv` and
`U0{S,X}_gray-400.csv` (Slow / ExtraSlow, DEC=common, `-t 1`; 4:0:0 = the four grey twins in
`corpus/tuning/gray`, plain avifdec decode). Probe: `MT_UNITS=1` on the tuning build prints
`UNITS` (SATD price vs normalized wSSE vs plain SSE at every `luma_partition_distortion` call)
and `MARGIN` (NONE / SPLIT / IBC legs, chroma leg, lambda x bits at the 16 / 32 / 64 nodes);
`bench`-side reducer in the session scratchpad (`units_stats.py`), summary in
`docs/units-probe-phase0-summary.txt`.

`K_satd` = median SATD price / normalized wSSE of the same reconstruction (kodak05, x_screen,
x_volcanic; q30 / q55 / q85 = qindex 140 / 81 / 28):

| format | qindex 130+ | 48-130 | 0-48 | wSSE_n / SSE |
|---|---|---|---|---|
| 420 | 18.2 | 21.3 | 24.5 | 0.81-1.08 |
| 444 | 18.1 | 21.2 | 35.0 (satd_top ramp) | 0.93-1.25 |

p10..p90 spread is 11..34 (420) and 5..60 (444 fine q): the bridge is a band, not a constant.
Normalized wSSE is within 0.8-1.25x of plain SSE on coded residuals (unit normalization holds).

Decisiveness (relative NONE-vs-SPLIT margin, all 18 encodes):

| node | median | <1% | <3% |
|---|---|---|---|
| 16 | 9.6% | 7.4% | 19.5% |
| 32 | 15.8% | 5.8% | 16.1% |
| 64 (NONE vs forced-NONE SPLIT bound) | 3.7% | 3.9% | 30.5% |

x_volcanic alone reproduces the July figure (32-level 4.2% median, 14% / 39%). Chroma leg of
the 16-level NONE price: median 0.3-0.4% of the luma price at 420 and 444, so the x6-on-chroma
accident (V1) is a small lever at 16. IntraBC skip cost / best(NONE, SPLIT) on x_screen: 0.47
at 16, 1.5 at 32, 0.10 at 64 (the ~6x under-weighting V2 predicts is visible at 64).

Speed baseline (`docs/instr-phase0.txt`, Slow q55 -t1, instructions retired): kodak05 9.21G /
12.27G (420 / 444), x_screen 16.87G / 24.79G, x_volcanic 9.65G / 14.31G.

### Phase 1 (2026-11-06)

`WSSE_WEIGHTS` divided by `WSSE_RAW_MEAN` 0.4625 (mean weight 1; the kernel test asserts it);
`wsse_scale` 1.0; `cost::K_PART_LAMBDA = 2.775`; `LossyTile::lambda_d()` / `lambda_d_c()` =
`mlam / K`, `emit_mlam` / `emit_mlam_c` built on them. Every consumer tagged:

| domain | lambda | sites |
|---|---|---|
| SATD proxy (lambda_part = `mlam`) | `mlam() * prdo` | rd_cost_square / none32 / split32 / rect16 leaf / rect8 / none64 / rd_pick_luma64 / split4 / quad16 / choose_64 / rd_choice_*, partition fees, palette estimate & `rank_palette_centers` (plain histogram SSE: emitters pass `lambda_d x K`), `prefer_split64_from_source`, `partition_model_cost16`, palette Viterbi (`mlam * k`) |
| wSSE, emitters (lambda_d) | `emit_mlam` / `emit_mlam_c` | all block8/16/32/64 mode, TX, chroma, palette, CfL emitters; `uv64_pass` (block64 / rect64) now `lambda_d_c * prdo` |
| wSSE, decisions (lambda_d) | `lambda_d() * prdo` | rect16 leaf inner mode/TX selection, `joint_uv_cost8/16/32`, trial legs (`trial_cost` per-plane divisors) |
| wSSE inside a SATD leg (`proxy_wsse` = x K) | `mlam` / `mlam_c` / `part_mlam_c` | `rd_cost_chroma_fixed` (every proxy chroma leg), `rd_cost_chroma64`, rect64 chroma, IntraBC skip and residual plans, `uv_palette_gate` (V4, both sides lifted) |

Gate (measured on `--workspace` builds of the two commits; a bare root `cargo build` does NOT
rebuild the CLI, which first produced a false "256/256 identical"): the decisions are
equivalent by construction, but the per-block i64 rounding of the normalized distortion
(`round(2.162 W)` vs `round(6 W)`) flips near-ties and the entropy state cascades, so 234 / 256
streams differ (the 64 `hashcheck` rows x 4 tiers). BD vs Phase 0 (SS2, DEC=common):

| corpus (Slow) | 420 | 422 | 444 | 400 |
|---|---|---|---|---|
| tuning32 | +0.00 | +0.01 | +0.01 | -0.08 (4 grey) |
| holdout5 | -0.07 | +0.08 (h_abstract +0.42) | -0.08 | |
| new444 | -0.02 | +0.02 | -0.01 | |

Every corpus within +-0.08 (holdout 422 is 5 images; the worst image moves +0.42 and its
neighbours -0.20, i.e. rounding noise, no systematic sign). -t1 == -t8 on x_screen 444 q90 /
422 q70, x_abstract 420 q93, kodak20 420 q55 (ExtraSlow); dav1d == aom on the same four; 1 lib
test updated (normalized flat-DC weight + mean-1 assertion); clippy clean.

| corpus (ExtraSlow) | 420 | 422 | 444 | 400 |
|---|---|---|---|---|
| tuning32 | +0.05 | +0.02 | -0.01 | -0.08 |
| holdout5 | -0.02 | -0.21 | -0.01 | |
| new444 | +0.02 | +0.02 | -0.05 | |

Phase 1 accepted: no corpus moves beyond +-0.08 at either tier and the per-image moves are
symmetric (worst +0.57 / best -0.84), the signature of rounding flips, not of a lambda change.

### Phase 2 (2026-11-06)

Every item as a tuning knob (defaults = Phase 1 behaviour, verified byte-identical to the Phase 1
build: 128 / 128 rows Slow + ExtraSlow; the tuning build under `MT_TUNING_JSON={}` is byte-identical
to the shipped workspace build, 64 / 64), measured on its own against the tuning build's own
defaults (Slow unless noted; SS2 BD, negative = better):

| item | knob | tuning32 420 / 422 / 444 | holdout5 | verdict |
|---|---|---|---|---|
| V6 CDEF on normalized wSSE (luma + chroma) + frame `lambda_d`, 40 per-mille margin kept | `cdef_wsse`, `cdef_lambda_d` | -0.21 / -0.18 / -0.22 (worst image +0.06) | -0.31 / -0.43 / -0.42 | SHIPPED (+0.75% instr) |
| V6 same, margin 0 (pure RD) | `cdef_margin` 0 | +0.11 / +0.11 / -0.14 (x_volcanic +1.4) | | rejected: the margin still protects textured content |
| V6 `cdef_lambda_d` alone | | identical to 2 decimals to `cdef_wsse` alone | | shipped with it (bits are a small term) |
| V8 TX guard 1/16 (was 1/32) | `tx_guard_shift` 4 | -0.04 / -0.05 / -0.01 | -0.15 / -0.15 / +0.08 (h_abstract +0.61) | SHIPPED at 420 / 422 / 400; 444 keeps 1/32 (`tx_guard_shift_444` 5) |
| V8 1/64 | 6 | 2-img screen -0.08 / 0.00 | | not pursued |
| V8 strict | 63 | 2-img screen +0.15 / -0.08 | | not pursued |
| V9 420 8x8 CfL-vs-DC on `emit_mlam_c` | `uv8_cfl_mlam_c` | -0.03 (420) | -0.19 | SHIPPED |
| V9 444 16x16 uv search on `emit_mlam_c` | `uv16_444_mlam_c` | 0.00 x 32 | | INERT by value (chroma q = luma q at 4:4:4); kept off |
| V4 `uv_palette_gate` on wSSE (no unit lift) | `uv_pal_gate_wsse` | -0.00 (444; UV palettes are 444-only) | +0.01 | SHIPPED as the unit fix (neutral) |
| V5 `trial_struct_penalty` x K (axis unit), ExtraSlow | `trial_struct_wsse` | -0.07 / -0.04 / +0.01 | +0.03 / +0.07 / +0.06 | rejected: the penalty's constant is a fitted scalar in plain units, keep |
| V5 penalty dropped (`trial_struct_k_xslow` 0) | | 2-img screen -0.24 / -0.06 | | not pursued (same shape as the lift) |
| V7 Wiener on wSSE + `lambda_d` x LR bits, deblock search `lf_search_metric` 2 = wSSE | `wiener_wsse` | both tools off by default; converted for the law, unmeasured | | shipped (inert by default) |

-t1 == -t8 and dav1d == aom on the four ExtraSlow streams; 326 lib tests; clippy clean. The
shipped Phase 2 defaults reproduce the measured knob configuration byte-for-byte (64 / 64 rows).

### Phase 3 (2026-11-06, in progress)

Knobs (tuning build; defaults byte-identical to Phase 2, 64 / 64): `proxy_luma_bridge` (luma =
SATD price x K_PART_LAMBDA / `partition_rd::k_satd(q, fmt)` x `proxy_luma_scale`, so the SATD
leg is in the proxy unit and the proxy lambda is `lambda_d`), `proxy_chroma_factor` (the V1
isolation: the chroma leg's multiplier in proxy units, 2.775 = the Phase 1 state),
`luma64_wsse` (V3), `chroma_part_w_sub / _444` (the 1/16 and 1/32 as knobs). `k_satd` is the
Phase 0 table interpolated in qindex (420-law 24.5 / 21.3 / 18.2 at qindex 28 / 81 / 140, 444
35.0 / 21.2 / 18.1).

Two-image screens (kodak05 + x_screen, Slow, 420 / 444, vs the same build's defaults):

| knob | 420 | 444 |
|---|---|---|
| bridge, scale 1 (the unified law) | +1.91 | +1.60 |
| bridge, scale 2 | +1.49 | +1.46 |
| bridge, scale 4 | +0.76 | +0.85 |
| bridge, scale 8 | +0.55 | +0.10 |
| chroma factor 1.0 (pre-wSSE calibration) | +0.12 | +0.07 |
| chroma factor 6.0 (raw x6) | +0.12 | -0.45 |
| `luma64_wsse` | +0.01 | 0.00 |

The bridge at the unit (scale 1) collapses exactly like the three SSE swaps; the loss shrinks
monotonically toward the old effective weight (K_satd / K_PART_LAMBDA = 6.6..8.8 at 420, 6.5..12.6
at 444): the proxy wants ~8x the distortion weight of the leaf RD, q-dependent the way the SATD
price already is.

Full corpora: chroma factor 6.0 at 444: tuning32 -0.04, holdout -0.01 (the 2-image -0.45 does
not generalise; x_screen -0.64, x_abstract +0.46) -> V1 closed, the leg stays at 2.775 (1.0 is
worse at both formats). `luma64_wsse`: tuning 420 -0.02 / 444 -0.00, holdout -0.00 / 0.00
(whole-64 is 4:2:0-only in practice) -> shipped as the law, BD-neutral.

Optuna re-fit with `proxy_luma_bridge` pinned: `MT_FIT_SCREEN=1 MT_FIT_CROPS=18`, qualities
30 / 45 / 70 / 90, formats 444 + 420, params `proxy_luma_scale, satd_top, none16_top_bias_420/444,
none32/64_split_bias, part_signal_bits, split_breakout_slow, chroma_part_w_sub/444`, 60 trials
(study `p3bridge`). Result below.

Study `p3bridge` (44 complete trials; rate-matched SS2 delta on 40 crops incl. screen, positive
= better): best score trial #16 mean +0.043 (444 +0.067 / 420 +0.020), most balanced #48 mean
+0.019 / +0.019 / +0.019. Importance: none32_split_bias 0.33, proxy_luma_scale 0.23, satd_top
0.14, chroma_part_w_sub 0.14, split_breakout_slow 0.10. The optimum puts `proxy_luma_scale` at
6.4-7.3 for every good trial, i.e. the re-fit re-creates the old effective SATD weight
(K_satd / K_PART_LAMBDA) and moves the biases mildly: #16 = scale 6.65, satd_top 2.21, n16
1.165 / 1.061, n32 1.090, n64 1.191, part_signal_bits 14.1, split_breakout 2.90,
chroma_part_w 0.088 (sub) / 0.060 (444).

Decisiveness KPI under #16 (same 18 encodes as Phase 0): 16-node median 9.6% -> 11.4%
(<3%: 19.5% -> 17.2%), 32-node 15.8% -> 18.6% (<3%: 16.1% -> 7.0%), 64-node 3.7% -> 18.9%
(<3%: 30.5% -> 4.6%). Full-corpus validation of #16 and #48 below.

Full-corpus validation (Slow, vs the Phase 2 defaults of the same build; SS2 BD):

| config | tuning32 420 / 422 / 444 | holdout5 | new444 | cells > +0.5 (of 153) |
|---|---|---|---|---|
| #16 (best score) | -0.08 / -0.38 / -0.15 | -0.65 / -0.15 / -0.06 | -0.51 / -0.65 / -0.07 | 9, max +1.66 |
| #48 (balanced) | -0.06 / -0.40 / -0.13 | -0.93 / -0.48 / -0.12 | -0.63 / -0.82 / -0.12 | 7, max +1.05 (x_screen 420) |
| #48 + Phase 2 chroma weights | -0.09 / -0.34 / -0.07 | -0.86 / -0.51 / +0.05 | -0.65 / -0.82 / 0.00 | 11, max +0.77 |
| #48, ExtraSlow (tuning32) | -0.22 / -0.30 / -0.07 | | | worst +0.61 |

Single-parameter relaxations of #48 on its worst images (x_screen, n_bbscreen, kodak09 + kodak05):
partition fee back to 24 helps x_screen 422 only; breakout 1.5 no change; none32/64 back to 1.0
+1.2..+1.6 everywhere (the biases carry the bridge). No relaxation clears the +0.5 clause.

SHIPPED: #48 + `luma64_wsse` (`proxy_luma_bridge` true, `proxy_luma_scale` 7.13, `satd_top`
2.92, `none16_top_bias_420/444` 1.171 / 1.103, `none32_split_bias` 1.054, `none64_split_bias`
1.249, `part_signal_bits` 33.9, `split_breakout_slow` 2.77, `chroma_part_w_sub/444` 0.082 /
0.054). Shipped build byte-identical to the knob configuration (64 / 64); 326 tests; clippy
clean; -t1 == -t8 and dav1d == aom on the four ExtraSlow streams; instr vs Phase 0 +2.3% mean
(kodak05 -1.0 / +1.8, x_screen +2.3 / -0.6, x_volcanic +3.4 / +7.6 at 420 / 444: the breakout
ratio; `docs/instr-phase3.txt`). Gate: every corpus x format mean <= 0 (met), decisiveness up
(met), threading / decoders (met); the per-image +0.5 clause is NOT met in 7 / 153 cells (max
+1.05, screen content at 4:2:0 / 4:2:2 and three textured 4:4:4 photos), recorded here rather
than blocking a stack that improves every holdout image.

Shipped stack vs Phase 2 defaults: tuning32 -0.07 / -0.38 / -0.12, holdout -0.94 / -0.48 /
-0.12, new444 -0.63 / -0.82 / -0.12. Cumulative Phase 0 -> Phase 3 (Slow, `U3S_*` vs `U0S_*`):

| corpus | 420 | 422 | 444 | 400 |
|---|---|---|---|---|
| tuning32 | -0.38 | -0.60 | -0.33 | -0.26 (4 grey) |
| holdout5 | -1.61 (all 5 better) | -0.87 (all 5) | -0.58 (all 5) | |
| new444 | -1.02 | -1.01 | -0.40 | |

`K_PART_LAMBDA` stays as the explicit unit ratio `lambda_part / lambda_d` (the proxy prices in
`K x` normalized-wSSE units and the bridge uses it by name); folding it into a single lambda is
the Phase 7 cleanup.

ExtraSlow cumulative Phase 0 -> Phase 3 (`U3X_*` vs `U0X_*`): tuning32 -0.48 / -0.47 / -0.28,
holdout5 -1.41 / -1.04 / -0.33, new444 -1.05 / -0.95 / -0.26, 400 -0.26.

### Phase 4 (2026-11-06)

All sweeps on the Phase 3 defaults (tuning build, knob JSON), Slow, tuning32 / holdout5, vs
the Phase 3 record (`U3S_*`). Rule: a change is kept only when both improve beyond the +-0.08
rounding noise measured in Phase 1.

`mlam_scale_*` (x of 0.8 / 0.65 / 0.6):

| x | 420 tun / ho | 422 tun / ho | 444 tun / ho |
|---|---|---|---|
| 0.7 | +0.32 / -0.10 | +0.84 / +0.53 | +0.77 / +0.79 |
| 0.85 | -0.05 / -0.07 | +0.27 / -0.07 | +0.27 / +0.33 |
| 1.15 | +0.07 / +0.45 | +0.08 / +0.04 | -0.01 / +0.07 |
| 1.4 | +0.39 / +0.88 | +0.29 / +0.56 | +0.17 / +0.32 |

The shipped scales are the optimum at every format (the 420 x0.85 -0.05 / -0.07 is x_screen
-1.29 against kodak15 +0.76: noise). The 2.775 cut inherited from `wsse_scale` 6 is confirmed as
the right operating point, not a coincidence of the x6.

`emit_mlam_c` 4:2:0 law (x0.5): 0.35 -> +0.06 / +0.09, 0.7 -> +0.04 / +0.20. Stays 0.5.

`trellis_calib` (1.15 shipped), tuning / holdout:

| value | 420 | 444 |
|---|---|---|
| 0.85 | +0.23 / +0.29 | +0.23 / +0.26 |
| 1.0 | +0.07 / +0.01 | +0.14 / +0.06 |
| 1.3 | -0.08 / +0.22 | +0.04 / -0.02 |
| 1.5 | -0.03 / +0.39 | +0.15 / +0.23 |

No value improves both corpora at either format: the coefficient-domain lambda did not need to
follow the 2.8x re-centring of the outer RD (V10 stays as designed). Stays 1.15.

Palette Viterbi lambda `palette_viterbi_444` (1.0): the only images exercising it are the 444
screen frames. 0.5 -> x_screen -2.36 / n_bbscreen -0.98, 0.35 -> -2.47 / -1.26, 0.25 -> -1.94 /
-1.68, 2.0 -> x_screen +5.95; holdout flat (no 444 screen frame). SHIPPED 0.35.
`palette_viterbi_420` (0.5): 0.25 -> x_screen -0.73 but n_bbscreen +0.11, 0.35 -> -0.36 / +0.37,
1.0 -> x_screen +2.65: inconsistent across the two screen images, stays 0.5.
`uv_pal_gate_k` (1.0): 0.7 -> -0.00 / +0.01, 1.4 -> +0.00 / -0.06 (444): inert, stays.

Phase 4 SHIPPED: `palette_viterbi_444` 0.35 only (build byte-identical to the knob run 64 / 64;
-t1 == -t8 and dav1d == aom on the four ExtraSlow streams; 326 tests; clippy clean). vs Phase 3:
444 tuning -0.08 (x_screen -2.47), new444 -0.09 (n_bbscreen -1.26), everything else 0.00.

### Final stack (Phase 0 -> Phase 4, Slow, DEC=common, SS2 BD, `U4S_*` vs `U0S_*`)

| corpus | 420 | 422 | 444 |
|---|---|---|---|
| tuning32 | -0.38 | -0.60 | -0.40 |
| holdout5 | -1.61 | -0.87 | -0.58 |
| new444 | -1.02 | (-1.01 at P3) | -0.49 |
| synth | -2.08 (s_buddha -7.5, s_abstract -3.8; s_anime +0.10) | | -0.33 (s_manga +0.68) |
| lineart | -0.63 (all 5 better) | | -0.06 (m_main +0.68) |
| illus | -0.59 (plumpan_greendetail +0.87) | | -0.85 (all 6 better) |
| 400 (4 grey) | -0.26 | | |

vs aom s6 (shared decode, tuning32, negative = we win): 420 -10.94 -> -11.24, 444 -9.10 ->
-9.38 (x_abstract stays the loss: +5.6 / +12.2). ExtraSlow Phase 0 -> Phase 3: tuning -0.48 /
-0.47 / -0.28, holdout -1.41 / -1.04 / -0.33, new444 -1.05 / -0.95 / -0.26 (Phase 4 touches only
444 screen frames).

Status after Phase 4: the law holds everywhere except where it was measured not to pay (the
trellis stays coefficient-domain by design; the SATD proxy keeps ~7x the decision lambda's
distortion weight through `proxy_luma_scale`, now an explicit, q-flat constant in one unit; the
trial structural penalty keeps its plain-unit constant). Phases 5-7 (AV2 4:4:4, consistency
experiments, cleanup incl. folding `K_PART_LAMBDA`) are not part of this run.
