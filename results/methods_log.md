# Methods log

Dated record of method decisions for the TOM shape-flow DEM study. Newest entries at the bottom.
Entries before 2026-11-05 10:00 were back-filled from the session record when this log was started.

Release note (2026-11-08): this is the study's working log, unedited except for this note and the study name above.
Paths are relative to the repository. In this repository the grid templates and shapes_summary.csv are in
shapes/repose_grid/ (results/repose_grid/templates/ during the study). Files the log mentions that are not in git
(slump frames, raw heap.csv, run logs, guard.csv, progress.csv, gpu_clock.csv, queue_status.json, videos) are in the
data bundle (data/large/*.zip, Zenodo). Entries about chute and hopper work refer to later phases of the study that
are not part of this release. The L0 templates (the calibration anchor) are in shapes/templates/; the L1-L5
templates, the L0-L5 calibration-level results and results/repose_archive_flatfloor/ are not included.

## 2026-11-04

- Literature estimates set in config.json (no lab data yet): walls.mu_particle_wall 0.35 (smooth 6061 vs 2 mm
  angular grains, ~19 deg; brackets 0.25 and 0.50; replace with tilt test); repose.target_angle_deg 40.
- Chute scoring guard added: a run only scores as flowing if the delivered feed over the score window is at least
  chute.feed_fraction_min (0.95) of the target feed rate.
- Repose test changed to a rough floor: one layer of same-level grains settled and frozen over the whole floor, heap
  heights measured above it; column 0.08 m wide x 0.04 m tall (aspect 0.5). The flat-floor result (L0, mu 0.4,
  18.5 deg) is archived in results/repose_archive_flatfloor/.
- particles.E_Pa 1e7. Time step: 1e-5 s diverged on L0 (three DEME contact-count aborts); kept at 5e-6 s.
- Friction alone could not reach 40 deg on L0 (mu 0.6: 28.35 deg; mu 0.8: 27.60 deg). particles.mu fixed at 0.6 and
  the angle calibrated with rolling resistance instead; Crr pairs set explicitly (grain-grain and grain-6061 = Crr,
  grain-slice = 0).
- L0 Crr sweep (seed 1): 0 -> 28.35, 0.2 -> 38.12, 0.25 -> 40.35, 0.3 -> 42.55 deg. Crr set to 0.25.

## 2026-11-05

- Crr is the same for every level (no per-level tuning); deviations from 42 - 17R are reported as model error.
- Repose test times scale with gravity by sqrt(9.81 / g) (fill, settle and slump limits; speed thresholds divided by
  the same factor). Earth runs are unchanged (factor 1).
- Repose shape grid: roundness 0.13 / 0.45 / 0.80 / 1.0 x aspect (b/a = c/a) 0.6 / 0.8 / 1.0, 8 templates per shape,
  same volumes as the shipped levels. Templates accepted within +-0.03 of the target aspect; measured roundness mean
  within +-0.05 of target. Bump tolerance 3.5% of d for roundness 0.13 (GPU memory), 2% otherwise. Roundness 1.0 at
  aspect < 1 is a capsule.
- 3-seed L0 mean 41.54 ± 0.67 deg (SE) at Crr 0.25; Crr set to 0.22 by interpolation to 40 deg. Applies to the repose
  grid and the lunar subset. The dt 2.5e-6 check keeps Crr 0.25 (it tests the time step, not Crr).
- Grid calibration check (roundness 0.13 / aspect 0.8): two-seed mean must be within ±2.5 deg of 40; also reported
  against 42 - 17R at its measured roundness (0.179 -> 39.0 deg).
- Time-step check (L0, mu 0.6, Crr 0.25, seed 1): dt 2.5e-6 gave 42.08 deg, against the dt 5e-6 3-seed mean of
  41.54 ± 0.67 deg (SE) — within one SE, so no evidence of a time-step effect. The repose grid stays at dt 5e-6.
  L0 seeds 2 and 3 at dt 2.5e-6 (Crr 0.25) are queued after the lunar subset for the convergence table.
- Time-step convergence (L0, mu 0.6, Crr 0.25, rough-floor slump test, 3 seeds each):
  dt 5e-6: 40.35, 41.61, 42.66 -> 41.54 ± 0.67 deg (SE); dt 2.5e-6: 42.08, 39.54, 42.78 -> 41.47 ± 0.98 deg (SE).
  Difference 0.07 deg, well inside one SE: dt 5e-6 is converged for the repose test.
- Grid check, roundness 0.13 / aspect 0.8 (R013_A08) at Crr 0.22: seeds 37.78 and 36.05 deg, mean 36.92 deg, -3.08
  against 40 (outside ±2.5), so the grid paused. Decision: treat R013_A08 as a grid result, not a calibration
  failure, and resume unchanged at Crr 0.22. Crr was calibrated on the old L0 (c/a 0.55-0.69, R 0.132, ~5% bumps);
  R013_A08 is less flat (c/a 0.78), rounder (0.179) and smoother (3.5% bumps), all of which lower the angle. It sits
  2.0 deg below 42 - 17R at its measured roundness (38.96 deg), inside ±2.5 against that reference.
- Calibration anchor at the grid's Crr: the shipped L0 templates at Crr 0.22, dt 5e-6, seeds 1 and 2 (video on
  seed 1), run at the end of the Earth grid, before the lunar subset.
- Note for when chute/hopper work resumes: for L0, dt 2.5e-6 was faster per run than 5e-6 (5.5 vs 13.9 ms/step in
  the slump phase; 72 vs 99 min per repose run). DEME's adaptive contact margin scales with velocity x dt, and with
  ~36 small spheres per L0 grain the candidate-contact count drops sharply with the smaller margin. Not yet checked
  for the hopper or chute, or for few-sphere shapes (where the doubled step count likely dominates).

## 2026-11-06

- Fallback rule (decided while R013_A06 seed 1 ran at 11.5 of 12.3 GB GPU memory, ~40 ms/step): if any grid run
  aborts on memory or a DEME error, the whole shape (both seeds; the single seed for a lunar run) is rerun at
  dt 2.5e-6, justified by the L0 time-step convergence result (41.54 ± 0.67 deg SE at dt 5e-6 vs 41.47 ± 0.98 at
  dt 2.5e-6). Reruns go to <shape>_<gravity>_dt2.5e-06_s<seed>; the grid table (time_step_s column) and media use
  them for that shape. Each fallback adds its own entry below.
- Run-time prediction for R045_A06, logged 2026-11-06 22:46 CST while seed 1 was still in its
  fill/settle (5 h 44 min elapsed, no slump frames yet): 11-point fit over the mean wall time of the 11 finished Earth
  shapes (R013_A06 188, R013_A08 106, R013_A10 117, R045_A08 127, R045_A10 91, R080_A06 83, R080_A08 49, R080_A10 32,
  R100_A06 12, R100_A08 6, R100_A10 6 min), ln(minutes) = c0 + c1 ln(spheres per grain) + c2 [aspect 0.6]:
  minutes = 4.00 x N^0.944 x 1.01 for aspect 0.6. Prediction for R045_A06 (N = 49.4): 160 min per run. Seed 1 had
  already exceeded it at the time of logging (likely cause: fill stages and/or the final settle hitting their time
  caps at ~40 ms/step and ~11.4 GB GPU memory).
- Settle logging (from R045_A06 seed 2 onwards): every repose run records, for the floor settle, each fill stage and
  the final settle, whether the rms speed reached its threshold (0.01 / 0.05 / 0.01 m/s, scaled by sqrt(g_E/g)) or
  the time cap was hit, plus the rms speed at gate opening (summary.json: settle_log, rms_at_gate_m_s,
  fill_stages_timed_out, final_settle_timed_out). For earlier seed-1 runs the rms at gate opening is taken from the
  first saved slump frame (translational rms of the free grains) and a final-settle timeout inferred from it (rms >=
  0.01 m/s); none of the 11 finished seed-1 runs timed out (rms at gate 0.0001-0.003 m/s). Earlier seed-2 runs have no
  record. grid_results.csv flags any run whose final settle timed out (settle_flag column).
- Queue change (2026-11-07 00:45): R045_A06 seed 1 runs to completion (it tests the 11-point fit prediction), then the
  L0 anchor (both seeds), then the lunar subset; R045_A06 seed 2 moves to an optional last item that can be
  cancelled. If it is dropped, R045_A06 is marked "1 seed" in the results table and figures.
- Fill procedure, to revisit before chute/hopper work resumes. [Corrected 2026-11-07 03:50; the first version of
  this entry said elongated shapes need ~2x the fill stages, which was wrong.] The fill drops grains in stages of
  loose grid layers spaced 2.05 x the grain bounding radius, so the stage count scales with the bounding radius. For
  the repose column (0.08 x 0.04 m, 8749 grains) the count is: R045_A06 3 stages (rb 1.69 mm), R013_A06 3 (rb 1.85 mm),
  shipped L0 5 (rb 2.25 mm). With all settles capped, R045_A06 can simulate at most 5.8 s before the gate opens
  (floor 1 s + 3 x 0.6 s + final 3 s). Its seed 1 spent over 10.7 h in that phase, i.e. at least ~67 ms/step, while
  GPU memory sat at 11.4-11.7 of 12.3 GB. The slowness is per-step cost, most likely GPU memory pressure (possible
  spill to shared system memory under WSL), not the number of stages. To revisit: memory use of the heaviest shapes
  (more spheres per grain, elongated) and the fill/settle caps, keeping a strict final settle before the gate opens.

## 2026-11-07

- R045_A06 seed 1 (dt 5e-6) stopped by decision at 07:51, 14 h 47 min after its process started (2026-11-06 17:04).
  The gate opened after 11 h 01 min; the rms speed in the first slump frame was ~0 m/s, so the final settle reached
  its threshold (inferred; this run started before settle logging). The slump then ran at ~37 min per 0.02 s frame
  (~560 ms/step; 7 frames = 0.12 s simulated in 3 h 44 min), against 41.5 ms/step for R013_A06 (similar spheres per
  grain and aspect), with total GPU memory in use at 11.35-11.71 of 12.28 GB. The 1.0 s minimum slump would have
  needed ~30 h more. No angle was recorded; the partial frames stay in runs/R045_A06_earth_s1/ as a record. This is a
  stop by decision, not an abort. R045_A06 is rerun at dt 2.5e-6 after the L0 anchor and the lunar subset (seed 1
  required, seed 2 optional), justified by the L0 time-step convergence result (41.54 ± 0.67 deg SE at dt 5e-6 vs
  41.47 ± 0.98 at dt 2.5e-6).
- R045_A06 carries two separate predictions from the 11 finished Earth shapes. (1) Run time: the fit predicted 160
  min per run (entry of 2026-11-06 22:46). This failed: seed 1 ran 887 min without finishing. (2) Angle: the fit
  over the 11 two-seed means predicts ~38.5 deg (a linear least-squares check, angle = 47.17 - 4.44 R - 11.95 c/a,
  rms residual 1.1 deg, gives 38.3 deg at the measured R 0.415 and c/a 0.585). The angle prediction is NOT tested
  yet; the dt 2.5e-6 rerun will test it.
- Run watchdog (all runs from 2026-11-07 08:00): a run is stopped and reported, with no retry and no fallback, if for
  10 min without a break (polled every 30 s) either
  (a) total GPU memory in use exceeds 11.0 GB (nvidia-smi memory.used > 11000 MiB, the same quantity quoted
      elsewhere in this log, including the ~1.1 GB idle baseline), or
  (b) ms/step exceeds 3x the shape's expected rate.
  Expected rates are the slump-phase median of the shape's Earth seed-1 run at dt 5e-6 (frame timestamps): R100_A08
  0.9, R045_A08 22.2, R013_A08 19.2 ms/step. In every finished run the fill and settle phases ran cheaper per step
  than the slump, so this reference is conservative. L0 uses 13.9 ms/step (5.5 at dt 2.5e-6) from the dt check;
  lunar runs use the shape's Earth rate. R045_A06 at dt 2.5e-6 has no healthy reference and uses R013_A06's 41.5
  ms/step. ms/step during a run comes from runs/<run>/progress.csv, which logs the start and end of every solver call
  (the calls themselves are unchanged); while a call is still running, the elapsed time over its step count is used
  as a lower bound. Every poll is logged to logs/<run>.guard.csv. After a stop the shape's remaining runs are
  skipped. For reference, R013_A06 (11.46-11.66 GB, finished normally) would have been stopped under rule (a); L0
  repose runs peaked at ~7.9 GB.
- If R045_A06 still exceeds these limits at dt 2.5e-6, nothing more is run on it until options are agreed (fewer
  grains in the column, or a narrower slice).
- Watchdog stop (2026-11-07 11:16): R100_A08_lunar_s1 stopped after 12 min: ms/step above 3x expected (2.7) for 10 min (now 13.6 ms/step, last completed call; GPU memory 1883 MiB). No result recorded for this run; not retried.
  Diagnosis of this stop: per-call cost rose during the floor settle from 0.6 to 13.6 ms/step (Earth R100_A08 ran
  the whole test in 6.5 min). At 11:07 nvidia-smi showed the GPU in P8 (270 MHz, throttle reason "GPU idle", 32%
  utilization, 1.9 GB in use); the next run (R045_A08 lunar) ran at P1, 2415 MHz. Likely cause: the GPU power state
  dropping under the light 2-sphere load (seen before in the smoke test), not memory. From 11:21 the GPU state is
  logged every 30 s to results/repose_grid/logs/gpu_clock.csv.
- Queue additions (2026-11-07 12:55): after R045_A06, R100_A08 lunar seed 1 is rerun (the 11:16 run gave no result;
  its records are kept as runs/R100_A08_lunar_s1_stopped_watchdog/ and logs/R100_A08_lunar_s1_stopped_watchdog.*).
  Then, if time allows, lunar seed 2 for R100_A08, R045_A08 and R013_A08 (optional; cancel with
  results/repose_grid/CANCEL_LUNAR_S2). Same watchdog. A DEME abort on these is reported, not rerun automatically.
- Watchdog change (2026-11-07 13:00): for roundness-1.0 shapes (R100_*) the watchdog stops on GPU memory only, not
  on ms/step (ms/step is still logged): a slowdown there costs minutes, not days. All other shapes keep both checks.
- L0 anchor at Crr 0.22 (dt 5e-6, rough-floor slump test): seeds 38.85 and 39.72 deg -> 39.28 ± 0.44 deg (2 seeds,
  ± half range), 0.72 deg below the 40 deg target and within 0.3 deg of LHS-1's 39.6 deg. Calibration accepted;
  no re-tune of Crr.
- dt 2.5e-6 memory finding, R045_A06 seed 1 (logged 2026-11-07 22:40, run in progress, angle pending). At dt 5e-6
  the slump ran at ~540-560 ms/step (36-40 min per 0.02 s frame) with total GPU memory in use at 11.4-11.7 GB, and
  the gate opened after 11 h 01 min. At dt 2.5e-6 (same templates, seed, fill and settle) the slump runs at ~13 ms/step
  (median 13.6, range 12.7-22.3; ~2 min per frame), GPU memory peaks at 6.9-7.4 GB, and the gate opened after 68 min
  (all 3 fill stages and the final settle reached their thresholds; rms at gate 0.0003 m/s). Per simulated second the
  dt 2.5e-6 run is ~18x faster despite twice the steps. Interpretation: DEME's adaptive contact margin scales with
  velocity x dt, so halving dt shrinks the candidate-contact lists; for this many-sphere (49.4 per grain), elongated
  shape that brings memory below the point where the dt 5e-6 run likely spilled out of GPU memory (consistent with
  the L0 note of 2026-11-05: 5.5 vs 13.9 ms/step). Note for chute/hopper work: for many-sphere shapes, dt 2.5e-6
  may be both cheaper and lighter on memory than 5e-6.
- Lunar seed 2 time step (2026-11-07 22:50): the optional lunar seed-2 runs of R045_A08 and R013_A08 use dt 2.5e-6
  (converged on L0: 41.54 ± 0.67 deg SE at dt 5e-6 vs 41.47 ± 0.98 at dt 2.5e-6, 3 seeds each; faster for
  many-sphere shapes; more watchdog headroom). R100_A08 lunar (seed 1 rerun and seed 2) stays at dt 5e-6. Their
  lunar seed 1 runs are at dt 5e-6, so those shapes mix time steps across seeds; grid_results.csv records the dt per
  run (time_step_s1, time_step_s2; time_step_s = "mixed" when they differ).
- Lunar minus Earth uncertainty (for the paper): one lunar seed against a 2-seed Earth mean is taken as ±1.4 deg
  (1 sd). Equivalent per-seed sd 1.4 / sqrt(1 + 1/2) = 1.14 deg; with other seed counts sd = 1.14 x sqrt(1/n_lunar +
  1/n_earth) (e.g. 1.14 deg for 2 vs 2). grid_results.csv: earth_mean, lunar_minus_earth, lunar_minus_earth_sd on
  lunar rows. Seed-1 results: R045_A08 lunar 35.83 vs Earth 35.02 -> +0.81 ± 1.4 deg (0.6 sd); R013_A08 lunar 35.05
  vs Earth 36.92 -> -1.87 ± 1.4 deg (1.3 sd). Neither differs from zero at 2 sd.
- Approved (2026-11-07 22:45): lunar minus Earth sd = 1.14 x sqrt(1/n_lunar + 1/n_earth) deg (1.4 deg for 1 lunar
  seed vs a 2-seed Earth mean; 1.14 deg for 2 vs 2), as implemented in grid_results.csv.
- R045_A06 angle-prediction test (2026-11-07 23:24): dt 2.5e-6 seed 1 gave 36.14 deg (165 min, peak GPU memory
  8.3 GB, final settle reached, rms at gate 0.0003 m/s). Predicted ~38.5 deg from the 11 finished Earth shapes (linear
  check 38.3 deg, rms residual 1.1 deg): seed 1 is 2.4 deg below (2.2 below the linear check), ~1.4-1.5 sd against
  the combined fit residual and per-seed scatter (sqrt(1.1^2 + 1.14^2) = 1.6 deg). In the aspect-0.6 row it sits
  below both neighbours (R013_A06 39.38, R080_A06 37.31). Seed 2 (dt 2.5e-6, optional) started 23:24.

## 2026-11-08

- R045_A06 at dt 2.5e-6: seeds 36.14 and 36.29 deg -> 36.21 ± 0.08 deg (2 seeds, ± half range); seed 2 took 156 min,
  peak GPU memory 9.1 GB, no watchdog stop.
- Blind prediction for R045_A06 was 38.5 deg; measured 36.21 deg (2-seed mean); miss +2.3 deg (prediction high).
  The R045_A06 dip in the aspect-0.6 row (R013_A06 39.38, R045_A06 36.21, R080_A06 37.31) is not seed scatter: its
  two seeds agree to 0.15 deg.
- Design-chart surface refitted on all 12 grid shapes (Earth, 2-seed means; R = measured roundness, A = measured
  (b/a + c/a)/2; scripts/11_design_fit.py -> results/repose_grid/design_fit.json). Three forms, compared by
  leave-one-out (LOO) RMSE:
    linear       angle = 45.99 - 4.28 R - 10.71 A           in-sample RMSE 1.18, LOO RMSE 1.59 deg
    linear + R^2 angle = 46.36 - 6.10 R - 10.72 A + 1.59 R^2  in-sample RMSE 1.17, LOO RMSE 1.75 deg
    R(1-A) only  angle = 34.22 + 5.81 R(1-A)               in-sample RMSE 2.48, LOO RMSE 2.85 deg
  Kept: linear (lowest LOO RMSE). Worst LOO miss: R100_A06, predicted 2.95 deg below measured; next R100_A08 +2.89,
  R080_A10 -2.15, R045_A06 +2.12. The chart will state ±1.6 deg (LOO RMSE) as its accuracy. Caveat: the form was
  chosen as the lowest of three LOO scores, which biases that RMSE slightly low.
- Lunar subset complete (2026-11-08 09:20), 2 seeds per shape, all settles clean. Lunar minus Earth (sd = 1.14 x
  sqrt(1/2 + 1/2) = 1.14 deg): R100_A08 31.80 vs 30.72 -> +1.08 ± 1.14 (dt 5e-6 both seeds); R045_A08 35.21 vs
  35.02 -> +0.19 ± 1.14; R013_A08 35.67 vs 36.92 -> -1.25 ± 1.14 (both: seed 1 dt 5e-6, seed 2 dt 2.5e-6). None
  differs from zero at 2 sd: no gravity effect on the angle is resolved at this precision.
