# Project Decisions (Phase 1 / Phase 2 split)

## 1. Testbed
Simulated edge environment on a laptop (no Jetson Orin Nano / Raspberry Pi).
Faults are injected in software. The paper must say "simulated edge environment" -- no hardware claims.

## 2. Forecasting task
Predict Global_active_power 15 minutes ahead from the previous 60 minutes.
Chronological split 70/15/15 (train 2006-12 -> 2009-09, val -> 2010-04, test 2010-04 -> 2010-11).
Test period contains no winter (add to Limitations). Scaler fitted on train only.

## 3. Phase 1 scope (to demo)
- Data pipeline + chronological split                     [done]
- Primary model (Bi-LSTM)                                  [done]
- Fallback model (selected by measurement, see section 7)  [selected]
- Health score Z_t (Eq. 1-4) with severe-fault override
- 4-state machine (PRIMARY / FALLBACK / RECOVERY / FAILSAFE) with hysteresis, debounce, dwell
- Simulated faults (5): process kill, latency spike, input noise, CPU saturation, memory pressure
- Baselines: A (Primary only), B (Fallback only), C (simple threshold)
- Basic metrics: availability A, quality-conditioned availability Q, switches/hour

## 4. Phase 2 scope (kept for later)
- Backoff + canary probing refinements
- Full ablation D0-D3
- Remaining fault scenarios (network delay, OOD inputs)
- Config E: Primary + auto-restart (stronger baseline)
- Statistics: 30 repetitions, bootstrap CI, Wilcoxon
- Optional C-MAPSS dataset, energy estimates
- Final paper update with real results

## 5. Success criteria (what would FALSIFY each hypothesis)
- H1: falsified if Config A does not have lower Q than P under hard/perf faults.
- H2: falsified if Config C does not switch more often than P.
- H3/H4: to be written in Phase 2.

## 6. Motivating scenario (to write: 2-3 sentences)
TODO

## 7. Model selection (Step 4b)
Pre-registered rule (set BEFORE seeing results): Fallback = most accurate candidate
whose single-request p95 latency <= 0.5 x Primary p95 (Primary p95 = 1.29 ms -> <= 0.65 ms).

Validation results (batch=1, 1 thread, seed 42):
- Persistence      MAE 0.4180 kW  p95 0.00 ms
- Ridge            MAE 0.4040 kW  p95 0.33 ms   <- SELECTED (Fallback)
- RF 10 trees d8   MAE 0.3838 kW  p95 2.19 ms   (fails rule)
- RF 30 trees d10  MAE 0.3788 kW  p95 3.40 ms   (fails rule)
- RF 100 trees d15 MAE 0.3767 kW  p95 11.89 ms  (fails rule)
- Primary Bi-LSTM  MAE 0.3779 kW  p95 1.29 ms

Notes: original plan (RF Fallback) was falsified -- every RF was slower than the Primary.
Primary is only ~6% more accurate than the Fallback; framework value is availability,
not accuracy. Paper text must be updated: Fallback = Ridge (not Random Forest).
Training is reproducible (seed 42 -> identical val MAE across runs); latency is hardware-dependent.
