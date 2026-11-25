# Project Decisions (Phase 1 / Phase 2 split)

## 1. Testbed
Simulated edge environment on a laptop (no Jetson Orin Nano / Raspberry Pi).
Primary and Fallback run as separate OS processes; faults are injected in software.
The paper must say "simulated edge environment" -- no hardware claims.

## 2. Forecasting task
Predict Global_active_power 15 minutes ahead from the previous 60 minutes.
Chronological split 70/15/15 (train 2006-12 -> 2009-09, val -> 2010-04, test 2010-04 -> 2010-11).
Test period contains no winter (add to Limitations). Scaler fitted on train only.

## 3. Phase 1 scope
- Data pipeline + chronological split                                   [done]
- Primary model: Bi-LSTM (val MAE 0.378 kW)                              [done]
- Fallback model: Ridge, selected by measurement (section 7)             [done]
- Reliability score c (normalised conformal interval width)             [done]
- Health score Z_t (Eq. 1-4) with severe-fault override + unit tests    [done]
- 4-state machine with hysteresis, debounce, dwell + unit tests         [done]
- Runtime system: worker processes, heartbeat, watchdog, restart, probes [done]
- Calibration of thresholds from a fault-free run (section 8)           [done]
- Faults (4): process kill, process hang, latency spike (+25 ms), input noise (1 kW)
- Policies: P (proposed), A (Primary only), B (Fallback only),
            C (simple threshold), E (Primary + auto-restart)
- Metrics: A, Q, detection, service recovery, back-to-Primary, switches (section 9)
- Dashboard: live demo with click-to-inject faults, experiment runner, results

## 4. Phase 2 scope (kept for later)
- CPU saturation and memory pressure faults (code exists in src/faults/injectors.py, not run)
- Network delay and OOD-input faults
- Input-quality check in the data layer (F3 noise was NOT detected in Phase 1)
- Backoff (D doubling); FAILSAFE<->FALLBACK dwell
- Full ablation D0-D3
- Statistics: 30 repetitions, bootstrap CI, Wilcoxon
- Stronger Primary (larger accuracy gap to the Fallback)
- Optional C-MAPSS dataset, energy estimates

## 5. Success criteria (what would FALSIFY each hypothesis)
- H1: falsified if Config A does not have lower Q than P under hard/perf faults.
- H2: falsified if Config C does not switch more often than P.
- H3/H4: Phase 2.

## 6. Motivating scenario (DRAFT -- review and rewrite in your own words)
Who: an edge gateway in a home or small building that runs a home-energy-management controller.
The controller uses short-term load forecasts to decide when to run flexible loads (water heater,
EV charging, battery) and whether it can take part in a utility demand-response request.
Problem: if the forecasting service crashes, hangs or becomes slow, the controller either acts on
stale numbers or stops controlling, so it can miss a demand-response window or create a peak.
A slightly less accurate forecast that arrives on time, and is flagged as degraded, is more useful
than no forecast. Our framework keeps forecasts flowing (Fallback) and tells the controller when
they are degraded (serving tag).

Expected examiner question: "The forecast is 15 minutes ahead, why do milliseconds matter?"
Answer: the millisecond SLO is not the business deadline. It is a health signal: a gateway may serve
many requests (several devices or households), and a sudden latency rise is an early sign of
contention or a failing process. The business-level benefit is continuity: the controller always has
a timely forecast, and knows when it is degraded.
Limits to state honestly: this scenario is a motivation, not something we measured; Phase 1 uses one
household's data and a simulated gateway.

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

Original plan (RF Fallback) was falsified -- every RF was slower than the Primary.
Primary is only ~6% more accurate than the Fallback; framework value is availability,
not accuracy. Paper text must say Fallback = Ridge (not Random Forest).
Training is reproducible (seed 42); latency is hardware-dependent.

## 8. Calibration (Step 8b)
Provisional L_SLO = 5 ms caused a false switch in a fault-free run (system p95 = 5.5 ms,
because of inter-process communication). Thresholds are now derived from a fault-free run
by fixed rules (experiments/calibrate.py): L_SLO = 14 ms, Z_switch = 0.50, Z_recover = 0.34,
C_target = 0.236, C_min = 0.118. The fault-free check then gave 0 transitions.
Calibration and this check use the same validation data (in-sample); final numbers use the test set.

## 9. Measurement decisions
- Every 50 ms slot is one request. Slots missed while a request was blocked are recorded as
  "missed" (no catch-up), otherwise outages are under-counted (coordinated omission).
- Model faults hit the model that is serving at fault onset: the Primary, or the Fallback
  in policy B. Each policy starts only the models it uses.
- Q tolerance: |error| <= 0.5 kW (fixed before the results).
- Service recovery: requests within the SLO again for 2 s (>= 95% of them).

## 10. Findings so far (pilot, 1 run each, validation data)
- Kill: P detects in 0.5 s and loses ~1% of requests; E (restart only) has a ~5.7 s outage
  (fault-window availability 88.5% vs 99.0%). Restart takes ~5-6 s on this laptop.
- Hang: detected in 0.6 s by the inference watchdog (faster than the 1 s heartbeat).
- Latency +25 ms: soft switch after 1.5 s (p95 reacts faster than the paper's tau_w + N_w*dt bound);
  return 6 s after the fault ends (probe window must clear).
- Input noise: NO switch; MAE rose from ~0.38 to ~0.57 kW. The reliability proxy is too weak at the
  calibrated threshold. Report as a negative result.
