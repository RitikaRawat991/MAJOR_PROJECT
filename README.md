# Fault-Tolerant Adaptive AI Interface (Phase 1)

A runtime orchestration layer that serves electricity-load forecasts from an IoT data stream
and switches between a **Primary** model (Bi-LSTM) and a **Fallback** model (Ridge) when the
system or the model becomes unhealthy. Simulated edge node on a laptop.

## Run everything (one command)

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run dashboard.py
```

Use `python -m` (needed on Windows for the worker processes). Then, in the browser:

| Page | What it does |
|---|---|
| 1. Live demo | Start the system with a policy, inject faults by clicking, watch it react |
| 2. Run experiments | Run fault scenarios x policies in the background (validation or test data), or run any project command (tests, one scenario, calibration, training) and see its terminal output |
| 3. Results | Metrics of all runs, charts, hypothesis checks, CSV download |
| 4. Run viewer | Timeline of one saved run |
| 5. Setup & calibration | Thresholds, calibration run, fault-free check, clear results |

## First-time setup (only once)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Put the UCI file at `data/raw/household_power_consumption.txt`, then build the data and models:
```powershell
python -m src.data.load_data            # clean + chronological split
python -m src.models.primary_lstm       # Primary (Bi-LSTM)
python -m src.models.fallback_ridge     # Fallback (Ridge)
python -m src.framework.reliability     # reliability score for the Primary
```
Then calibrate on page 5 of the dashboard. Optional: `python -m src.eval.fallback_sweep`
reproduces the Fallback selection; `python -m pytest tests -q` runs the unit tests.

## Project layout

```
src/data/        load_data.py (cleaning, split), windows.py (scaling, windows, features)
src/models/      primary_lstm.py, fallback_ridge.py
src/framework/   the orchestration layer:
                 workers.py (model processes), monitor.py (CPU/memory), reliability.py (score c),
                 health_score.py (Z), state_machine.py (4 states), service.py (main loop)
src/faults/      injectors.py (kill, hang, latency, noise; cpu/memory for Phase 2)
src/eval/        metrics.py (A, Q, detection, recovery), fallback_sweep.py (model selection)
experiments/     calibrate.py, run_nominal.py, run_scenario.py
configs/         default.yaml (starting values), calibrated.yaml (used by everything)
tests/           health score and state machine tests
dashboard.py     UI: drives and shows the framework (it is not the framework itself)
DECISIONS.md     every design decision and why
```

## Policies

| | Models | On a fault |
|---|---|---|
| P | Primary + Fallback | proposed policy: health score, hysteresis, debounce, dwell, probing |
| A | Primary only | nothing |
| E | Primary only | restart the Primary |
| B | Fallback only | nothing |
| C | Primary + Fallback | single threshold, no debounce, no dwell |

## Status
Phase 1 pilot: one run per scenario and policy. Phase 2 (more faults, repetitions,
statistics) is listed in DECISIONS.md.
