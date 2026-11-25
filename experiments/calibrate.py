"""calibrate.py -- Step 8b. Derive health-score parameters from a FAULT-FREE run.
Run:  python -m experiments.calibrate        (or the "Setup & calibration" page of the dashboard)

Rules (fixed BEFORE looking at the data):
  L_slo    = ceil(2 x p99 of the windowed-p95 latency)            [ms]
  U_max    = max(0.85, 1.2 x p99 of nominal U);  U_crit = min(1, U_max + 0.10)
  C_target = p5 of the windowed mean reliability;  C_min = 0.5 x C_target
  Z_recover= p99 of nominal Z (computed with the parameters above)
  Z_switch = clip(mean(Z) + 5 std(Z), Z_recover + 0.10, 0.50)
Soft switching is disabled during this run (N_w huge) so the system stays in PRIMARY.
"""
import dataclasses
import json
import math
import numpy as np
import yaml
from src.data.windows import load_split, make_windows
from src.framework.health_score import HealthConfig, health_score
from src.framework.service import EdgeService, RuntimeConfig
from src.framework.state_machine import SwitchConfig

SRC, OUT, DURATION = "configs/default.yaml", "configs/calibrated.yaml", 90.0


def calibrate(src=SRC, out=OUT, duration=DURATION, log=print):
    scaler = json.load(open("models/scaler.json"))
    Xva, yva, _ = make_windows(load_split("val"), scaler, stride=5)
    h = len(Xva) // 2
    X, y = Xva[h:], yva[h:]

    hc = HealthConfig.from_yaml(src)
    sw = dataclasses.replace(SwitchConfig.from_yaml(src), N_w=10 ** 9)   # never soft-switch
    rt = RuntimeConfig.from_yaml(src)
    svc = EdgeService(hc, sw, rt, scaler["std"])
    try:
        log("calibration run (no faults, switching disabled)...")
        svc.run(X, y, duration)
    finally:
        svc.shutdown()

    tk = [t for t in svc.ticks if t["t"] >= rt.tau_w and t["L_p95"] is not None and t["C"] is not None]
    L = np.array([t["L_p95"] for t in tk])
    C = np.array([t["C"] for t in tk])
    U = np.array([t["U"] for t in tk])
    log(f"ticks used: {len(tk)}   state changes during run: {len(svc.ctl.log)}")
    log(f"windowed p95 latency (ms): median {np.median(L):.2f}  p99 {np.percentile(L,99):.2f}  max {L.max():.2f}")
    log(f"U: median {np.median(U):.2f}  p99 {np.percentile(U,99):.2f}  max {U.max():.2f}")
    log(f"windowed c: p5 {np.percentile(C,5):.3f}  median {np.median(C):.3f}  min {C.min():.3f}")

    L_slo = float(math.ceil(2 * np.percentile(L, 99)))
    U_max = max(0.85, round(1.2 * float(np.percentile(U, 99)), 2))
    U_crit = min(1.0, round(U_max + 0.10, 2))
    C_target = round(float(np.percentile(C, 5)), 3)
    C_min = round(0.5 * C_target, 3)
    hc2 = dataclasses.replace(hc, L_slo=L_slo, U_max=U_max, U_crit=U_crit,
                              C_target=C_target, C_min=C_min)
    Z = np.array([health_score(hc2, l, u, c)[0] for l, u, c in zip(L, U, C)])
    Z_recover = round(math.ceil(float(np.percentile(Z, 99)) * 100) / 100, 2)
    Z_switch = round(float(np.clip(Z.mean() + 5 * Z.std(), Z_recover + 0.10, 0.50)), 2)
    log(f"nominal Z with new parameters: mean {Z.mean():.3f}  std {Z.std():.3f}  "
        f"p99 {np.percentile(Z,99):.3f}  max {Z.max():.3f}")
    if Z_recover + 0.10 > 0.50:
        log("WARNING: nominal Z is too high/noisy for a usable dead band; check the testbed.")

    cfg = yaml.safe_load(open(src))
    cfg["health"].update(L_slo=L_slo, U_max=U_max, U_crit=U_crit, C_target=C_target,
                         C_min=C_min, Z_recover=Z_recover, Z_switch=max(Z_switch, Z_recover + 0.01))
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    log(f"calibrated health parameters -> {out}")
    params = {k: cfg["health"][k] for k in
              ("L_slo", "U_max", "U_crit", "C_target", "C_min", "Z_recover", "Z_switch")}
    for k, v in params.items():
        log(f"  {k:10s} = {v}")
    return params


if __name__ == "__main__":
    calibrate()
