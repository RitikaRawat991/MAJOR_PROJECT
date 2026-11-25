"""run_nominal.py -- Step 8. Fault-free run of the full system (checks for false switches).
Run:  python -m experiments.run_nominal [config.yaml]   (or the dashboard's Setup page)
"""
import json
import sys
import numpy as np
from src.data.windows import load_split, make_windows
from src.framework.health_score import HealthConfig
from src.framework.service import EdgeService, RuntimeConfig
from src.framework.state_machine import SwitchConfig

DURATION = 60.0


def nominal_check(cfg="configs/calibrated.yaml", duration=DURATION, log=print):
    scaler = json.load(open("models/scaler.json"))
    Xva, yva, _ = make_windows(load_split("val"), scaler, stride=5)
    h = len(Xva) // 2                           # 2nd half: NOT used to fit the reliability model
    X, y = Xva[h:], yva[h:]

    svc = EdgeService(HealthConfig.from_yaml(cfg), SwitchConfig.from_yaml(cfg),
                      RuntimeConfig.from_yaml(cfg), scaler["std"])
    try:
        log("starting workers (loading models in separate processes)...")
        svc.run(X, y, duration)
    finally:
        svc.shutdown()
    svc.save("nominal")

    done = [r for r in svc.requests if r["lat_ms"] is not None]
    lat = np.array([r["lat_ms"] for r in done])
    tk = [t for t in svc.ticks if t["Z"] is not None]
    Z = np.array([t["Z"] for t in tk])
    out = {"requests": len(svc.requests), "answered": len(done),
           "latency p50 ms": round(float(np.percentile(lat, 50)), 2),
           "latency p95 ms": round(float(np.percentile(lat, 95)), 2),
           "transitions (should be 0)": len(svc.ctl.log),
           "Z median": round(float(np.median(Z)), 3) if len(Z) else None,
           "Z p99": round(float(np.percentile(Z, 99)), 3) if len(Z) else None}
    for k, v in out.items():
        log(f"{k}: {v}")
    return out


if __name__ == "__main__":                      # required on Windows (spawn)
    nominal_check(sys.argv[1] if len(sys.argv) > 1 else "configs/calibrated.yaml")
