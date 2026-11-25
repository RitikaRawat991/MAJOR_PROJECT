"""run_nominal.py -- Step 8. Fault-free run of the full system (calibration run).
Run:  python -m experiments.run_nominal
"""
import json
import sys
import numpy as np
from src.data.windows import load_split, make_windows
from src.framework.health_score import HealthConfig
from src.framework.service import EdgeService, RuntimeConfig
from src.framework.state_machine import SwitchConfig

CFG = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
DURATION = 60.0

if __name__ == "__main__":                      # required on Windows (spawn)
    scaler = json.load(open("models/scaler.json"))
    Xva, yva, _ = make_windows(load_split("val"), scaler, stride=5)
    h = len(Xva) // 2                           # 2nd half: NOT used to fit the reliability model
    X, y = Xva[h:], yva[h:]

    svc = EdgeService(HealthConfig.from_yaml(CFG), SwitchConfig.from_yaml(CFG),
                      RuntimeConfig.from_yaml(CFG), scaler["std"])
    try:
        print("starting workers (loading torch in 2 processes)...")
        svc.run(X, y, DURATION)
    finally:
        svc.shutdown()
    svc.save("nominal")

    done = [r for r in svc.requests if r["lat_ms"] is not None]
    lat = np.array([r["lat_ms"] for r in done])
    print(f"\nrequests: {len(svc.requests)}   answered: {len(done)}")
    print(f"served by: " + str({k: sum(1 for r in svc.requests if r['served'] == k)
                                for k in ('primary', 'fallback', None)}))
    print(f"latency ms  p50={np.percentile(lat,50):.2f}  p95={np.percentile(lat,95):.2f}  "
          f"p99={np.percentile(lat,99):.2f}  max={lat.max():.2f}")
    print(f"transitions: {len(svc.ctl.log)}  {[(t.src.value, t.dst.value, t.reason) for t in svc.ctl.log]}")

    tk = [t for t in svc.ticks if t["Z"] is not None]
    Z = np.array([t["Z"] for t in tk])
    U = np.array([t["U"] for t in svc.ticks])
    Cw = np.array([t["C"] for t in tk if t["C"] is not None])
    print(f"U_t   : min {U.min():.2f}  median {np.median(U):.2f}  max {U.max():.2f}")
    print(f"Z_t   : median {np.median(Z):.3f}  p95 {np.percentile(Z,95):.3f}  "
          f"p99 {np.percentile(Z,99):.3f}  max {Z.max():.3f}")
    print(f"windowed c: p5 {np.percentile(Cw,5):.3f}  median {np.median(Cw):.3f}")
