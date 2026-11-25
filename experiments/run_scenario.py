"""run_scenario.py -- Step 9. Run the full policy (P) against ONE fault scenario.
Run:  python -m experiments.run_scenario kill configs/calibrated.yaml
Scenarios: kill hang latency noise cpu memory
"""
import json
import sys
import numpy as np
from src.data.windows import load_split, make_windows
from src.faults.injectors import make_fault
from src.framework.health_score import HealthConfig
from src.framework.service import EdgeService, RuntimeConfig
from src.framework.state_machine import SwitchConfig

T_ON, T_OFF, DURATION = 30.0, 80.0, 130.0


def summarise(svc, std, l_slo, scenario):
    R = svc.requests
    t = np.array([r["t"] for r in R])
    ok = np.array([r["lat_ms"] is not None and r["lat_ms"] <= l_slo for r in R])
    err = np.array([abs(r["pred"] - r["y"]) * std if r["pred"] is not None else np.nan for r in R])
    served = np.array([r["served"] or "none" for r in R])
    print(f"\n=== scenario: {scenario}  (fault ON {T_ON:.0f}s -> OFF {T_OFF:.0f}s) ===")
    print("phase    requests  availability  MAE(kW)  served-by-fallback")
    for name, m in [("before", t < T_ON), ("fault", (t >= T_ON) & (t < T_OFF)), ("after", t >= T_OFF)]:
        mae = np.nanmean(err[m]) if np.isfinite(err[m]).any() else float("nan")
        print(f"{name:7s}  {m.sum():8d}  {ok[m].mean()*100:10.1f}%  {mae:7.3f}  "
              f"{(served[m] == 'fallback').mean()*100:14.1f}%")
    print("\ntransitions:")
    for tr in svc.ctl.log:
        print(f"  t={tr.t:6.1f}s  {tr.src.value:8s} -> {tr.dst.value:8s} ({tr.reason})")
    if not svc.ctl.log:
        print("  (none)")
    first = [tr for tr in svc.ctl.log if tr.t >= T_ON]
    back = [tr for tr in svc.ctl.log if tr.dst.value == "PRIMARY" and tr.t >= T_ON]
    print(f"\ndetection+switch delay: " + (f"{first[0].t - T_ON:.1f} s" if first else "NO SWITCH"))
    print(f"first return to PRIMARY: " + (f"t={back[0].t:.1f}s ({back[0].t - T_OFF:+.1f}s vs fault OFF)"
                                          if back else "never within the run"))
    print("restarts:", [(round(t, 1), e) for t, e in svc.events_log if e.startswith("restart")])
    print("false switches before onset:", sum(1 for tr in svc.ctl.log if tr.t < T_ON))


if __name__ == "__main__":
    scenario = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else "configs/calibrated.yaml"
    scaler = json.load(open("models/scaler.json"))
    Xva, yva, _ = make_windows(load_split("val"), scaler, stride=5)
    h = len(Xva) // 2
    X, y = Xva[h:], yva[h:]

    hc = HealthConfig.from_yaml(cfg)
    svc = EdgeService(hc, SwitchConfig.from_yaml(cfg), RuntimeConfig.from_yaml(cfg), scaler["std"])
    on, off = make_fault(scenario)
    try:
        svc.run(X, y, DURATION, events=[(T_ON, on), (T_OFF, off)])
    finally:
        svc.shutdown()
    svc.save(f"scenario_{scenario}")
    summarise(svc, scaler["std"], hc.L_slo, scenario)