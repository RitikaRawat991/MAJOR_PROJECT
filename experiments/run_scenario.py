"""run_scenario.py -- Steps 9-10. Run ONE policy against ONE fault scenario.
Usually started from the dashboard (page 2). From a terminal:
    python -m experiments.run_scenario kill P [configs/calibrated.yaml] [val|test]
Scenarios: kill hang latency noise        Policies: P A B C E
Data: "val"  = 2nd half of the validation set (development runs; thresholds were tuned on validation data)
      "test" = test set (final evaluation, run once at the end)
"""
import dataclasses
import json
import sys
import numpy as np
from src.data.windows import load_split, make_windows
from src.faults.injectors import make_fault
from src.framework.health_score import HealthConfig
from src.framework.service import EdgeService, RuntimeConfig
from src.framework.state_machine import SwitchConfig

T_ON, T_OFF, DURATION = 30.0, 80.0, 130.0
POLICIES = ("P", "A", "B", "C", "E")
DEFAULT_CFG = "configs/calibrated.yaml"
PREFIX = {"val": "scenario", "test": "testset"}       # file name prefix in results/


def load_configs(policy, cfg):
    hc, sw, rt = HealthConfig.from_yaml(cfg), SwitchConfig.from_yaml(cfg), RuntimeConfig.from_yaml(cfg)
    if policy == "C":                    # simple threshold: one threshold, no debounce, no dwell
        hc = dataclasses.replace(hc, Z_recover=round(hc.Z_switch - 0.01, 2))
        sw = dataclasses.replace(sw, N_w=1, N_rec=1, D=0.0)
    return hc, sw, rt


def load_stream(split="val"):
    scaler = json.load(open("models/scaler.json"))
    if split == "test":
        X, y, _ = make_windows(load_split("test"), scaler, stride=5)
        return X, y, scaler
    Xva, yva, _ = make_windows(load_split("val"), scaler, stride=5)
    h = len(Xva) // 2                    # same held-out half for every run
    return Xva[h:], yva[h:], scaler


def run_one(scenario, policy, cfg=DEFAULT_CFG, split="val"):
    X, y, scaler = load_stream(split)
    hc, sw, rt = load_configs(policy, cfg)
    svc = EdgeService(hc, sw, rt, scaler["std"], policy=policy)
    on, off = make_fault(scenario)
    try:
        svc.run(X, y, DURATION, events=[(T_ON, on), (T_OFF, off)])
    finally:
        svc.shutdown()
    svc.save(f"{PREFIX[split]}_{scenario}_{policy}")
    return svc, scaler["std"], hc.L_slo


def summarise(svc, std, l_slo, label):
    R = svc.requests
    t = np.array([r["t"] for r in R])
    ok = np.array([r["lat_ms"] is not None and r["lat_ms"] <= l_slo for r in R])
    err = np.array([abs(r["pred"] - r["y"]) * std if r["pred"] is not None else np.nan for r in R])
    served = np.array([r["served"] or "none" for r in R])
    print(f"\n=== {label}  (fault ON {T_ON:.0f}s -> OFF {T_OFF:.0f}s) ===")
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
    print("restarts:", [(round(t, 1), e) for t, e in svc.events_log if e.startswith("restart")])
    print("false switches before onset:", sum(1 for tr in svc.ctl.log if tr.t < T_ON))


if __name__ == "__main__":
    scenario, policy = sys.argv[1], sys.argv[2]
    cfg = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CFG
    split = sys.argv[4] if len(sys.argv) > 4 else "val"
    svc, std, l_slo = run_one(scenario, policy, cfg, split)
    summarise(svc, std, l_slo, f"{scenario} / policy {policy} / {split} data")
