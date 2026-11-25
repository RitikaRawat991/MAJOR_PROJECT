"""metrics.py -- Step 11. Metrics for one saved run (paper Eq. 5-8 and Table V).

Definitions (fixed BEFORE looking at the results):
  s_i              = 1 if request i got an answer within L_SLO                 (Eq. 5)
  A                = mean(s_i)                                                 availability
  Q                = mean(s_i * 1[|error_i| <= TOL_KW])                        (Eq. 6)
  detection        = time from fault onset to the first reaction of the system
                     (a state transition or a restart)
  service recovery = time from onset until requests are again answered within the SLO
                     for RECOVERY_WINDOW seconds (>= 95% of them)               (Eq. 8, timeliness)
  back to Primary  = time from onset until the controller returns to PRIMARY
  false switches   = transitions before the fault starts
TOL_KW = 0.5 kW is a design choice (an error a household-load user can still act on);
it is fixed here, not tuned on the results.
"""
import numpy as np
import pandas as pd

TOL_KW = 0.5
RECOVERY_WINDOW = 2.0
T_ON, T_OFF = 30.0, 80.0


def _first_after(times, t_on):
    times = np.sort(np.asarray(times, dtype=float))
    times = times[times >= t_on]
    return float(times[0] - t_on) if len(times) else np.nan


def service_recovery(t, ok, t_on, window=RECOVERY_WINDOW, frac=0.95):
    after = t >= t_on
    tt, oo = t[after], ok[after]
    if len(tt) == 0:
        return np.nan
    if oo.all():
        return 0.0
    first_bad = int(np.argmax(~oo))
    for j in range(first_bad, len(tt)):
        if not oo[j] or tt[j] + window > tt[-1]:
            continue
        m = (tt >= tt[j]) & (tt < tt[j] + window)
        if oo[m].mean() >= frac:
            return float(tt[j] - t_on)
    return np.nan


def run_metrics(req, trans, events, l_slo, std, fault=True, t_on=T_ON, t_off=T_OFF):
    """req/trans/events: DataFrames saved by EdgeService.save(). Returns a dict of metrics."""
    t = req["t"].to_numpy(float)
    lat = pd.to_numeric(req["lat_ms"], errors="coerce").to_numpy(float)
    ok = ~np.isnan(lat) & (lat <= l_slo)
    pred = pd.to_numeric(req["pred"], errors="coerce").to_numpy(float)
    err = np.abs(pred - req["y"].to_numpy(float)) * std
    good = ok & (np.nan_to_num(err, nan=np.inf) <= TOL_KW)
    served = req["served"].fillna("none").astype(str).to_numpy()
    duration = float(t.max()) if len(t) else 0.0

    tr_t = trans["t"].to_numpy(float) if trans is not None and not trans.empty else np.array([])
    tr_dst = trans["dst"].astype(str).to_numpy() if trans is not None and not trans.empty else np.array([])
    if events is not None and not events.empty:
        restarts = events.loc[events["event"].astype(str).str.startswith("restart"), "t"].to_numpy(float)
    else:
        restarts = np.array([])

    m = {"requests": len(t),
         "A %": ok.mean() * 100 if len(t) else np.nan,
         "Q %": good.mean() * 100 if len(t) else np.nan,
         "MAE kW": float(np.nanmean(err)) if np.isfinite(err).any() else np.nan,
         "transitions": int(len(tr_t)),
         "switches/hour": len(tr_t) / duration * 3600 if duration else np.nan}
    for key in ("primary", "fallback"):
        l = lat[(served == key) & ~np.isnan(lat)]
        m[f"p95 {key} ms"] = float(np.percentile(l, 95)) if len(l) else np.nan
    if fault:
        w = (t >= t_on) & (t < t_off)
        m.update({
            "A fault %": ok[w].mean() * 100 if w.any() else np.nan,
            "Q fault %": good[w].mean() * 100 if w.any() else np.nan,
            "fallback share fault %": (served[w] == "fallback").mean() * 100 if w.any() else np.nan,
            "detection s": _first_after(np.concatenate([tr_t, restarts]), t_on),
            "service recovery s": service_recovery(t, ok, t_on),
            "back to Primary s": _first_after(tr_t[tr_dst == "PRIMARY"], t_on),
            "false switches": int((tr_t < t_on).sum()),
        })
    return m
