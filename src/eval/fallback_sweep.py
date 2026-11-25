"""fallback_sweep.py -- compare candidate Fallback models on accuracy, latency, size.
Run:  python -m src.eval.fallback_sweep
"""
import json
import pickle
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from src.data.windows import load_split, make_windows
from src.models.fallback_rf import window_to_features

SEED = 42
N_WARM, N_TIMED = 20, 300

scaler = json.load(open("models/scaler.json"))
std = scaler["std"]
Xtr, ytr, _ = make_windows(load_split("train"), scaler, stride=10)
Xva, yva, _ = make_windows(load_split("val"), scaler, stride=5)
Ftr, Fva = window_to_features(Xtr), window_to_features(Xva)
Xt, _, _ = make_windows(load_split("val"), scaler, stride=50)
Xt = Xt[:N_WARM + N_TIMED]                       # windows for timing


def mae(pred):
    return float(np.mean(np.abs(pred - yva)) * std)


def measure(fn):
    for x in Xt[:N_WARM]:
        fn(x)
    ts = []
    for x in Xt[N_WARM:]:
        t = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t) * 1000)
    return np.percentile(ts, [50, 95])


candidates = {
    "Persistence": None,
    "Ridge": Ridge(alpha=1.0),
    "RF 10 trees d8": RandomForestRegressor(10, max_depth=8, min_samples_leaf=10,
                                            n_jobs=-1, random_state=SEED),
    "RF 30 trees d10": RandomForestRegressor(30, max_depth=10, min_samples_leaf=10,
                                             n_jobs=-1, random_state=SEED),
    "RF 100 trees d15": RandomForestRegressor(100, max_depth=15, min_samples_leaf=10,
                                              n_jobs=-1, random_state=SEED),
}

print(f"{'candidate':18s} {'val MAE':>8s} {'p50 ms':>8s} {'p95 ms':>8s} {'size MB':>8s}")
for name, model in candidates.items():
    if model is None:
        val_mae = mae(Fva[:, 0])                  # feature 0 = latest value
        fn, size = (lambda x: x[-1, 0]), 0.0
    else:
        model.fit(Ftr, ytr)
        val_mae = mae(model.predict(Fva))
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1                      # single-request latency
        fn = lambda x, m=model: m.predict(window_to_features(x[None]))[0]
        size = len(pickle.dumps(model)) / 1e6
    p50, p95 = measure(fn)
    print(f"{name:18s} {val_mae:8.4f} {p50:8.2f} {p95:8.2f} {size:8.2f}")

print("\nPrimary Bi-LSTM reference: val MAE 0.3779, p50 0.87 ms, p95 1.29 ms")