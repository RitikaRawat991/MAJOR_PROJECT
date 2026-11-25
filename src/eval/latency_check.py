"""latency_check.py -- single-request latency of Primary vs Fallback.
Run:  python -m src.eval.latency_check
"""
import json
import time
import numpy as np
import torch
import joblib
from src.data.windows import load_split, make_windows
from src.models.primary_lstm import BiLSTM
from src.models.fallback_rf import window_to_features

N_WARM, N_TIMED = 20, 300
scaler = json.load(open("models/scaler.json"))
Xw, _, _ = make_windows(load_split("val"), scaler, stride=50)
Xw = Xw[:N_WARM + N_TIMED]

torch.set_num_threads(1)                    # fair: one thread each
lstm = BiLSTM()
lstm.load_state_dict(torch.load("models/primary_bilstm.pt"))
lstm.eval()
rf = joblib.load("models/fallback_rf.joblib")
rf.n_jobs = 1                               # default parallelism adds overhead at batch=1


def run_lstm(x):
    with torch.no_grad():
        return lstm(torch.from_numpy(x[None])).item()


def run_rf(x):                              # includes feature extraction
    return rf.predict(window_to_features(x[None]))[0]


def measure(fn):
    for x in Xw[:N_WARM]:                   # warm-up (first calls are slow)
        fn(x)
    ts = []
    for x in Xw[N_WARM:]:
        t = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t) * 1000)
    return np.percentile(ts, [50, 95, 99])


for name, fn in [("Primary Bi-LSTM", run_lstm), ("Fallback RF", run_rf)]:
    p50, p95, p99 = measure(fn)
    print(f"{name:16s}  p50={p50:6.2f} ms   p95={p95:6.2f} ms   p99={p99:6.2f} ms")