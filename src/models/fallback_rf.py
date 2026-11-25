"""fallback_rf.py -- Step 4. Train the Fallback Random Forest.
Run:  python -m src.models.fallback_rf
"""
import json
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from src.data.windows import load_split, make_windows

SEED = 42
MODEL_DIR = Path("models")
LAGS = [1, 2, 3, 5, 10, 15, 30, 45, 60]     # minutes back from "now"


def window_to_features(Xw):
    """Xw: (N, 60, 3) -> (N, F). Same windows as the Primary => fair comparison."""
    load = Xw[:, :, 0]
    cols = [load[:, -k] for k in LAGS]                   # lag features
    cols += [load[:, -15:].mean(1),                      # 15-min mean
             load.mean(1),                               # 60-min mean
             load[:, -15:].std(1)]                       # 15-min variability
    cols += [Xw[:, -1, 1], Xw[:, -1, 2]]                 # time of day (sin, cos)
    return np.stack(cols, axis=1).astype("float32")


def mae_kw(pred, true, std):
    return float(np.mean(np.abs(pred - true)) * std)


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    scaler = json.load(open(MODEL_DIR / "scaler.json"))  # SAME scaler as Primary
    std = scaler["std"]

    train_df, val_df = load_split("train"), load_split("val")
    Xtr, ytr, _ = make_windows(train_df, scaler, stride=10)
    Xva, yva, _ = make_windows(val_df, scaler, stride=5)
    Ftr, Fva = window_to_features(Xtr), window_to_features(Xva)
    print(f"features: {Ftr.shape[1]}   train: {len(Ftr):,}   val: {len(Fva):,}")

    rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                               min_samples_leaf=10, n_jobs=-1,
                               random_state=SEED)
    t0 = time.time()
    rf.fit(Ftr, ytr)
    print(f"train time: {time.time()-t0:.0f}s")

    pv = rf.predict(Fva)
    print(f"[baseline] persistence val MAE: {mae_kw(Xva[:, -1, 0], yva, std):.4f} kW")
    print(f"[Fallback RF] val MAE: {mae_kw(pv, yva, std):.4f} kW")

    joblib.dump(rf, MODEL_DIR / "fallback_rf.joblib")
    size_mb = (MODEL_DIR / "fallback_rf.joblib").stat().st_size / 1e6
    print(f"saved models/fallback_rf.joblib ({size_mb:.1f} MB)")