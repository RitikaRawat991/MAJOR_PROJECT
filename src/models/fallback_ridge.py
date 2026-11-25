"""fallback_ridge.py -- Step 4c. Train and save the selected Fallback (Ridge).
Run:  python -m src.models.fallback_ridge
"""
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from src.data.windows import load_split, make_windows
from src.models.fallback_rf import window_to_features

MODEL_DIR = Path("models")

if __name__ == "__main__":
    scaler = json.load(open(MODEL_DIR / "scaler.json"))   # same scaler as Primary
    std = scaler["std"]

    Xtr, ytr, _ = make_windows(load_split("train"), scaler, stride=10)
    Xva, yva, _ = make_windows(load_split("val"), scaler, stride=5)
    Ftr, Fva = window_to_features(Xtr), window_to_features(Xva)

    model = Ridge(alpha=1.0).fit(Ftr, ytr)
    mae = float(np.mean(np.abs(model.predict(Fva) - yva)) * std)
    print(f"Fallback Ridge val MAE: {mae:.4f} kW")      # expect ~0.4040

    joblib.dump(model, MODEL_DIR / "fallback_ridge.joblib")
    print("saved models/fallback_ridge.joblib")