"""windows.py -- scaling + sliding windows (shared by Primary and Fallback)."""
import numpy as np
import pandas as pd

TARGET = "Global_active_power"
LOOKBACK = 60      # minutes of history fed to the model
HORIZON = 15       # predict this many minutes ahead


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(f"data/processed/{name}.csv",
                       index_col="timestamp", parse_dates=True)


def fit_scaler(train_df: pd.DataFrame) -> dict:
    s = train_df[TARGET]                       # pandas skips NaN
    return {"mean": float(s.mean()), "std": float(s.std())}


def make_features(df: pd.DataFrame, scaler: dict):
    y = ((df[TARGET].values - scaler["mean"]) / scaler["std"]).astype("float32")
    hour = df.index.hour.values + df.index.minute.values / 60.0
    sin = np.sin(2 * np.pi * hour / 24).astype("float32")   # time of day,
    cos = np.cos(2 * np.pi * hour / 24).astype("float32")   # circular encoding
    X = np.stack([y, sin, cos], axis=1)                     # shape (T, 3)
    return X, y


def make_windows(df: pd.DataFrame, scaler: dict, stride: int = 1):
    """Returns Xw (N, 60, 3), yw (N,), starts (N,). Windows with NaN are dropped."""
    X, y = make_features(df, scaler)
    T = len(y)
    n = T - LOOKBACK - HORIZON + 1
    starts = np.arange(0, n, stride)

    nan = np.isnan(y).astype(np.int32)
    c = np.concatenate([[0], np.cumsum(nan)])               # fast NaN counting
    nan_in_input = c[starts + LOOKBACK] - c[starts]
    nan_in_target = nan[starts + LOOKBACK - 1 + HORIZON]
    starts = starts[(nan_in_input == 0) & (nan_in_target == 0)]

    idx = starts[:, None] + np.arange(LOOKBACK)[None, :]
    Xw = X[idx]
    yw = y[starts + LOOKBACK - 1 + HORIZON]
    return Xw, yw, starts