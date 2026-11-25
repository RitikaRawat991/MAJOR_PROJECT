"""reliability.py -- Step 5. Reliability score c for the Primary model.
c = 1 - min(1, w / w_ref),  w = width of a 90% prediction interval.
Run:  python -m src.framework.reliability
"""
from pathlib import Path
import json
import numpy as np
import torch
import joblib
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from src.data.windows import load_split, make_windows, window_to_features
from src.models.primary_lstm import BiLSTM, predict

MODEL_DIR = Path("models")
COVERAGE = 0.90      # target coverage of the interval
W_REF_Q = 0.95       # w_ref = this quantile of widths on nominal data
EPS = 1e-3


def scale_features(Xw, pred):
    return np.column_stack([window_to_features(Xw), pred]).astype("float32")


class Reliability:
    def fit(self, Xw, y, pred):
        err = np.abs(y - pred)
        self.scale_model = Ridge(alpha=1.0).fit(scale_features(Xw, pred), err)
        s = self._scale(Xw, pred)
        # k: |err| <= k*s for 90% of samples (normalised conformal)
        self.k = float(np.quantile(err / s, COVERAGE))
        self.w_ref = float(np.quantile(self.width(Xw, pred), W_REF_Q))
        return self

    def _scale(self, Xw, pred):
        return np.maximum(self.scale_model.predict(scale_features(Xw, pred)), EPS)

    def width(self, Xw, pred):
        return 2 * self.k * self._scale(Xw, pred)

    def score(self, Xw, pred):
        return 1 - np.minimum(1, self.width(Xw, pred) / self.w_ref)

    def save(self, path):                      # save plain parts (safe to reload)
        joblib.dump({"scale_model": self.scale_model, "k": self.k,
                     "w_ref": self.w_ref}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        r = cls()
        r.scale_model, r.k, r.w_ref = d["scale_model"], d["k"], d["w_ref"]
        return r


if __name__ == "__main__":
    scaler = json.load(open(MODEL_DIR / "scaler.json"))
    std = scaler["std"]
    lstm = BiLSTM()
    lstm.load_state_dict(torch.load(MODEL_DIR / "primary_bilstm.pt"))
    lstm.eval()

    Xva, yva, _ = make_windows(load_split("val"), scaler, stride=5)
    pva = predict(lstm, Xva)
    h = len(Xva) // 2                                  # chronological halves
    XA, yA, pA = Xva[:h], yva[:h], pva[:h]
    XB, yB, pB = Xva[h:], yva[h:], pva[h:]

    rel = Reliability().fit(XA, yA, pA)                # fit on A only
    rel.save(MODEL_DIR / "reliability.joblib")

    # ---- validation on held-out half B ----
    err = np.abs(yB - pB)
    w = rel.width(XB, pB)
    c = rel.score(XB, pB)
    cover = float(np.mean(err <= w / 2))
    rho = spearmanr(w, err).correlation
    print(f"k = {rel.k:.3f}   w_ref = {rel.w_ref:.3f} (scaled units)")
    print(f"[check 1] coverage on B: {cover*100:.1f}%   (target 90%)")
    print(f"[check 2] Spearman(width, |error|) on B: {rho:.3f}")
    print(f"[check 3] c on nominal B: mean {c.mean():.3f}, "
          f"p5 {np.quantile(c,0.05):.3f}, median {np.median(c):.3f}")

    # ---- check 4: does c drop when inputs are noisy? (F3-style fault) ----
    rng = np.random.default_rng(42)
    print("\nnoise_sigma(kW)   MAE(kW)   mean c")
    for sigma_kw in [0.0, 0.25, 0.5, 1.0]:
        Xn = XB.copy()
        Xn[:, :, 0] += rng.normal(0, sigma_kw / std, Xn[:, :, 0].shape)
        pn = predict(lstm, Xn)
        mae = float(np.mean(np.abs(pn - yB)) * std)
        print(f"{sigma_kw:15.2f}   {mae:7.4f}   {rel.score(Xn, pn).mean():.3f}")