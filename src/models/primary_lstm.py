"""primary_lstm.py -- Step 3. Train the Primary Bi-LSTM forecaster.
Run:  python -m src.models.primary_lstm
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data.windows import load_split, fit_scaler, make_windows

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_DIR = Path("models")


class BiLSTM(nn.Module):
    def __init__(self, n_feat=3, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)                  # h_n: (2, B, hidden)
        h = torch.cat([h_n[0], h_n[1]], dim=1)      # forward + backward states
        return self.head(h).squeeze(-1)


def mae_kw(pred, true, std):
    return float(np.mean(np.abs(pred - true)) * std)   # back to kW


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    train_df, val_df = load_split("train"), load_split("val")

    scaler = fit_scaler(train_df)                   # TRAIN only -> no leakage
    json.dump(scaler, open(MODEL_DIR / "scaler.json", "w"))
    std = scaler["std"]

    Xtr, ytr, _ = make_windows(train_df, scaler, stride=10)   # stride: speed
    Xva, yva, _ = make_windows(val_df, scaler, stride=5)
    print(f"train windows: {len(Xtr):,}   val windows: {len(Xva):,}")

    # --- baseline: persistence (predict "same as now") ---
    persist = Xva[:, -1, 0]
    print(f"[baseline] persistence val MAE: {mae_kw(persist, yva, std):.4f} kW")
    print(f"[baseline] mean-predictor val MAE: {mae_kw(0*yva, yva, std):.4f} kW")

    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=256, shuffle=True)
    Xva_t = torch.from_numpy(Xva)

    model = BiLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best = float("inf")
    for epoch in range(1, 9):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).numpy()
        val_mae = mae_kw(pv, yva, std)
        print(f"epoch {epoch}  val MAE: {val_mae:.4f} kW")
        if val_mae < best:                          # keep best epoch only
            best = val_mae
            torch.save(model.state_dict(), MODEL_DIR / "primary_bilstm.pt")

    print(f"best val MAE: {best:.4f} kW  (saved to models/primary_bilstm.pt)")