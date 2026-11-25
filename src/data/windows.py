"""windows.py -- scaling + sliding windows (shared by Primary and Fallback)."""
import numpy as np
import pandas as pd

TARGET = "Global_active_power"    #target variable jisse aage predict krenge hum
LOOKBACK = 60     # kitni readings pdni h
HORIZON = 15   # kitne mins aage wala predict krne h
# f is raw data here name ki jgh jo b str se aayega means train or val wo lg jyega

def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(f"data/processed/{name}.csv",
                       index_col="timestamp", parse_dates=True)
#timestamp ko index bna do aur date ko parse krdo mtlb abh index.minutes se mins access hojyenge parse date se hours mins alg alg hojyenge
#dataframe aaya train_df m gya or dict return krega

def fit_scaler(train_df: pd.DataFrame) -> dict:
    s = train_df[TARGET]                       # pandas skips NaN
    return {"mean": float(s.mean()), "std": float(s.std())}

#return kia mean or standard deviation
def make_features(df: pd.DataFrame, scaler: dict):
    y = ((df[TARGET].values - scaler["mean"]) / scaler["std"]).astype("float32") #Values ko 32-bit floating point format mein convert karta hai. km memory leta h
    hour = df.index.hour.values + df.index.minute.values / 60.0   #hour m convert kia kuki time of day pr depend kr skta h,decimal m bdla
    sin = np.sin(2 * np.pi * hour / 24).astype("float32")   # time of day,
    cos = np.cos(2 * np.pi * hour / 24).astype("float32")   # circular encoding ki kuki 23.00 or 00 bohooot dur h aise lgenge
    X = np.stack([y, sin, cos], axis=1)                     # shape (T, 3)
    return X, y
    # teeno ko combine krke return power sin or cos 
#dataframe ko usable features m covert kia  or scaler m train se calculated mean or std
   


#stride =1 mtlb ek minute se window shift hogi
def make_windows(df: pd.DataFrame, scaler: dict, stride: int = 1):
    """Returns Xw (N, 60, 3), yw (N,), starts (N,). Windows with NaN are dropped."""
    X, y = make_features(df, scaler)   #function call hua or x or y m store kia
    T = len(y)  #total number of timestamps
    n = T - LOOKBACK - HORIZON + 1  #possible windows kitni bnri
    starts = np.arange(0, n, stride)
# hr window ke start index ko calculate kia, hr window 1 row se aage khiskegi
    nan = np.isnan(y).astype(np.int32)
    # y m kitni value nan h
    c = np.concatenate([[0], np.cumsum(nan)]) 
 # fast NaN counting hr position pr nan ho tou 1 wrna 0
    nan_in_input = c[starts + LOOKBACK] - c[starts]
    # kisi range m kitne nan h
    nan_in_target = nan[starts + LOOKBACK - 1 + HORIZON]
    starts = starts[(nan_in_input == 0) & (nan_in_target == 0)]
    
    
    #wo window left jinke start m b nan 0 or baadme b 0
    #starts filter hogys abh pehle start m starting rows daalkr
    #  usse colum bnadia usme 0,1,2 add kia or un location ki windows ki value uthadi 3*3 matrix bn gyi 
    idx = starts[:, None] + np.arange(LOOKBACK)[None, :]
    Xw = X[idx]
    yw = y[starts + LOOKBACK - 1 + HORIZON]
    return Xw, yw, starts


# ---------------- tabular features for the Fallback (Ridge) and the reliability model ----------------
LAGS = [1, 2, 3, 5, 10, 15, 30, 45, 60]     # minutes back from "now"


def window_to_features(Xw):
    """Xw: (N, 60, 3) -> (N, 14). Same windows as the Primary => fair comparison."""
    load = Xw[:, :, 0]
    cols = [load[:, -k] for k in LAGS]                   # lag features
    cols += [load[:, -15:].mean(1),                      # 15-min mean
             load.mean(1),                               # 60-min mean
             load[:, -15:].std(1)]                       # 15-min variability
    cols += [Xw[:, -1, 1], Xw[:, -1, 2]]                 # time of day (sin, cos)
    return np.stack(cols, axis=1).astype("float32")