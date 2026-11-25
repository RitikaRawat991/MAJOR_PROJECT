"""
load_data.py  --  Step 2
Load raw UCI data -> clean -> chronological train/val/test split -> save CSVs.
Run from project root:  python -m src.data.load_data
"""
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/household_power_consumption.txt")
OUT = Path("data/processed")
TARGET = "Global_active_power"
MAX_GAP = 10          # interpolate gaps up to 10 minutes only


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, sep=";", na_values="?", low_memory=False)
    # merge Date + Time into one timestamp (day-first format)
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("timestamp").sort_index()
    return df.astype("float32")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1) make sure every minute exists (missing minutes become NaN rows)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="min")
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    # 2) remember which rows were originally missing (useful later as an
    #    "input-quality" signal for F3 faults)
    df["was_missing"] = df[TARGET].isna().astype("int8")

    # 3) fill SHORT gaps only; long gaps stay NaN (honest: we don't invent data)
    df = df.interpolate(method="linear", limit=MAX_GAP, limit_area="inside")
    return df


def chrono_split(df: pd.DataFrame, train=0.70, val=0.15):
    n = len(df)
    i_tr, i_va = int(n * train), int(n * (train + val))
    return df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]   # NO shuffling


if __name__ == "__main__":
    df = clean(load_raw())
    print(f"rows: {len(df):,}")
    print(f"originally missing: {df['was_missing'].mean()*100:.2f}%")
    print(f"still NaN after short-gap fill: {df[TARGET].isna().mean()*100:.2f}%")

    train, val, test = chrono_split(df)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT / f"{name}.csv")
        print(f"{name:5s}: {len(part):>9,} rows  {part.index.min()} -> {part.index.max()}")