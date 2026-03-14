import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TICKERS_DIR = Path(os.getenv("TICKERS_DIR", "artifacts/tickers"))
IN_PATH = TICKERS_DIR / "prices.parquet"

PROC_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
OUT_PATH = PROC_DIR / os.getenv("PRICES_WEEKLY_FILE", "prices_weekly_adjclose.parquet")

WEEKLY_RULE = os.getenv("WEEKLY_RULE", "W-FRI")


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if not IN_PATH.exists():
        raise FileNotFoundError(f"No existe el input esperado: {IN_PATH}")

    df = pd.read_parquet(IN_PATH)

    required_cols = {"TICKER", "ADJ_CLOSE"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {IN_PATH}: {missing}")

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.set_index("DATE")
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = "DATE"

    prices = df.reset_index()[["DATE", "TICKER", "ADJ_CLOSE"]].copy()

    wide = prices.pivot_table(
        index="DATE",
        columns="TICKER",
        values="ADJ_CLOSE",
        aggfunc="last"
    ).sort_index()

    weekly = wide.resample(WEEKLY_RULE).last()
    weekly = weekly.sort_index()
    weekly.index = pd.to_datetime(weekly.index)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(OUT_PATH)

    print(f"[INFO] input long parquet: {IN_PATH}")
    print(f"[INFO] output weekly parquet: {OUT_PATH}")
    print(f"[INFO] weekly shape: {weekly.shape}")
    print(f"[INFO] weekly date range: {weekly.index.min()} -> {weekly.index.max()}")


if __name__ == "__main__":
    main()