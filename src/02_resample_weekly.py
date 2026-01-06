import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = os.path.join(os.getenv("DATA_RAW_DIR","data/raw"), os.getenv("PRICES_DAILY_FILE","prices_daily_adjclose.parquet"))
OUT_DIR = os.getenv("DATA_PROCESSED_DIR","data/processed")
OUT_PATH = os.path.join(OUT_DIR, os.getenv("PRICES_WEEKLY_FILE","prices_weekly_adjclose.parquet"))

FREQ = os.getenv("WEEKLY_FREQ","W-FRI")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(RAW_PATH)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Weekly last observation ending on Friday
    weekly = df.resample(FREQ).last()

    weekly.to_parquet(OUT_PATH)

    print("[INFO] weekly saved:", OUT_PATH)
    print("[INFO] weekly shape:", weekly.shape)
    print("[INFO] weekly date_min:", weekly.index.min().date(), "date_max:", weekly.index.max().date())
    print("[INFO] weekly freq sample head:", list(weekly.index[:3].date))
    print("[INFO] weekly missing total:", round(float(weekly.isna().mean().mean()), 6))

if __name__ == "__main__":
    main()
