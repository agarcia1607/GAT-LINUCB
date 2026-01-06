import os, json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

PROC_DIR = os.getenv("DATA_PROCESSED_DIR","data/processed")
ART_DIR  = os.getenv("ARTIFACTS_DIR","artifacts")
LOG_DIR  = os.getenv("LOGS_DIR","logs")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

WEEKLY_PRICES_PATH = os.path.join(PROC_DIR, os.getenv("PRICES_WEEKLY_FILE","prices_weekly_adjclose.parquet"))
TICKERS_FINAL_PATH = os.path.join(ART_DIR, "tickers_final.json")
RETURNS_OUT_PATH   = os.path.join(PROC_DIR, os.getenv("RETURNS_WEEKLY_FILE","weekly_returns.parquet"))

META_PATH = os.path.join(ART_DIR, "metadata.json")
SNAPSHOT_PATH = os.path.join(LOG_DIR, os.getenv("FINAL_SNAPSHOT_FILE","final_snapshot.txt"))

RETURNS_TYPE = os.getenv("RETURNS_TYPE","simple")  # simple or log
NAN_POLICY = os.getenv("NAN_POLICY_RETURNS","intersection_all_assets")  # intersection_all_assets or keep_nans

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if RETURNS_TYPE.lower() == "log":
        return (prices / prices.shift(1)).apply(lambda s: s.apply(lambda x: None if pd.isna(x) else x)).pipe(lambda df: df).applymap(lambda x: x)
    # simple
    return prices.pct_change()

def main():
    prices = pd.read_parquet(WEEKLY_PRICES_PATH).sort_index()
    with open(TICKERS_FINAL_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    tickers = d["tickers"]

    prices = prices[tickers]

    rets = prices.pct_change()

    # Apply NaN policy for "stable dataset"
    if NAN_POLICY == "intersection_all_assets":
        rets_clean = rets.dropna(axis=0, how="any")
    else:
        rets_clean = rets

    rets_clean.to_parquet(RETURNS_OUT_PATH)

    # Metadata update (non-destructive)
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

    meta.setdefault("params", {})
    meta["params"].update({
        "start_date": os.getenv("START_DATE","2015-01-01"),
        "freq": os.getenv("WEEKLY_FREQ","W-FRI"),
        "price_field": os.getenv("PRICE_FIELD","Adj Close"),
        "coverage_rule": float(os.getenv("COVERAGE_RULE","0.95")),
        "returns_type": RETURNS_TYPE,
        "nan_policy_returns": NAN_POLICY
    })

    meta.setdefault("run", {})
    meta["run"].update({
        "block1_completed_at_local": datetime.now().isoformat(timespec="seconds"),
        "weekly_prices_path": WEEKLY_PRICES_PATH,
        "weekly_returns_path": RETURNS_OUT_PATH,
        "tickers_final_path": TICKERS_FINAL_PATH,
        "n_assets_final": len(tickers),
        "n_weeks_prices": int(prices.shape[0]),
        "n_weeks_returns": int(rets_clean.shape[0]),
        "returns_date_min": str(rets_clean.index.min().date()),
        "returns_date_max": str(rets_clean.index.max().date()),
        "returns_nan_total": int(rets_clean.isna().sum().sum())
    })

    meta.setdefault("status", {})
    meta["status"].update({
        "download_done": True,
        "weekly_resample_done": True,
        "coverage_filter_done": True,
        "returns_done": True,
        "artifacts_saved": True
    })

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Human snapshot
    lines = []
    lines.append("BLOQUE 1 — FINAL SNAPSHOT")
    lines.append(f"Assets final: {len(tickers)}")
    lines.append(f"Weekly prices: {prices.index.min().date()} -> {prices.index.max().date()} | shape={prices.shape}")
    lines.append(f"Weekly returns ({NAN_POLICY}): {rets_clean.index.min().date()} -> {rets_clean.index.max().date()} | shape={rets_clean.shape}")
    lines.append(f"Total NaNs in returns: {rets_clean.isna().sum().sum()}")
    lines.append(f"Saved returns: {RETURNS_OUT_PATH}")
    lines.append(f"Saved tickers_final: {TICKERS_FINAL_PATH}")
    lines.append(f"Saved metadata: {META_PATH}")
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[INFO] returns saved:", RETURNS_OUT_PATH)
    print("[INFO] returns shape:", rets_clean.shape)
    print("[INFO] returns date_min:", rets_clean.index.min().date(), "date_max:", rets_clean.index.max().date())
    print("[INFO] returns total NaNs:", int(rets_clean.isna().sum().sum()))
    print("[INFO] snapshot saved:", SNAPSHOT_PATH)
    print("[INFO] metadata updated:", META_PATH)

if __name__ == "__main__":
    main()
