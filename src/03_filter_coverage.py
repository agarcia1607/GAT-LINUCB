import os, json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

IN_PATH = os.path.join(os.getenv("DATA_PROCESSED_DIR","data/processed"),
                       os.getenv("PRICES_WEEKLY_FILE","prices_weekly_adjclose.parquet"))

ART_DIR = os.getenv("ARTIFACTS_DIR","artifacts")
LOG_DIR = os.getenv("LOGS_DIR","logs")
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

COVERAGE_RULE = float(os.getenv("COVERAGE_RULE","0.95"))
OUT_TICKERS = os.path.join(ART_DIR, "tickers_final.json")
OUT_REPORT = os.path.join(LOG_DIR, os.getenv("COVERAGE_REPORT_FILE","coverage_weekly.csv"))

def main():
    w = pd.read_parquet(IN_PATH)
    w.index = pd.to_datetime(w.index)
    w = w.sort_index()

    # Coverage on the existing W-FRI calendar in the data (stable and simple)
    total_weeks = w.shape[0]
    coverage = (w.notna().sum(axis=0) / total_weeks).sort_values(ascending=False)

    first_week = w.apply(lambda s: s.first_valid_index())
    last_week  = w.apply(lambda s: s.last_valid_index())
    n_present  = w.notna().sum(axis=0)

    report = pd.DataFrame({
        "ticker": coverage.index,
        "coverage": coverage.values,
        "n_weeks_present": n_present.loc[coverage.index].values,
        "n_weeks_total": total_weeks,
        "first_week": first_week.loc[coverage.index].values,
        "last_week": last_week.loc[coverage.index].values,
    })
    report["keep"] = report["coverage"] >= COVERAGE_RULE
    report.to_csv(OUT_REPORT, index=False)

    tickers_keep = report.loc[report["keep"], "ticker"].tolist()

    with open(OUT_TICKERS, "w", encoding="utf-8") as f:
        json.dump({
            "coverage_rule": COVERAGE_RULE,
            "n_weeks_total": total_weeks,
            "n_keep": len(tickers_keep),
            "tickers": tickers_keep
        }, f, indent=2)

    print(f"[INFO] coverage rule: {COVERAGE_RULE}")
    print(f"[INFO] total tickers: {w.shape[1]}")
    print(f"[INFO] keep tickers: {len(tickers_keep)}")
    print(f"[INFO] saved tickers_final: {OUT_TICKERS}")
    print(f"[INFO] saved report: {OUT_REPORT}")

    # Quick sanity: show worst 5
    worst = report.sort_values("coverage").head(5)[["ticker","coverage","keep"]]
    print("[INFO] worst_5:")
    print(worst.to_string(index=False))

if __name__ == "__main__":
    main()
