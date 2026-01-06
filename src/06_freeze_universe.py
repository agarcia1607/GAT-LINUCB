from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def main():
    in_path = Path("data/processed/weekly_returns.parquet")
    out_dir = Path("artifacts/snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    df = df.sort_index()

    tickers = list(df.columns)
    node_id = {t: i for i, t in enumerate(tickers)}
    id_node = {str(i): t for i, t in enumerate(tickers)}  # keys as str for JSON

    payload = {
        "N": len(tickers),
        "tickers_in_order": tickers,
        "ticker_to_id": node_id,
        "id_to_ticker": id_node,
        "source": str(in_path),
    }

    out_path = out_dir / "node_map.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[INFO] node map saved:", out_path)
    print("[INFO] N =", payload["N"])
    print("[INFO] first 5 tickers:", tickers[:5])

if __name__ == "__main__":
    main()
