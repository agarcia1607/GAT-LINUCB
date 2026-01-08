from pydantic import validate_call
import pandas as pd
import os

class DownloadPrices() :
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        from dotenv import load_dotenv

        load_dotenv()
        self.start_date = os.getenv("START_DATE", "2015-01-01")
        self.data_raw_dir = os.getenv("DATA_RAW_DIR", "data/raw")
        self.logs_dir = os.getenv("LOGS_DIR", "logs")
        self.out_parquet = os.path.join(
            self.data_raw_dir,
            os.getenv("PRICES_DAILY_FILE", "prices_daily_adjclose.parquet")
        )
        self.out_csv = self.out_parquet.replace(".parquet", ".csv")
        self.report_csv = os.path.join(
            self.logs_dir,
            os.getenv("DOWNLOAD_REPORT_FILE", "download_report.csv")
        )

    def main(self):
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        tickers = self.load_tickers(ART_TICKERS)
        print(f"[INFO] Tickers loaded: {len(tickers)}")

        # Download
        # auto_adjust=False to keep 'Adj Close' column in output
        df = yf.download(
            tickers=tickers,
            start=START_DATE,
            auto_adjust=False,
            group_by="column",
            progress=True,
            threads=True,
        )

        if df.empty:
            raise RuntimeError("Downloaded dataframe is empty. Check internet / Yahoo availability / tickers.")

        # Extract Adj Close
        # yf returns columns: ('Adj Close', ticker) etc when multiple tickers
        report_rows = []
        adj = None

        if isinstance(df.columns, pd.MultiIndex):
            if "Adj Close" not in df.columns.get_level_values(0):
                raise RuntimeError("Adj Close not found in downloaded data. Columns: " + str(df.columns.levels[0]))
            adj = df["Adj Close"].copy()
        else:
            # single ticker case (not expected, but handle)
            if "Adj Close" not in df.columns:
                raise RuntimeError("Adj Close not found for single ticker download.")
            adj = df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

        # Clean index
        adj.index = pd.to_datetime(adj.index)
        adj = adj.sort_index()
        adj = adj[~adj.index.duplicated(keep="last")]

        # Build report per ticker
        for t in tickers:
            if t not in adj.columns:
                report_rows.append({
                    "ticker": t,
                    "status": "missing_column",
                    "start_found": None,
                    "end_found": None,
                    "n_rows_nonnull": 0,
                    "pct_missing": 1.0,
                    "notes": "Ticker not present in Adj Close columns"
                })
                continue

            s = adj[t]
            nonnull = s.dropna()
            status = "ok" if len(nonnull) > 0 else "all_nan"
            start_found = nonnull.index.min().date().isoformat() if len(nonnull) else None
            end_found = nonnull.index.max().date().isoformat() if len(nonnull) else None
            n_rows_nonnull = int(nonnull.shape[0])
            pct_missing = float(s.isna().mean()) if len(s) else 1.0

            report_rows.append({
                "ticker": t,
                "status": status,
                "start_found": start_found,
                "end_found": end_found,
                "n_rows_nonnull": n_rows_nonnull,
                "pct_missing": round(pct_missing, 6),
                "notes": ""
            })

        report = pd.DataFrame(report_rows).sort_values(["status", "pct_missing", "ticker"], ascending=[True, True, True])
        report.to_csv(REPORT_CSV, index=False)

        # Save raw adjclose (keep all tickers for now; cleaning comes in Paso 3/5)
        saved_path = safe_to_parquet(adj, OUT_PARQUET, OUT_CSV)

        # Update metadata
        update_metadata(download_done=True, saved_path=saved_path, n_tickers=len(tickers), n_dates=adj.shape[0])

        # Quick validation prints
        ok_count = (report["status"] == "ok").sum()
        print(f"[INFO] Download complete. Dates: {adj.index.min().date()} -> {adj.index.max().date()} | rows={adj.shape[0]}")
        print(f"[INFO] Tickers status: ok={ok_count} / total={len(tickers)}")
        print(f"[INFO] Saved: {saved_path}")
        print(f"[INFO] Report: {REPORT_CSV}")

    @validate_call
    def get_tickers(self) -> list[str]:
        """Fetch S&P 500 tickers from TICKERS_URL and return the Symbol list."""
        import requests
        import pandas as pd
        from io import StringIO

        url = os.getenv("TICKERS_URL")
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        if self.verbose:
            print(f"[INFO] Fetching tickers from: {url}")

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        html = StringIO(resp.text)

        sp500 = pd.read_html(html)[0]
        tickers = sp500["Symbol"].tolist()

        if self.verbose:
            print(f"[INFO] Parsed {len(tickers)} tickers.")

        return tickers

    @validate_call
    def load_tickers(self, path: str) -> list[str]:
        import json

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        tickers = []
        for _, group in d["groups"].items():
            tickers.extend(group)
        # unique preserving order
        seen = set()
        tickers_unique = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                tickers_unique.append(t)
        return tickers_unique

    """
    @validate_call
    def safe_to_parquet(self, df: pd.DataFrame, parquet_path: str, csv_path: str) -> str:
        try:
            df.to_parquet(parquet_path, index=True)
            return parquet_path
        except Exception as e:
            print(f"[WARN] Parquet failed ({e}). Falling back to CSV.")
            df.to_csv(csv_path, index=True)
            return csv_path

    @validate_call
    def update_metadata(self, download_done: bool, saved_path: str, n_tickers: int, n_dates: int):
        return
    
        if not os.path.exists(META_JSON):
            return
        with open(META_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta.setdefault("run", {})
        meta["run"]["download_timestamp_local"] = datetime.now().isoformat(timespec="seconds")
        meta["run"]["download_saved_path"] = saved_path
        meta["run"]["n_tickers_candidate"] = n_tickers
        meta["run"]["n_dates_downloaded"] = n_dates

        meta.setdefault("status", {})
        meta["status"]["download_done"] = download_done

        with open(META_JSON, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    """
