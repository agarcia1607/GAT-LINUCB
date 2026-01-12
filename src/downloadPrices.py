from pydantic import validate_call
from pathlib import Path
import os

class DownloadPrices() :
    @validate_call
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        return

    def main(self):
        print("[START] Download prices")

        print("\t[RUNNING] get_tickers()")
        tickers = self._get_tickers()
        print("\t[DONE] get_tickers()")

        print("\t[RUNNING] download_tickers()")
        self._download_tickers(tickers)
        print("\t[DONE] download_tickers()")

        print("\t[RUNNING] verify_coverage()")
        self._verify_coverage()
        print("\t[DONE] verify_coverage()")

        print("[FINISHED] Download prices")

        return

    @validate_call
    def _get_tickers(self) -> list[str]:
        """Fetch tickers from TICKERS_URL, save them to JSON in ARTIFACTS_DIR, and return the list."""

        import requests
        from io import StringIO
        import pandas as pd
        import json

        URL = os.getenv("TICKERS_URL")
        if URL is None:
            raise RuntimeError("TICKERS_URL environment variable not set.")

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        if self.verbose:
            print(f"\t\t[INFO] Fetching tickers from: {URL}")

        resp = requests.get(URL, headers=headers)
        resp.raise_for_status()
        # Use a file-like buffer for pandas HTML parsing.
        html = StringIO(resp.text)

        # First HTML table is expected to include the Symbol column.
        sp500 = pd.read_html(html)[0]
        tickers: list[str] = sp500["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]

        # Persist the downloaded tickers in the artifacts directory.
        TICKERS_DIR = os.getenv("TICKERS_DIR")
        if TICKERS_DIR is None:
            raise RuntimeError("TICKERS_DIR environment variable not set.")
        save_json_path = Path(TICKERS_DIR) / Path("tickers.json")
        with open(save_json_path, "w", encoding="utf-8") as f:
            data = {
                "tickers": tickers
            }
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"\t\t[INFO] Tickers saved to: {save_json_path}")

        if self.verbose:
            print(f"\t\t[INFO] Parsed {len(tickers)} tickers.")

        return tickers
    
    def _download_tickers(self, tickers: list[str]) -> bool:
        """Download historical prices for tickers, normalize to long format, and save as parquet."""

        from yfinance import download

        START_DATE = os.getenv("START_DATE")
        if START_DATE is None:
            raise RuntimeError("START_DATE environment variable not set.")

        INTERVAL = os.getenv("INTERVAL")
        if INTERVAL is None:
            raise RuntimeError("INTERVAL environment variable not set.")

        if self.verbose:
            print(f"\t\t[INFO] Downloading {len(tickers)} tickers from {START_DATE}...")

        df = download(
            tickers=tickers,
            start=START_DATE,
            auto_adjust=False,
            group_by="column",
            progress=True,
            threads=True,
            interval=INTERVAL
        )

        # Convert wide multi-index columns into a long, row-per-ticker format.
        long = (
            df.stack(level=1, future_stack=True)
            .rename_axis(index=["date", "ticker"])
            .reset_index()
        )
        # Normalize column names for downstream consistency.
        long.columns = [str(c).upper().replace(" ", "_") for c in long.columns]
        long.set_index("DATE", inplace=True)

        if self.verbose:
            print(f"\t\t[INFO] Download complete. Dates: {long.index.min().date()} -> {long.index.max().date()} | rows={long.shape[0]}")

        TICKERS_DIR = os.getenv("TICKERS_DIR")
        if TICKERS_DIR is None:
            raise RuntimeError("TICKERS_DIR environment variable not set.")
        prices_raw_path = Path(TICKERS_DIR) / Path("prices_raw.parquet")
        # Persist the normalized dataset for later steps.
        long.to_parquet(prices_raw_path, index=True)

        if self.verbose:
            print(f"\t\t[INFO] Saved to: {prices_raw_path}")

        return True
    
    @validate_call
    def _verify_coverage(self) -> None:
        """Filter tickers by minimum coverage and write the cleaned parquet."""
        from pandas import read_parquet
        from json import load

        TICKERS_DIR = os.getenv("TICKERS_DIR")
        if TICKERS_DIR is None:
            raise RuntimeError("TICKERS_DIR environment variable not set.")
        prices_raw_path = Path(TICKERS_DIR) / Path("prices_raw.parquet")
        if self.verbose:
            print(f"\t\t[INFO] loading prices: {prices_raw_path}")
        prices = read_parquet(prices_raw_path)
        # Keep only rows with complete data.
        prices = prices.dropna(axis=0, how="any")

        tickers_path = Path(TICKERS_DIR) / Path("tickers.json")
        if self.verbose:
            print(f"\t\t[INFO] loading tickers: {tickers_path}")
        with tickers_path.open("r") as f:
            data = load(f)
            tickers = data["tickers"]

        MINIMUM_COVERAGE = os.getenv("MINIMUM_COVERAGE")
        if MINIMUM_COVERAGE is None:
            raise RuntimeError("MINIMUM_COVERAGE environment variable not set.")
        MINIMUM_COVERAGE = int(MINIMUM_COVERAGE)
        if self.verbose:
            print(f"\t\t[INFO] minimum coverage: {MINIMUM_COVERAGE}")
            print(f"\t\t[INFO] tickers to check: {len(tickers)}")

        for ticker in tickers:
            prices_ticker = prices[prices.TICKER == ticker]
            # Drop tickers with insufficient history.
            if prices_ticker.shape[0] < MINIMUM_COVERAGE:
                prices = prices[prices.TICKER != ticker]

        prices_path = Path(TICKERS_DIR) / Path("prices.parquet")
        if self.verbose:
            print(f"\t\t[INFO] writing filtered prices: {prices_path}")
        prices.to_parquet(prices_path)
        if self.verbose:
            print(f"\t\t[INFO] rows kept: {prices.shape[0]}")
            print(f"\t\t[INFO] tickers kept: {prices['TICKER'].nunique()}")

        return
