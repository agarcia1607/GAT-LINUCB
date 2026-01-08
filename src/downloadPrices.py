from pydantic import validate_call
import pandas as pd
from pathlib import Path
import os

class DownloadPrices() :
    def __init__(self, verbose: bool = False):

        self.verbose = verbose

        return

    def main(self):
        print("[START] Download prices")

        tickers = self._get_tickers()
        print("\t[DONE] get_tickers()")

        self._download_tickers(tickers)
        print("\t[DONE] download_tickers()")

        print("[FINISHED] Download prices")

        return

    @validate_call
    def _get_tickers(self) -> list[str]:
        """Fetch tickers from TICKERS_URL, save them to JSON in ARTIFACTS_DIR, and return the list."""

        import requests
        from io import StringIO
        import pandas as pd
        import json

        url = os.getenv("TICKERS_URL")
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        if self.verbose:
            print(f"\t[INFO] Fetching tickers from: {url}")

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        # Use a file-like buffer for pandas HTML parsing.
        html = StringIO(resp.text)

        # First HTML table is expected to include the Symbol column.
        sp500 = pd.read_html(html)[0]
        tickers = sp500["Symbol"].tolist()

        # Persist the downloaded tickers in the artifacts directory.
        tickers_dir = os.getenv("TICKERS_DIR")
        if tickers_dir is None:
            raise RuntimeError("TICKERS_DIR environment variable not set.")
        save_json_path = Path(tickers_dir) / Path("tickers.json")
        with open(save_json_path, "w", encoding="utf-8") as f:
            data = {
                "tickers": tickers
            }
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"\t[INFO] Tickers saved to: {save_json_path}")

        if self.verbose:
            print(f"\t[INFO] Parsed {len(tickers)} tickers.")

        return tickers
    
    def _download_tickers(self, tickers: list[str]) -> bool:
        """Download historical prices for tickers, normalize to long format, and save as parquet."""
        
        from yfinance import download

        start_date = os.getenv("START_DATE")
        if start_date is None:
            raise RuntimeError("START_DATE environment variable not set.")

        if self.verbose:
            print(f"\t[INFO] Downloading {len(tickers)} tickers from {start_date}...")

        df = download(
            tickers=tickers,
            start=start_date,
            auto_adjust=False,
            group_by="column",
            progress=True,
            threads=True,
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
            print(f"\t[INFO] Download complete. Dates: {long.index.min().date()} -> {long.index.max().date()} | rows={long.shape[0]}")

        tickers_dir = os.getenv("TICKERS_DIR")
        if tickers_dir is None:
            raise RuntimeError("TICKERS_DIR environment variable not set.")
        # Persist the normalized dataset for later steps.
        long.to_parquet(Path(tickers_dir) / Path("prices.parquet"), index=True)

        if self.verbose:
            print(f"\t[INFO] Saved to: {Path(tickers_dir) / Path('prices.parquet')}")

        return True
