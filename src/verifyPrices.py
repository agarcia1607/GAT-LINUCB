from pydantic import validate_call
from pathlib import Path
import os

class VerifyPrices:
    @validate_call
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        return
    
    def main(self):
        print("[START] Verify prices")

        print("\t[RUNNING] verify_coverage()")
        self._verify_coverage()
        print("\t[DONE] _verify_coverage()")

        print("[FINISHED] Verify prices")

        return
    
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
