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

        print("[FINISHED] Verify prices")

        return
    
    def _verify_coverage(self) -> None:
        from pandas import read_parquet
        from json import load

        TICKERS_DIR = os.getenv("TICKERS_DIR")
        if TICKERS_DIR is None:
            raise RuntimeError("TICKERS_DIR environment variable not set.")
        prices_raw_path = Path(TICKERS_DIR) / Path("prices_raw.parquet")
        prices = read_parquet(prices_raw_path)
        prices = prices.dropna(axis=0, how="any")

        tickers_path = Path(TICKERS_DIR) / Path("tickers.json")
        with tickers_path.open("r") as f:
            data = load(f)
            tickers = data["tickers"]

        MINIMUM_COVERAGE = os.getenv("MINIMUM_COVERAGE")
        if MINIMUM_COVERAGE is None:
            raise RuntimeError("MINIMUM_COVERAGE environment variable not set.")
        MINIMUM_COVERAGE = int(MINIMUM_COVERAGE)

        for ticker in tickers:
            prices_ticker = prices[prices.TICKER == ticker]
            if prices_ticker.shape[0] < MINIMUM_COVERAGE:
                prices = prices[prices.TICKER != ticker]

        return
