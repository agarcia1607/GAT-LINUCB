import pandas as pd

from src.lib.filtration import FiltrationSpec, iter_filtration

def main():
    print("[INFO] Loading weekly returns...")
    df = pd.read_parquet("data/processed/weekly_returns.parquet")

    print("[INFO] Data shape:", df.shape)
    print("[INFO] Date range:", df.index.min().date(), "->", df.index.max().date())

    spec = FiltrationSpec(
        W_corr=24,
        W_mom=4,
        start_date="2023-01-01",
    )

    print("[INFO] Iterating filtration...")

    it = iter_filtration(df, spec)
    first = next(it)

    print("\n=== FIRST SNAPSHOT (TEST) ===")
    print("snapshot t          :", first.t.date())
    print("F_{t-1} ends at     :", first.hist_until.date())
    print("corr window         :", 
          first.window_corr.index.min().date(),
          "->",
          first.window_corr.index.max().date())
    print("momentum window     :", 
          first.window_mom.index.min().date(),
          "->",
          first.window_mom.index.max().date())
    print("============================\n")

if __name__ == "__main__":
    main()
