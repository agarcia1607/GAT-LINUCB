from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.lib.io_universe import load_weekly_returns_aligned
from src.lib.filtration import FiltrationSpec, iter_filtration
from src.lib.features import build_X_t

def main():
    df = load_weekly_returns_aligned()

    spec = FiltrationSpec(W_corr=24, W_mom=4, start_date="2023-01-01")

    out_dir = Path("artifacts/snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, sl in enumerate(iter_filtration(df, spec)):
        X = build_X_t(sl.window_corr, sl.window_mom)

        # Validaciones obligatorias (2.3)
        assert X.shape == (df.shape[1], 2), f"Bad X shape at {sl.t}: {X.shape}"
        assert int(X.isna().sum().sum()) == 0, f"NaNs in X at {sl.t}"

        rows.append({
            "date": str(sl.t.date()),
            "mom4_mean": float(X["mom4"].mean()),
            "vol24_mean": float(X["vol24"].mean()),
            "mom4_min": float(X["mom4"].min()),
            "mom4_max": float(X["mom4"].max()),
            "vol24_min": float(X["vol24"].min()),
            "vol24_max": float(X["vol24"].max()),
        })

        if i == 9:  # 10 snapshots de preview
            break

    diag = pd.DataFrame(rows)
    diag_path = out_dir / "X_preview_10.csv"
    diag.to_csv(diag_path, index=False)

    print("[INFO] saved:", diag_path)
    print(diag)

if __name__ == "__main__":
    main()
