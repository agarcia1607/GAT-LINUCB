from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.lib.io_universe import load_weekly_returns_aligned
from src.lib.filtration import FiltrationSpec, iter_filtration
from src.lib.correlation import compute_corr

def offdiag(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    return x[~np.eye(n, dtype=bool)]

def main():
    df = load_weekly_returns_aligned()
    spec = FiltrationSpec(W_corr=24, W_mom=4, start_date="2023-01-01")

    rows = []
    for i, sl in enumerate(iter_filtration(df, spec)):
        C = compute_corr(sl.window_corr).to_numpy()
        od = offdiag(C)

        rows.append({
            "date": str(sl.t.date()),
            "full_mean": float(od.mean()),
            "full_mean_abs": float(np.abs(od).mean()),
            "full_neg_frac": float((od < 0).mean()),
            "full_min": float(od.min()),
            "full_max": float(od.max()),
        })

        if i == 9:  # preview 10 semanas
            break

    out = Path("artifacts/snapshots/full_corr_preview_10.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print("[INFO] saved:", out)
    print(pd.DataFrame(rows))

if __name__ == "__main__":
    main()
