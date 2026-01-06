from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.lib.io_universe import load_weekly_returns_aligned
from src.lib.filtration import FiltrationSpec, iter_filtration
from src.lib.correlation import compute_corr
from src.lib.knn_graph import knn_from_corr
from src.lib.symmetrize import symmetrize_edges
from src.lib.edge_attr import build_edge_tensors

def main():
    df = load_weekly_returns_aligned()
    spec = FiltrationSpec(W_corr=24, W_mom=4, start_date="2023-01-01")

    rows = []
    for i, sl in enumerate(iter_filtration(df, spec)):
        C = compute_corr(sl.window_corr)
        edges_knn = knn_from_corr(C, k=8)
        edges_sym = symmetrize_edges(edges_knn)

        # raw
        _, edge_raw, _ = build_edge_tensors(edges_sym, C, normalize=False, clip_value=None)
        # normalized
        _, edge_norm, stats = build_edge_tensors(edges_sym, C, normalize=True, clip_value=3.0)

        rows.append({
            "date": str(sl.t.date()),
            "E": int(edge_norm.shape[0]),
            "mu_t": float(stats["mu_t"]),
            "sigma_t": float(stats["sigma_t"]),
            "raw_mean": float(edge_raw.mean()),
            "raw_mean_abs": float(np.mean(np.abs(edge_raw))),
            "raw_neg_frac": float(np.mean(edge_raw < 0.0)),
            "norm_mean": float(edge_norm.mean()),
            "norm_mean_abs": float(np.mean(np.abs(edge_norm))),
            "norm_neg_frac": float(np.mean(edge_norm < 0.0)),
            "norm_min": float(edge_norm.min()),
            "norm_max": float(edge_norm.max()),
        })

        if i == 9:
            break

    diag = pd.DataFrame(rows)
    out = Path("artifacts/snapshots/edge_attr_norm_preview_10.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(out, index=False)

    print("[INFO] saved:", out)
    print(diag)

if __name__ == "__main__":
    main()
