from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib.io_universe import load_weekly_returns_aligned, load_node_map
from src.lib.filtration import FiltrationSpec, iter_filtration
from src.lib.features import build_X_t
from src.lib.correlation import compute_corr
from src.lib.knn_graph import knn_from_corr
from src.lib.symmetrize import symmetrize_edges
from src.lib.edge_attr import build_edge_tensors


def _degrees_from_edge_index(edge_index: np.ndarray, N: int) -> np.ndarray:
    # edge_index shape (2, E), directed degrees out-degree
    deg = np.zeros(N, dtype=np.int64)
    src = edge_index[0]
    for u in src:
        deg[int(u)] += 1
    return deg


def save_snapshot_npz(out_dir: Path, date_str: str, X: np.ndarray, edge_index: np.ndarray,
                      edge_attr: np.ndarray, edge_attr_raw: np.ndarray, stats: dict) -> str:
    """
    Guarda snapshot como .npz:
      - X: (N, F)
      - edge_index: (2, E)
      - edge_attr: (E, 1)  (z-score clipped)
      - edge_attr_raw: (E, 1)  (corr signed)
      - stats: dict (mu_t, sigma_t, etc.) guardado como json string
    """
    fname = f"{date_str}.npz"
    path = out_dir / fname
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        edge_index=edge_index.astype(np.int64),
        edge_attr=edge_attr.astype(np.float32),
        edge_attr_raw=edge_attr_raw.astype(np.float32),
        stats_json=json.dumps(stats),
    )
    return fname


def main():
    # --- Config BLOQUE 2 (cerrado)
    spec = FiltrationSpec(W_corr=24, W_mom=4, start_date="2015-01-01")
    k = 8
    clip_value = 3.0
    normalize_edge_attr = True

    # --- Paths
    root_out = Path("artifacts/snapshots")
    snap_dir = root_out / "npz"
    root_out.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data (aligned)
    df = load_weekly_returns_aligned()
    node_map = load_node_map()
    tickers = node_map["tickers_in_order"]
    N = df.shape[1]
    assert N == int(node_map["N"])

    # --- Iterate filtration and build snapshots
    index_rows = []
    diag_rows = []

    for sl in iter_filtration(df, spec):
        t = sl.t
        date_str = str(t.date())

        # X_t
        X_df = build_X_t(sl.window_corr, sl.window_mom)
        # enforce order (defensive)
        X_df = X_df.loc[tickers]
        X = X_df.to_numpy(dtype=np.float32)  # (N, 2)
        F = X.shape[1]

        # C_t
        C = compute_corr(sl.window_corr)

        # kNN directed -> symmetrize
        edges_knn = knn_from_corr(C, k=k)
        edges_sym = symmetrize_edges(edges_knn)

        # edge_index + edge_attr_raw + edge_attr(z)
        # raw
        edge_index_raw, edge_attr_raw, _ = build_edge_tensors(
            edges_sym, C, normalize=False, clip_value=None
        )
        # normalized (z-score vs off-diagonal full, clipped)
        edge_index, edge_attr, stats = build_edge_tensors(
            edges_sym, C, normalize=normalize_edge_attr, clip_value=clip_value
        )

        # --- Validaciones obligatorias
        # dims
        assert X.shape == (N, 2), f"Bad X shape at {date_str}: {X.shape}"
        assert edge_index.shape[0] == 2, f"Bad edge_index shape at {date_str}: {edge_index.shape}"
        E = edge_index.shape[1]
        assert edge_attr.shape == (E, 1), f"Bad edge_attr shape at {date_str}: {edge_attr.shape}"
        assert edge_attr_raw.shape == (E, 1), f"Bad edge_attr_raw shape at {date_str}: {edge_attr_raw.shape}"
        assert np.isfinite(X).all(), f"NaNs in X at {date_str}"
        assert np.isfinite(edge_attr).all(), f"NaNs in edge_attr at {date_str}"
        assert np.isfinite(edge_attr_raw).all(), f"NaNs in edge_attr_raw at {date_str}"

        # graph non-empty and not almost complete
        assert E > 0, f"Empty graph at {date_str}"
        # Not almost complete (directed complete without self loops = N*(N-1))
        assert E < 0.6 * (N * (N - 1)), f"Graph too dense at {date_str}: E={E}"

        # no self-loops
        assert np.all(edge_index[0] != edge_index[1]), f"Self-loops found at {date_str}"

        # no isolated nodes (out-degree >= 1)
        deg = _degrees_from_edge_index(edge_index, N)
        assert int(deg.min()) >= 1, f"Isolated node found at {date_str}"

        # --- Save snapshot
        saved_name = save_snapshot_npz(
            snap_dir, date_str, X, edge_index, edge_attr, edge_attr_raw, stats
        )

        # --- Index + diagnostics
        index_rows.append({
            "date": date_str,
            "file": saved_name,
            "N": int(N),
            "F": int(F),
            "E": int(E),
        })

        diag_rows.append({
            "date": date_str,
            "E": int(E),
            "min_deg": int(deg.min()),
            "max_deg": int(deg.max()),
            "mean_deg": float(deg.mean()),
            "raw_mean": float(edge_attr_raw.mean()),
            "raw_mean_abs": float(np.mean(np.abs(edge_attr_raw))),
            "raw_neg_frac": float(np.mean(edge_attr_raw < 0.0)),
            "mu_t": float(stats.get("mu_t", 0.0)),
            "sigma_t": float(stats.get("sigma_t", 1.0)),
            "z_mean": float(edge_attr.mean()),
            "z_mean_abs": float(np.mean(np.abs(edge_attr))),
            "z_neg_frac": float(np.mean(edge_attr < 0.0)),
            "z_min": float(edge_attr.min()),
            "z_max": float(edge_attr.max()),
        })

    # --- Save index and diagnostics
    index_df = pd.DataFrame(index_rows).sort_values("date")
    diag_df = pd.DataFrame(diag_rows).sort_values("date")

    index_path = root_out / "snapshots_index.csv"
    diag_path = root_out / "snapshots_diag.csv"
    index_df.to_csv(index_path, index=False)
    diag_df.to_csv(diag_path, index=False)

    # --- Metadata (reproducibility)
    meta = {
        "N": int(N),
        "F": 2,
        "tickers_in_order": tickers,
        "k": int(k),
        "W_corr": int(spec.W_corr),
        "W_mom": int(spec.W_mom),
        "start_date": spec.start_date,
        "graph": {
            "rule": "kNN over |corr| then symmetrize by union",
            "self_loops": False,
        },
        "edge_attr": {
            "raw": "signed Pearson corr in [-1,1]",
            "normalized": bool(normalize_edge_attr),
            "normalization": "z-score using off-diagonal full matrix per snapshot",
            "clip_value": clip_value,
        },
        "outputs": {
            "snapshots_dir": str(snap_dir).replace("\\", "/"),
            "index_csv": str(index_path).replace("\\", "/"),
            "diag_csv": str(diag_path).replace("\\", "/"),
        },
        "n_snapshots": int(len(index_df)),
        "date_min": str(index_df["date"].min()) if len(index_df) else None,
        "date_max": str(index_df["date"].max()) if len(index_df) else None,
    }

    meta_path = root_out / "snapshots_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[INFO] snapshots saved:", len(index_df))
    print("[INFO] range:", meta["date_min"], "->", meta["date_max"])
    print("[INFO] index:", index_path)
    print("[INFO] diag :", diag_path)
    print("[INFO] meta :", meta_path)


if __name__ == "__main__":
    main()

