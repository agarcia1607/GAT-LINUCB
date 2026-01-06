# src/12_block5_temporal_stability.py
# BLOQUE 5 — 5.3 Estabilidad temporal de embeddings relacionales (Z_t)
# - Drift por activo y global (L2, cos, Frobenius)
# - Alineamiento Procrustes para evitar "falso drift" por rotación
# - Baseline permutado (rompe coherencia temporal por activo)
# - Outputs: CSVs + PNGs en artifacts/block5/temporal_stability

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------

@dataclass
class Config:
    embeddings_dir: Path = Path("artifacts/embeddings_gat/npy")
    xraw_dir: Path = Path("artifacts/X_raw/npy")
    node_map_path: Path = Path("artifacts/snapshots/node_map.json")  # optional but recommended
    out_dir: Path = Path("artifacts/block5/temporal_stability")
    seed: int = 7


# -------------------------
# Utils
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_dates_from_npy(folder: Path) -> List[str]:
    files = sorted(folder.glob("*.npy"))
    dates = [f.stem for f in files]
    return dates

def load_matrix_npy(path: Path) -> np.ndarray:
    z = np.load(path)
    if z.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {z.shape} at {path}")
    return z.astype(np.float64, copy=False)

def cosine_distance_rows(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # 1 - cos(row_i(A), row_i(B))
    num = np.sum(A * B, axis=1)
    den = (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)) + eps
    return 1.0 - (num / den)

def procrustes_align(Z_prev: np.ndarray, Z_curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve R = argmin ||Z_prev - Z_curr R||_F  s.t. R^T R = I
    Returns: Z_curr_aligned, R
    """
    # Cross-covariance
    M = Z_curr.T @ Z_prev
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    Z_aligned = Z_curr @ R
    return Z_aligned, R

def load_node_names(node_map_path: Path) -> List[str] | None:
    if not node_map_path.exists():
        return None
    with open(node_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # common patterns:
    # 1) {"AAPL":0, ...} or {0:"AAPL", ...}
    # 2) {"id_to_ticker": {...}} etc.
    if all(isinstance(k, str) and isinstance(v, int) for k, v in m.items()):
        inv = {v: k for k, v in m.items()}
        return [inv[i] for i in range(len(inv))]
    if all(isinstance(k, str) and isinstance(v, str) for k, v in m.items()):
        # unlikely, but handle
        return list(m.values())

    # nested possibilities
    for key in ["id_to_ticker", "idx_to_ticker", "id2ticker", "index_to_ticker"]:
        if key in m and isinstance(m[key], dict):
            inv = {int(k): v for k, v in m[key].items()} if all(isinstance(k, str) for k in m[key].keys()) else m[key]
            return [inv[i] for i in range(len(inv))]

    return None


# -------------------------
# Core analysis
# -------------------------

def compute_drifts(series: Dict[str, np.ndarray], align: bool) -> Dict[str, pd.DataFrame]:
    """
    series: dict date->matrix (N x d)
    align: whether to Procrustes-align Z_t to Z_{t-1}
    Returns dict with:
      - drift_global: per-date global drift metrics
      - drift_asset:  long table (date, asset_idx, l2, cos)
    """
    dates = sorted(series.keys())
    N, d = series[dates[0]].shape

    rows_global = []
    rows_asset = []

    for i in range(1, len(dates)):
        t_prev = dates[i-1]
        t = dates[i]
        A = series[t_prev]
        B = series[t]

        if B.shape != A.shape:
            raise ValueError(f"Shape mismatch {t_prev}:{A.shape} vs {t}:{B.shape}")

        if align:
            B_use, _ = procrustes_align(A, B)
        else:
            B_use = B

        diff = B_use - A

        # global
        frob = np.linalg.norm(diff, ord="fro")
        prev_norm = np.linalg.norm(A, ord="fro") + 1e-12
        frob_norm = frob / prev_norm

        rows_global.append({
            "date": t,
            "prev_date": t_prev,
            "frob": float(frob),
            "frob_norm": float(frob_norm),
            "prev_frob": float(np.linalg.norm(A, ord="fro")),
            "curr_frob": float(np.linalg.norm(B_use, ord="fro")),
        })

        # per-asset
        l2 = np.linalg.norm(diff, axis=1)
        cos = cosine_distance_rows(A, B_use)

        for a in range(N):
            rows_asset.append({
                "date": t,
                "prev_date": t_prev,
                "asset_idx": a,
                "l2": float(l2[a]),
                "cos": float(cos[a]),
            })

    drift_global = pd.DataFrame(rows_global)
    drift_asset = pd.DataFrame(rows_asset)
    return {"drift_global": drift_global, "drift_asset": drift_asset}

def permuted_baseline(series: Dict[str, np.ndarray], seed: int, align: bool) -> pd.DataFrame:
    """
    Break temporal identity: permute rows within each Z_t (except the first),
    then compute global frob_norm drift vs previous real Z_{t-1}.
    """
    rng = np.random.default_rng(seed)
    dates = sorted(series.keys())
    rows = []

    for i in range(1, len(dates)):
        t_prev = dates[i-1]
        t = dates[i]
        A = series[t_prev]
        B = series[t].copy()

        perm = rng.permutation(B.shape[0])
        B = B[perm, :]

        if align:
            B, _ = procrustes_align(A, B)

        diff = B - A
        frob = np.linalg.norm(diff, ord="fro")
        frob_norm = frob / (np.linalg.norm(A, ord="fro") + 1e-12)

        rows.append({
            "date": t,
            "prev_date": t_prev,
            "frob_perm": float(frob),
            "frob_norm_perm": float(frob_norm),
        })

    return pd.DataFrame(rows)


# -------------------------
# Plotting
# -------------------------

def plot_global(d_real: pd.DataFrame, d_align: pd.DataFrame, d_perm: pd.DataFrame, out_png: Path) -> None:
    fig = plt.figure()
    plt.plot(pd.to_datetime(d_real["date"]), d_real["frob_norm"], label="real (no align)")
    plt.plot(pd.to_datetime(d_align["date"]), d_align["frob_norm"], label="real (Procrustes)")
    plt.plot(pd.to_datetime(d_perm["date"]), d_perm["frob_norm_perm"], label="perm baseline (Procrustes)")
    plt.xlabel("date")
    plt.ylabel("normalized Frobenius drift")
    plt.title("Temporal drift of embeddings (global)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_hist_global(d_real: pd.DataFrame, d_perm: pd.DataFrame, out_png: Path) -> None:
    fig = plt.figure()
    plt.hist(d_real["frob_norm"].values, bins=40, alpha=0.7, label="real")
    plt.hist(d_perm["frob_norm_perm"].values, bins=40, alpha=0.7, label="permuted baseline")
    plt.xlabel("normalized Frobenius drift")
    plt.ylabel("count")
    plt.title("Distribution: real drift vs permuted baseline")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_top_assets_box(d_asset_align: pd.DataFrame, asset_names: List[str] | None, out_png: Path, top_k: int = 10) -> None:
    # select top_k by mean l2 drift
    g = d_asset_align.groupby("asset_idx")["l2"].mean().sort_values(ascending=False)
    top = g.head(top_k).index.tolist()

    data = [d_asset_align.loc[d_asset_align["asset_idx"] == a, "l2"].values for a in top]
    labels = []
    for a in top:
        if asset_names is None:
            labels.append(str(a))
        else:
            labels.append(asset_names[a])

    fig = plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("L2 drift (aligned)")
    plt.title(f"Top-{top_k} assets by mean aligned L2 drift")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def main() -> None:
    cfg = Config()
    ensure_dir(cfg.out_dir)
    np.random.seed(cfg.seed)

    # Load dates intersection between embeddings and xraw (for later comparisons)
    dates_Z = list_dates_from_npy(cfg.embeddings_dir)
    if len(dates_Z) < 2:
        raise RuntimeError(f"Not enough embedding files in {cfg.embeddings_dir}")

    # Load embeddings series
    Z_series: Dict[str, np.ndarray] = {}
    for dt in dates_Z:
        Z_series[dt] = load_matrix_npy(cfg.embeddings_dir / f"{dt}.npy")

    # Optional asset names
    asset_names = load_node_names(cfg.node_map_path)

    # Drift (no alignment) + Drift (Procrustes)
    dr_no = compute_drifts(Z_series, align=False)
    dr_al = compute_drifts(Z_series, align=True)

    # Permuted baseline (aligned)
    d_perm = permuted_baseline(Z_series, seed=cfg.seed, align=True)

    # Save tables
    dr_no["drift_global"].to_csv(cfg.out_dir / "drift_global_no_align.csv", index=False)
    dr_no["drift_asset"].to_csv(cfg.out_dir / "drift_asset_no_align.csv", index=False)
    dr_al["drift_global"].to_csv(cfg.out_dir / "drift_global_procrustes.csv", index=False)
    dr_al["drift_asset"].to_csv(cfg.out_dir / "drift_asset_procrustes.csv", index=False)
    d_perm.to_csv(cfg.out_dir / "drift_global_perm_baseline.csv", index=False)

    # Summary tables
    top_assets = (
        dr_al["drift_asset"]
        .groupby("asset_idx")[["l2", "cos"]]
        .mean()
        .sort_values("l2", ascending=False)
        .reset_index()
    )
    if asset_names is not None:
        top_assets["asset"] = top_assets["asset_idx"].map(lambda i: asset_names[int(i)])
        cols = ["asset_idx", "asset", "l2", "cos"]
        top_assets = top_assets[cols]
    top_assets.to_csv(cfg.out_dir / "top_assets_by_drift.csv", index=False)

    # Identify shock weeks (top 10 by aligned global drift)
    shocks = dr_al["drift_global"].sort_values("frob_norm", ascending=False).head(10)
    shocks.to_csv(cfg.out_dir / "top10_shock_weeks.csv", index=False)

    # Plots
    plot_global(
        d_real=dr_no["drift_global"],
        d_align=dr_al["drift_global"],
        d_perm=d_perm,
        out_png=cfg.out_dir / "global_drift_timeseries.png"
    )
    plot_hist_global(
        d_real=dr_al["drift_global"],
        d_perm=d_perm,
        out_png=cfg.out_dir / "global_drift_hist_real_vs_perm.png"
    )
    plot_top_assets_box(
        d_asset_align=dr_al["drift_asset"],
        asset_names=asset_names,
        out_png=cfg.out_dir / "top_assets_boxplot_l2_aligned.png",
        top_k=10
    )

    # Minimal console report
    print("[OK] Saved outputs to:", cfg.out_dir)
    print("[INFO] Dates:", dates_Z[0], "->", dates_Z[-1], " | n=", len(dates_Z))
    print("[INFO] Embedding shape:", Z_series[dates_Z[0]].shape)

    med_real = float(np.median(dr_al["drift_global"]["frob_norm"].values))
    med_perm = float(np.median(d_perm["frob_norm_perm"].values))
    print(f"[INFO] Median aligned drift: real={med_real:.6f}  perm={med_perm:.6f}")

if __name__ == "__main__":
    main()
