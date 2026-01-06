from __future__ import annotations
import numpy as np
import pandas as pd

def _offdiag_vals(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    return C[~np.eye(n, dtype=bool)]

def corr_snapshot_stats(corr: pd.DataFrame, eps: float = 1e-8) -> tuple[float, float]:
    """
    μ_t y σ_t sobre off-diagonal de C_t
    """
    C = corr.to_numpy(dtype=np.float64)
    od = _offdiag_vals(C)
    mu = float(np.mean(od))
    sigma = float(np.std(od, ddof=0))
    if sigma < eps:
        sigma = eps
    return mu, sigma

def build_edge_tensors(
    edges_sym: list[tuple[int, int]],
    corr: pd.DataFrame,
    normalize: bool = True,
    clip_value: float | None = 3.0,
):
    """
    Returns:
      edge_index: (2, E)
      edge_attr:  (E, 1)
      stats: dict(mu_t, sigma_t, ...)
    """
    C = corr.to_numpy(dtype=np.float64)

    mu_t, sigma_t = (0.0, 1.0)
    if normalize:
        mu_t, sigma_t = corr_snapshot_stats(corr)

    E = len(edges_sym)
    edge_index = np.empty((2, E), dtype=np.int64)
    edge_attr = np.empty((E, 1), dtype=np.float32)

    for k, (i, j) in enumerate(edges_sym):
        edge_index[0, k] = i
        edge_index[1, k] = j
        val = C[i, j]
        if normalize:
            val = (val - mu_t) / sigma_t
        if clip_value is not None:
            val = float(np.clip(val, -clip_value, clip_value))
        edge_attr[k, 0] = np.float32(val)

    stats = {
        "mu_t": float(mu_t),
        "sigma_t": float(sigma_t),
        "normalize": normalize,
        "clip_value": clip_value,
    }
    return edge_index, edge_attr, stats
