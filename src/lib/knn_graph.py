from __future__ import annotations
import numpy as np
import pandas as pd

def knn_from_corr(corr: pd.DataFrame, k: int = 8) -> list[tuple[int, int]]:
    """
    Construye aristas dirigidas i -> j usando kNN sobre |corr|.
    Retorna una lista de pares (i, j).
    """
    C = corr.to_numpy()
    N = C.shape[0]

    # usamos magnitud
    absC = np.abs(C)

    # prohibimos self-loops
    np.fill_diagonal(absC, -np.inf)

    edges = []
    for i in range(N):
        # índices de los k mayores |corr|
        nn = np.argpartition(-absC[i], kth=k-1)[:k]
        for j in nn:
            edges.append((i, j))

    return edges
