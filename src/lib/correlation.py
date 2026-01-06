from __future__ import annotations
import numpy as np
import pandas as pd

def compute_corr(window_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de correlación de Pearson sobre la ventana de 24 semanas.
    Input:
      window_corr: DataFrame (24 x N), columnas en orden fijo
    Output:
      corr: DataFrame (N x N), simétrica, diagonal = 1
    """
    corr = window_corr.corr(method="pearson")

    # saneo mínimo (defensivo)
    corr = corr.fillna(0.0)

    return corr
