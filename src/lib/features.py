from __future__ import annotations
import numpy as np
import pandas as pd

def momentum_from_window(window_mom: pd.DataFrame) -> pd.Series:
    """
    Momentum 4 semanas:
      mom4 = Π(1+r) - 1
    window_mom: últimas 4 filas (t-4..t-1)
    """
    return (1.0 + window_mom).prod(axis=0) - 1.0

def volatility_from_window(window_24: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Volatilidad 24 semanas:
      vol24 = std(r_{t-24..t-1}) * sqrt(52) si annualize
    window_24: últimas 24 filas (t-24..t-1)
    """
    vol = window_24.std(axis=0, ddof=1)
    if annualize:
        vol = vol * np.sqrt(52.0)
    return vol

def build_X_t(window_corr, window_mom):
    mom = momentum_from_window(window_mom).rename("mom4")
    vol = volatility_from_window(window_corr, annualize=True).rename("vol24")
    X = pd.concat([mom, vol], axis=1)
    return X

