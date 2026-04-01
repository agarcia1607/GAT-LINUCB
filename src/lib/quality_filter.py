"""
src/lib/quality_filter.py
Filtro de calidad de activos — excluye empresas con volatilidad extrema
por eventos corporativos extraordinarios (bancarrotas, problemas contables, etc.)

Criterio: excluir si max_weekly_return > 50% O
          (semanas con |retorno| > 30% > 5 Y max_drawdown < -75%)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def compute_asset_quality(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de calidad por activo sobre el histórico completo.
    
    returns: DataFrame (T x K) de retornos semanales
    
    Retorna DataFrame con columnas:
        ticker, max_weekly_return, extreme_weeks, max_drawdown, filtered
    """
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if len(r) < 52:
            rows.append({
                "ticker": ticker,
                "max_weekly_return": 0.0,
                "extreme_weeks": 0,
                "max_drawdown": 0.0,
                "filtered": False,
            })
            continue

        max_wk = float(r.max())
        extreme_wks = int((r.abs() > 0.30).sum())
        
        # Max drawdown sobre precio acumulado
        cum = (1 + r).cumprod()
        dd = float(((cum - cum.cummax()) / cum.cummax()).min())

        # Criterio combinado
        filtered = (max_wk > 0.50) or (extreme_wks > 5 and dd < -0.75)

        rows.append({
            "ticker": ticker,
            "max_weekly_return": max_wk,
            "extreme_weeks": extreme_wks,
            "max_drawdown": dd,
            "filtered": filtered,
        })

    return pd.DataFrame(rows).set_index("ticker")


def get_valid_tickers(returns: pd.DataFrame) -> list[str]:
    """Retorna lista de tickers que pasan el filtro de calidad."""
    quality = compute_asset_quality(returns)
    valid = quality[~quality["filtered"]].index.tolist()
    filtered = quality[quality["filtered"]].index.tolist()
    print(f"[QualityFilter] {len(valid)} activos válidos, {len(filtered)} filtrados")
    print(f"[QualityFilter] Filtrados: {filtered}")
    return valid


def apply_quality_filter(returns: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame solo con activos que pasan el filtro."""
    valid = get_valid_tickers(returns)
    return returns[valid]
