from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class FiltrationSpec:
    W_corr: int = 24          # ventana para correlación (grafo)
    W_mom: int = 4            # ventana para momentum (feature)
    start_date: str = "2023-01-01"  # snapshots con fecha >= start_date

@dataclass(frozen=True)
class FiltrationSlice:
    """
    Representa el 'estado de información' disponible justo antes de observar r_t.
    Es decir, F_{t-1}.
    """
    t: pd.Timestamp                 # fecha del snapshot
    t_pos: int                      # índice entero en el DataFrame
    hist_until: pd.Timestamp        # última fecha incluida en F_{t-1} (t-1)
    window_corr: pd.DataFrame       # retornos [t-W_corr, ..., t-1]
    window_mom: pd.DataFrame        # retornos [t-W_mom,  ..., t-1]

def iter_filtration(df_returns: pd.DataFrame, spec: FiltrationSpec):
    """
    df_returns:
      - index: fechas semanales ordenadas (Timestamp)
      - columns: tickers (N fijo)
      - values: retornos semanales r_t

    Genera slices (t, ventanas) tales que TODO usa solo pasado (<= t-1).
    """
    df = df_returns.sort_index()
    start = pd.Timestamp(spec.start_date)

    # Requisito para que ambas ventanas existan (corr domina porque W_corr >= W_mom)
    min_history = max(spec.W_corr, spec.W_mom)

    dates = df.index
    for t_pos, t in enumerate(dates):
        if t < start:
            continue
        if t_pos < min_history:
            continue

        # F_{t-1}: termina en t_pos-1 (la fila anterior)
        hist_until = dates[t_pos - 1]

        # Ventana de correlación: [t_pos - W_corr, ..., t_pos-1]
        window_corr = df.iloc[t_pos - spec.W_corr : t_pos]

        # Ventana momentum: [t_pos - W_mom, ..., t_pos-1]
        window_mom = df.iloc[t_pos - spec.W_mom : t_pos]

        # sanity: ambas terminan en t-1
        assert window_corr.index.max() == hist_until
        assert window_mom.index.max() == hist_until

        yield FiltrationSlice(
            t=t,
            t_pos=t_pos,
            hist_until=hist_until,
            window_corr=window_corr,
            window_mom=window_mom,
        )
