# src/09_build_X_raw_snapshots.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.lib.io_universe import load_weekly_returns_aligned
from src.lib.filtration import FiltrationSpec, iter_filtration
from src.lib.features import build_X_t


def main():
    # Carga retornos semanales alineados al universo fijo (48 activos)
    df = load_weekly_returns_aligned()

    # Especificación causal (misma del pipeline de snapshots)
    spec = FiltrationSpec(W_corr=24, W_mom=4, start_date="2015-01-01")

    out_dir = Path("artifacts/X_raw/npy")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sl in iter_filtration(df, spec):
        # X_t con columnas ["mom4","vol24"] por activo (48x2)
        Xdf = build_X_t(sl.window_corr, sl.window_mom)

        # Validaciones obligatorias
        assert Xdf.shape == (df.shape[1], 2), f"Bad X shape at {sl.t}: {Xdf.shape}"
        assert int(Xdf.isna().sum().sum()) == 0, f"NaNs in X at {sl.t}"

        # Guardar como .npy (48,2) float32
        X = Xdf[["mom4", "vol24"]].to_numpy(dtype=np.float32, copy=True)
        out_path = out_dir / f"{sl.t.date().isoformat()}.npy"
        np.save(out_path, X)

        # Diagnóstico simple
        rows.append({
            "date": str(sl.t.date()),
            "mom4_mean": float(Xdf["mom4"].mean()),
            "vol24_mean": float(Xdf["vol24"].mean()),
            "mom4_min": float(Xdf["mom4"].min()),
            "mom4_max": float(Xdf["mom4"].max()),
            "vol24_min": float(Xdf["vol24"].min()),
            "vol24_max": float(Xdf["vol24"].max()),
        })

    diag = pd.DataFrame(rows)
    diag_path = Path("artifacts/X_raw") / "X_raw_diag.csv"
    diag.to_csv(diag_path, index=False)

    print("[INFO] saved X_raw to:", out_dir)
    print("[INFO] diag:", diag_path)
    print(diag.tail(5))


if __name__ == "__main__":
    main()
