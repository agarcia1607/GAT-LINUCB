# src/13_lints_contextual.py
# Linear Thompson Sampling — k=1 (single asset per week)
# Bayesian counterpart to LinUCB (src/10_linucb_contextual.py)
# Same update rule, different selection: sample theta from posterior
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def soft_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_returns(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index.date)
    return df


def list_dates_from_dir(dir_: Path) -> list[pd.Timestamp]:
    files = sorted(dir_.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {dir_}")
    dates = [pd.to_datetime(f.stem) for f in files]
    return sorted(pd.to_datetime(pd.Series(dates).dt.date))


# -------------------------------------------------------------------
# Quality Filter
# -------------------------------------------------------------------

def apply_quality_filter(returns: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    valid_tickers = []
    filtered_tickers = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if len(r) < 52:
            valid_tickers.append(ticker)
            continue
        max_wk = float(r.max())
        extreme_wks = int((r.abs() > 0.30).sum())
        cum = (1 + r).cumprod()
        dd = float(((cum - cum.cummax()) / cum.cummax()).min())
        if (max_wk > 0.50) or (extreme_wks > 5 and dd < -0.75):
            filtered_tickers.append(ticker)
        else:
            valid_tickers.append(ticker)
    print(f"[QualityFilter] {len(valid_tickers)} valid, {len(filtered_tickers)} filtered")
    print(f"[QualityFilter] Filtered: {filtered_tickers}")
    valid_idx = [list(returns.columns).index(t) for t in valid_tickers]
    return returns[valid_tickers], valid_idx


# -------------------------------------------------------------------
# Reward functions (same as LinUCB)
# -------------------------------------------------------------------

def sharpe_reward(history: dict, asset_idx: int, r_raw: float,
                  window: int = 12, annualize: bool = True) -> float:
    if asset_idx not in history:
        history[asset_idx] = []
    history[asset_idx].append(r_raw)
    buf = history[asset_idx][-window:]
    if len(buf) < 4:
        return r_raw
    arr = np.array(buf)
    std = arr.std()
    if std < 1e-8:
        return 0.0
    sr = arr.mean() / std
    if annualize:
        sr *= np.sqrt(52)
    return float(np.clip(sr, -3.0, 3.0))


def sortino_reward(history: dict, asset_idx: int, r_raw: float,
                   window: int = 12, annualize: bool = True) -> float:
    if asset_idx not in history:
        history[asset_idx] = []
    history[asset_idx].append(r_raw)
    buf = history[asset_idx][-window:]
    if len(buf) < 4:
        return r_raw
    arr = np.array(buf)
    downside = arr[arr < 0]
    if len(downside) == 0:
        val = arr.mean() * np.sqrt(52) if annualize else arr.mean()
        return float(np.clip(val, -3.0, 3.0))
    down_std = downside.std()
    if down_std < 1e-8:
        return 0.0
    sr = arr.mean() / down_std
    if annualize:
        sr *= np.sqrt(52)
    return float(np.clip(sr, -3.0, 3.0))


# -------------------------------------------------------------------
# Linear Thompson Sampling state — same as LinUCB
# -------------------------------------------------------------------

@dataclass
class LinTSState:
    A_inv: np.ndarray   # Posterior covariance (up to v²)
    b: np.ndarray       # Sufficient statistic


def lints_init(d: int, lam: float) -> LinTSState:
    return LinTSState(
        A_inv=(1.0 / lam) * np.eye(d, dtype=np.float64),
        b=np.zeros(d, dtype=np.float64),
    )


def theta_hat(st: LinTSState) -> np.ndarray:
    return st.A_inv @ st.b


def thompson_sample(st: LinTSState, v: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample theta_tilde ~ N(theta_hat, v^2 * A_inv)
    Uses Cholesky decomposition for efficient sampling.
    """
    mu = theta_hat(st)
    # Cholesky of v^2 * A_inv
    try:
        L = np.linalg.cholesky(v ** 2 * st.A_inv)
        return mu + L @ rng.standard_normal(len(mu))
    except np.linalg.LinAlgError:
        # Fallback: add small jitter for numerical stability
        jitter = 1e-6 * np.eye(st.A_inv.shape[0])
        L = np.linalg.cholesky(v ** 2 * st.A_inv + jitter)
        return mu + L @ rng.standard_normal(len(mu))


def lints_update(st: LinTSState, x: np.ndarray, r: float) -> None:
    """Sherman-Morrison update — identical to LinUCB."""
    x = x.astype(np.float64, copy=False)
    Ainv_x = st.A_inv @ x
    denom = 1.0 + float(x.T @ Ainv_x)
    st.A_inv = st.A_inv - np.outer(Ainv_x, Ainv_x) / denom
    st.b = st.b + float(r) * x


# -------------------------------------------------------------------
# run_policy
# -------------------------------------------------------------------

def run_lints(
    context_dir: Path,
    returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    lam: float,
    v: float,
    seed: int,
    out_dir: Path,
    reward_mode: str = "sharpe",
    valid_idx: list[int] | None = None,
) -> dict:
    """
    Linear Thompson Sampling — selects 1 asset per week.

    Parameters
    ----------
    v : float
        Posterior sampling scale. Higher v → more exploration.
        Equivalent to alpha in LinUCB.
    """
    rng = np.random.default_rng(seed)
    T = len(dates)

    Z0_full = np.load(context_dir / f"{dates[0].date().isoformat()}.npy")
    Z0 = Z0_full[valid_idx] if valid_idx is not None else Z0_full
    K, d = int(Z0.shape[0]), int(Z0.shape[1])

    if K != returns.shape[1]:
        raise ValueError(f"K mismatch: embeddings {K}, returns {returns.shape[1]}")

    st = lints_init(d=d, lam=lam)

    rows = []
    cum_reward, cum_regret = 0.0, 0.0
    last_a = None
    repeats = 0
    reward_history: dict = {}

    print(f"[LinTS] K={K}, d={d}, T={T-1} weeks, v={v}, reward={reward_mode}")

    for t in range(T - 1):
        date_t = dates[t]
        date_next = dates[t + 1]

        X_full = np.load(context_dir / f"{date_t.date().isoformat()}.npy").astype(np.float64)
        X = X_full[valid_idx] if valid_idx is not None else X_full

        # Thompson sample: draw theta_tilde from posterior
        theta_tilde = thompson_sample(st, v, rng)

        # Select action with highest expected reward under theta_tilde
        scores = X @ theta_tilde
        a = int(np.argmax(scores))
        mu_a = float(scores[a])

        r_raw = float(returns.loc[date_next, returns.columns[a]])

        if reward_mode == "sharpe":
            r = sharpe_reward(reward_history, a, r_raw)
        elif reward_mode == "sortino":
            r = sortino_reward(reward_history, a, r_raw)
        else:
            r = r_raw

        best_next = float(np.max(returns.loc[date_next].values))
        regret_emp = best_next - float(returns.loc[date_next, returns.columns[a]])

        cum_reward += r
        cum_regret += regret_emp

        lints_update(st, X[a], r)

        if last_a is not None and a == last_a:
            repeats += 1
        last_a = a

        rows.append({
            "t": t,
            "date_t": str(date_t.date()),
            "date_reward": str(date_next.date()),
            "a_idx": a,
            "asset": returns.columns[a],
            "reward_raw": r_raw,
            "reward_used": r,
            "best_next_return": best_next,
            "regret_emp": regret_emp,
            "cum_reward": cum_reward,
            "cum_regret_emp": cum_regret,
            "mu_a": mu_a,
            "theta_norm": float(np.linalg.norm(theta_hat(st))),
        })

    log_df = pd.DataFrame(rows)
    log_path = out_dir / "logs_lints.csv"
    log_df.to_csv(log_path, index=False)

    WEEKS = 52
    r_series = log_df["reward_raw"]
    ann_return = float((1 + r_series).prod() ** (WEEKS / len(r_series)) - 1)
    sharpe_val = float(np.sqrt(WEEKS) * r_series.mean() / r_series.std()) if r_series.std() > 0 else 0.0
    down = r_series[r_series < 0].std()
    sortino_val = float(np.sqrt(WEEKS) * r_series.mean() / down) if down > 0 else 0.0
    cum_ret = (1 + r_series).cumprod()
    max_dd = float(((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min())

    summary = {
        "policy": "LinTS",
        "context_dir": str(context_dir),
        "T_effective": int(T - 1),
        "K": int(K),
        "d": int(d),
        "lambda": float(lam),
        "v": float(v),
        "reward_mode": reward_mode,
        "quality_filter": valid_idx is not None,
        "seed": int(seed),
        "ann_return": ann_return,
        "sharpe": sharpe_val,
        "sortino": sortino_val,
        "max_drawdown": max_dd,
        "cum_reward": float(cum_reward),
        "cum_regret_emp": float(cum_regret),
        "repeat_rate": float(repeats / max(1, T - 2)),
        "log_path": str(log_path),
    }

    print(f"[LinTS] Ann.Return: {ann_return:.1%} | Sharpe: {sharpe_val:.3f} | MaxDD: {max_dd:.1%}")
    return summary


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Linear Thompson Sampling — k=1")
    ap.add_argument("--context", choices=["embeddings", "X_raw"], default="embeddings")
    ap.add_argument("--returns_path", type=str, default="data/processed/weekly_returns.parquet")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="Regularization parameter (same as LinUCB)")
    ap.add_argument("--v", type=float, default=1.0,
                    help="Posterior sampling scale. Higher = more exploration.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward_mode", choices=["raw", "sharpe", "sortino"], default="sharpe")
    ap.add_argument("--no_filter", action="store_true", default=False)
    args = ap.parse_args()

    context_dir = (Path("artifacts/embeddings_gat/npy")
                   if args.context == "embeddings"
                   else Path("artifacts/X_raw/npy"))

    returns_raw = load_returns(Path(args.returns_path))
    dates = list_dates_from_dir(context_dir)

    if args.no_filter:
        returns = returns_raw
        valid_idx = None
        print(f"[INFO] Universe: {returns.shape[1]} assets (no filter)")
    else:
        returns, valid_idx = apply_quality_filter(returns_raw)
        print(f"[INFO] Universe: {returns_raw.shape[1]} → {returns.shape[1]} assets after quality filter")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts/linucb") / f"run_lints_{args.context}_{run_tag}"
    soft_mkdir(out_dir)

    summary = run_lints(
        context_dir=context_dir,
        returns=returns,
        dates=dates,
        lam=args.lam,
        v=args.v,
        seed=args.seed,
        out_dir=out_dir,
        reward_mode=args.reward_mode,
        valid_idx=valid_idx,
    )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": [summary]}, f, indent=2)

    print(f"\n[OK] saved to: {out_dir}")
    print(json.dumps({"runs": [summary]}, indent=2))


if __name__ == "__main__":
    main()