# src/12_linucb_combinatorial.py
# Combinatorial LinUCB — selects k assets per week simultaneously
# Portfolio reward: rolling Sharpe of the k-asset portfolio
# Does NOT modify src/10_linucb_contextual.py or src/11_linucb_filtered.py
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
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
# Quality Filter (same as src/11_linucb_filtered.py)
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
    valid_idx = [list(returns.columns).index(t) for t in valid_tickers]
    return returns[valid_tickers], valid_idx


# -------------------------------------------------------------------
# Reward functions
# -------------------------------------------------------------------

def portfolio_sharpe_reward(
    r_portfolio: np.ndarray,
    history: list,
    window: int = 12,
    annualize: bool = True,
) -> float:
    """
    Reward = rolling Sharpe of the equally-weighted portfolio of k assets.
    r_portfolio: array of k realized returns this week
    history: list of past portfolio weekly returns
    """
    portfolio_return = float(np.mean(r_portfolio))
    history.append(portfolio_return)
    buf = history[-window:]

    if len(buf) < 4:
        return portfolio_return  # fallback to raw return

    arr = np.array(buf)
    std = arr.std()
    if std < 1e-8:
        return 0.0
    sr = arr.mean() / std
    if annualize:
        sr *= np.sqrt(52)
    return float(np.clip(sr, -3.0, 3.0))


# -------------------------------------------------------------------
# LinUCB state
# -------------------------------------------------------------------

@dataclass
class LinUCBState:
    A_inv: np.ndarray
    b: np.ndarray


def linucb_init(d: int, lam: float) -> LinUCBState:
    return LinUCBState(
        A_inv=(1.0 / lam) * np.eye(d, dtype=np.float64),
        b=np.zeros(d, dtype=np.float64),
    )


def theta_hat(st: LinUCBState) -> np.ndarray:
    return st.A_inv @ st.b


def ucb_scores(st: LinUCBState, X: np.ndarray, alpha: float):
    th = theta_hat(st)
    mu = X @ th
    XA = X @ st.A_inv
    sigma = np.sqrt(np.sum(XA * X, axis=1))
    return mu + alpha * sigma, mu, sigma


def linucb_update(st: LinUCBState, x: np.ndarray, r: float) -> None:
    x = x.astype(np.float64, copy=False)
    Ainv_x = st.A_inv @ x
    denom = 1.0 + float(x.T @ Ainv_x)
    st.A_inv = st.A_inv - np.outer(Ainv_x, Ainv_x) / denom
    st.b = st.b + float(r) * x


# -------------------------------------------------------------------
# Combinatorial run_policy
# -------------------------------------------------------------------

def run_combinatorial(
    context_dir: Path,
    returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    alpha: float,
    lam: float,
    seed: int,
    out_dir: Path,
    k: int = 5,
    valid_idx: list[int] | None = None,
) -> dict:
    """
    CombLinUCB: selects top-k assets by UCB score each week.
    Reward: rolling Sharpe of the equally-weighted k-asset portfolio.
    """
    rng = np.random.default_rng(seed)
    T = len(dates)
    if T < 2:
        raise ValueError("Need at least 2 snapshots.")

    missing = [d for d in dates if d not in returns.index]
    if missing:
        raise ValueError(f"Returns missing {len(missing)} dates.")

    Z0_full = np.load(context_dir / f"{dates[0].date().isoformat()}.npy")
    Z0 = Z0_full[valid_idx] if valid_idx is not None else Z0_full
    K, d = int(Z0.shape[0]), int(Z0.shape[1])

    if K != returns.shape[1]:
        raise ValueError(f"K mismatch: embeddings {K}, returns {returns.shape[1]}")

    if k > K:
        raise ValueError(f"k={k} > K={K}")

    st = linucb_init(d=d, lam=lam)

    rows = []
    cum_reward = 0.0
    cum_regret = 0.0
    portfolio_history: list = []  # rolling buffer of portfolio returns
    theta_norm = None

    print(f"[CombLinUCB] k={k}, K={K}, d={d}, T={T-1} weeks")

    for t in range(T - 1):
        date_t = dates[t]
        date_next = dates[t + 1]

        X_full = np.load(context_dir / f"{date_t.date().isoformat()}.npy").astype(np.float64)
        X = X_full[valid_idx] if valid_idx is not None else X_full
        assert X.shape == (K, d)

        # CombLinUCB: select top-k by UCB score
        ucb, mu, sig = ucb_scores(st, X, alpha=alpha)
        top_k_idx = np.argsort(ucb)[-k:]  # k assets with highest UCB

        # Realized returns of selected portfolio
        r_portfolio = np.array([
            float(returns.loc[date_next, returns.columns[a]])
            for a in top_k_idx
        ])

        # Portfolio metrics this week
        portfolio_return_raw = float(np.mean(r_portfolio))

        # Reward: rolling Sharpe of portfolio
        r_reward = portfolio_sharpe_reward(r_portfolio, portfolio_history, window=12)

        # Regret: best possible single asset vs portfolio average
        best_next = float(np.max(returns.loc[date_next].values))
        regret_emp = best_next - portfolio_return_raw

        cum_reward += r_reward
        cum_regret += regret_emp

        # Update LinUCB for each selected asset with portfolio reward
        for a in top_k_idx:
            linucb_update(st, X[a], r_reward)

        theta_norm = float(np.linalg.norm(theta_hat(st)))

        # Asset diversity: correlation between selected assets
        selected_tickers = [returns.columns[a] for a in top_k_idx]

        rows.append({
            "t": t,
            "date_t": str(date_t.date()),
            "date_reward": str(date_next.date()),
            "assets": ",".join(selected_tickers),
            "portfolio_return": portfolio_return_raw,
            "reward_used": r_reward,
            "best_next_return": best_next,
            "regret_emp": regret_emp,
            "cum_reward": cum_reward,
            "cum_regret_emp": cum_regret,
            "theta_norm": theta_norm,
            "k": k,
        })

    log_df = pd.DataFrame(rows)
    log_path = out_dir / f"logs_combinatorial_k{k}.csv"
    log_df.to_csv(log_path, index=False)

    WEEKS = 52
    r_series = log_df["portfolio_return"]
    ann_return = float((1 + r_series).prod() ** (WEEKS / len(r_series)) - 1)
    sharpe = float(np.sqrt(WEEKS) * r_series.mean() / r_series.std()) if r_series.std() > 0 else 0.0
    down = r_series[r_series < 0].std()
    sortino = float(np.sqrt(WEEKS) * r_series.mean() / down) if down > 0 else 0.0
    cum_ret = (1 + r_series).cumprod()
    max_dd = float(((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min())

    summary = {
        "policy": f"CombLinUCB_k{k}",
        "context_dir": str(context_dir),
        "T_effective": int(T - 1),
        "K": int(K),
        "k": int(k),
        "d": int(d),
        "alpha": float(alpha),
        "lambda": float(lam),
        "reward_mode": "portfolio_sharpe",
        "quality_filter": valid_idx is not None,
        "seed": int(seed),
        "ann_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "cum_reward": float(cum_reward),
        "cum_regret_emp": float(cum_regret),
        "mean_portfolio_return": float(r_series.mean()),
        "log_path": str(log_path),
    }

    print(f"[CombLinUCB k={k}] Ann.Return: {ann_return:.1%} | Sharpe: {sharpe:.3f} | MaxDD: {max_dd:.1%}")
    return summary


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Combinatorial LinUCB — k assets per week")
    ap.add_argument("--context", choices=["embeddings", "X_raw"], default="embeddings")
    ap.add_argument("--returns_path", type=str, default="data/processed/weekly_returns.parquet")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5,
                    help="Number of assets to select per week (default: 5)")
    ap.add_argument("--k_values", type=str, default=None,
                    help="Comma-separated list of k values to compare (e.g. '3,5,10'). Overrides --k")
    ap.add_argument("--no_filter", action="store_true", default=False,
                    help="Disable quality filter (use full universe)")
    args = ap.parse_args()

    context_dir = (Path("artifacts/embeddings_gat/npy")
                   if args.context == "embeddings"
                   else Path("artifacts/X_raw/npy"))

    returns_raw = load_returns(Path(args.returns_path))
    dates = list_dates_from_dir(context_dir)

    # Quality filter
    if args.no_filter:
        returns = returns_raw
        valid_idx = None
        print(f"[INFO] Universe: {returns.shape[1]} assets (no filter)")
    else:
        returns, valid_idx = apply_quality_filter(returns_raw)
        print(f"[INFO] Universe: {returns_raw.shape[1]} → {returns.shape[1]} assets after quality filter")

    # k values to run
    if args.k_values:
        k_list = [int(x.strip()) for x in args.k_values.split(",")]
    else:
        k_list = [args.k]

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts/linucb") / f"run_combinatorial_{run_tag}"
    soft_mkdir(out_dir)

    summaries = []
    for k in k_list:
        print(f"\n{'='*50}")
        print(f"Running CombLinUCB with k={k}")
        print(f"{'='*50}")
        summary = run_combinatorial(
            context_dir=context_dir,
            returns=returns,
            dates=dates,
            alpha=args.alpha,
            lam=args.lam,
            seed=args.seed,
            out_dir=out_dir,
            k=k,
            valid_idx=valid_idx,
        )
        summaries.append(summary)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "runs": summaries,
            "quality_filter": not args.no_filter,
            "context": args.context,
            "alpha": args.alpha,
        }, f, indent=2)

    print(f"\n[OK] saved to: {out_dir}")

    # Summary table
    print("\n=== Results Summary ===")
    print(f"{'Policy':>20} | {'Ann.Return':>10} | {'Sharpe':>6} | {'Sortino':>7} | {'MaxDD':>7}")
    print("-" * 60)
    for s in summaries:
        print(f"{s['policy']:>20} | {s['ann_return']:>10.1%} | {s['sharpe']:>6.3f} | {s['sortino']:>7.3f} | {s['max_drawdown']:>7.1%}")


if __name__ == "__main__":
    main()