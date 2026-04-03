# src/14_lints_combinatorial.py
# Combinatorial Linear Thompson Sampling — k assets per week
# Bayesian counterpart to CombLinUCB (src/12_linucb_combinatorial.py)
from __future__ import annotations

import argparse
import json
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
# Portfolio reward
# -------------------------------------------------------------------

def portfolio_sharpe_reward(
    r_portfolio: np.ndarray,
    history: list,
    window: int = 12,
    annualize: bool = True,
) -> float:
    portfolio_return = float(np.mean(r_portfolio))
    history.append(portfolio_return)
    buf = history[-window:]
    if len(buf) < 4:
        return portfolio_return
    arr = np.array(buf)
    std = arr.std()
    if std < 1e-8:
        return 0.0
    sr = arr.mean() / std
    if annualize:
        sr *= np.sqrt(52)
    return float(np.clip(sr, -3.0, 3.0))


# -------------------------------------------------------------------
# LinTS state
# -------------------------------------------------------------------

@dataclass
class LinTSState:
    A_inv: np.ndarray
    b: np.ndarray


def lints_init(d: int, lam: float) -> LinTSState:
    return LinTSState(
        A_inv=(1.0 / lam) * np.eye(d, dtype=np.float64),
        b=np.zeros(d, dtype=np.float64),
    )


def theta_hat(st: LinTSState) -> np.ndarray:
    return st.A_inv @ st.b


def thompson_sample(st: LinTSState, v: float, rng: np.random.Generator) -> np.ndarray:
    mu = theta_hat(st)
    try:
        L = np.linalg.cholesky(v ** 2 * st.A_inv)
        return mu + L @ rng.standard_normal(len(mu))
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.eye(st.A_inv.shape[0])
        L = np.linalg.cholesky(v ** 2 * st.A_inv + jitter)
        return mu + L @ rng.standard_normal(len(mu))


def lints_update(st: LinTSState, x: np.ndarray, r: float) -> None:
    x = x.astype(np.float64, copy=False)
    Ainv_x = st.A_inv @ x
    denom = 1.0 + float(x.T @ Ainv_x)
    st.A_inv = st.A_inv - np.outer(Ainv_x, Ainv_x) / denom
    st.b = st.b + float(r) * x


# -------------------------------------------------------------------
# Combinatorial run
# -------------------------------------------------------------------

def run_comb_lints(
    context_dir: Path,
    returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    lam: float,
    v: float,
    seed: int,
    out_dir: Path,
    k: int = 5,
    valid_idx: list[int] | None = None,
) -> dict:
    """
    Combinatorial Linear Thompson Sampling.
    Selects top-k assets by score under sampled theta_tilde.
    Reward: rolling Sharpe of equally-weighted k-asset portfolio.
    """
    rng = np.random.default_rng(seed)
    T = len(dates)

    Z0_full = np.load(context_dir / f"{dates[0].date().isoformat()}.npy")
    Z0 = Z0_full[valid_idx] if valid_idx is not None else Z0_full
    K, d = int(Z0.shape[0]), int(Z0.shape[1])

    if K != returns.shape[1]:
        raise ValueError(f"K mismatch: embeddings {K}, returns {returns.shape[1]}")
    if k > K:
        raise ValueError(f"k={k} > K={K}")

    st = lints_init(d=d, lam=lam)

    rows = []
    cum_reward = 0.0
    cum_regret = 0.0
    portfolio_history: list = []

    print(f"[CombLinTS] k={k}, K={K}, d={d}, T={T-1} weeks, v={v}")

    for t in range(T - 1):
        date_t = dates[t]
        date_next = dates[t + 1]

        X_full = np.load(context_dir / f"{date_t.date().isoformat()}.npy").astype(np.float64)
        X = X_full[valid_idx] if valid_idx is not None else X_full

        # Thompson sample
        theta_tilde = thompson_sample(st, v, rng)

        # Select top-k by sampled scores
        scores = X @ theta_tilde
        top_k_idx = np.argsort(scores)[-k:]

        # Realized portfolio return
        r_portfolio = np.array([
            float(returns.loc[date_next, returns.columns[a]])
            for a in top_k_idx
        ])
        portfolio_return_raw = float(np.mean(r_portfolio))

        # Portfolio Sharpe reward
        r_reward = portfolio_sharpe_reward(r_portfolio, portfolio_history, window=12)

        best_next = float(np.max(returns.loc[date_next].values))
        regret_emp = best_next - portfolio_return_raw

        cum_reward += r_reward
        cum_regret += regret_emp

        # Update for each selected asset
        for a in top_k_idx:
            lints_update(st, X[a], r_reward)

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
            "theta_norm": float(np.linalg.norm(theta_hat(st))),
            "k": k,
        })

    log_df = pd.DataFrame(rows)
    log_path = out_dir / f"logs_comb_lints_k{k}.csv"
    log_df.to_csv(log_path, index=False)

    WEEKS = 52
    r_series = log_df["portfolio_return"]
    ann_return = float((1 + r_series).prod() ** (WEEKS / len(r_series)) - 1)
    sharpe_val = float(np.sqrt(WEEKS) * r_series.mean() / r_series.std()) if r_series.std() > 0 else 0.0
    down = r_series[r_series < 0].std()
    sortino_val = float(np.sqrt(WEEKS) * r_series.mean() / down) if down > 0 else 0.0
    cum_ret = (1 + r_series).cumprod()
    max_dd = float(((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min())

    summary = {
        "policy": f"CombLinTS_k{k}",
        "context_dir": str(context_dir),
        "T_effective": int(T - 1),
        "K": int(K),
        "k": int(k),
        "d": int(d),
        "lambda": float(lam),
        "v": float(v),
        "reward_mode": "portfolio_sharpe",
        "quality_filter": valid_idx is not None,
        "seed": int(seed),
        "ann_return": ann_return,
        "sharpe": sharpe_val,
        "sortino": sortino_val,
        "max_drawdown": max_dd,
        "cum_reward": float(cum_reward),
        "cum_regret_emp": float(cum_regret),
        "mean_portfolio_return": float(r_series.mean()),
        "log_path": str(log_path),
    }

    print(f"[CombLinTS k={k}] Ann.Return: {ann_return:.1%} | Sharpe: {sharpe_val:.3f} | MaxDD: {max_dd:.1%}")
    return summary


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Combinatorial Linear Thompson Sampling")
    ap.add_argument("--context", choices=["embeddings", "X_raw"], default="embeddings")
    ap.add_argument("--returns_path", type=str, default="data/processed/weekly_returns.parquet")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--v", type=float, default=1.0,
                    help="Posterior sampling scale. Higher = more exploration.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--k_values", type=str, default=None,
                    help="Comma-separated k values e.g. '3,5,10'")
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

    k_list = [int(x.strip()) for x in args.k_values.split(",")] if args.k_values else [args.k]

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts/linucb") / f"run_comb_lints_{run_tag}"
    soft_mkdir(out_dir)

    summaries = []
    for k in k_list:
        print(f"\n{'='*50}")
        print(f"Running CombLinTS with k={k}")
        print(f"{'='*50}")
        summary = run_comb_lints(
            context_dir=context_dir,
            returns=returns,
            dates=dates,
            lam=args.lam,
            v=args.v,
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
            "v": args.v,
        }, f, indent=2)

    print(f"\n[OK] saved to: {out_dir}")

    print("\n=== Results Summary ===")
    print(f"{'Policy':>20} | {'Ann.Return':>10} | {'Sharpe':>6} | {'Sortino':>7} | {'MaxDD':>7}")
    print("-" * 58)
    for s in summaries:
        print(f"{s['policy']:>20} | {s['ann_return']:>10.1%} | {s['sharpe']:>6.3f} | {s['sortino']:>7.3f} | {s['max_drawdown']:>7.1%}")


if __name__ == "__main__":
    main()