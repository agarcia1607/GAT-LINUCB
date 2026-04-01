# src/11_linucb_filtered.py
# Same as 10_linucb_contextual.py but with quality filter on asset universe
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


def clip_reward(r: float, c: float | None) -> float:
    return float(r) if c is None else float(np.clip(r, -c, c))


def sharpe_reward(history, asset_idx, r_raw, window=12, annualize=True):
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


def sortino_reward(history, asset_idx, r_raw, window=12, annualize=True):
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
# Quality Filter
# -------------------------------------------------------------------

def apply_quality_filter(returns: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """
    Filters assets with extreme volatility from corporate events
    (bankruptcies, accounting scandals, near-insolvency).

    Criteria:
        - max_weekly_return > 50%  (single week jump — crisis/event)
        - OR (extreme_weeks > 5 AND max_drawdown < -75%)  (sustained distress)

    Returns:
        filtered_returns: DataFrame with valid assets only
        valid_idx: column indices in original DataFrame (for embedding lookup)
    """
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
# LinUCB
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


def ucb_scores(st, X, alpha):
    th = theta_hat(st)
    mu = X @ th
    XA = X @ st.A_inv
    sigma = np.sqrt(np.sum(XA * X, axis=1))
    return mu + alpha * sigma, mu, sigma


def linucb_update(st, x, r):
    x = x.astype(np.float64, copy=False)
    Ainv_x = st.A_inv @ x
    denom = 1.0 + float(x.T @ Ainv_x)
    st.A_inv = st.A_inv - np.outer(Ainv_x, Ainv_x) / denom
    st.b = st.b + float(r) * x


# -------------------------------------------------------------------
# run_policy
# -------------------------------------------------------------------

def run_policy(
    policy: str,
    context_dir: Path,
    returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    alpha: float,
    lam: float,
    reward_clip: float | None,
    seed: int,
    out_dir: Path,
    reward_mode: str = "raw",
    valid_idx: list[int] | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    T = len(dates)
    if T < 2:
        raise ValueError("Need at least 2 snapshots.")

    missing = [d for d in dates if d not in returns.index]
    if missing:
        raise ValueError(f"Returns missing {len(missing)} dates.")

    # Load first snapshot to get K and d
    Z0_full = np.load(context_dir / f"{dates[0].date().isoformat()}.npy")
    Z0 = Z0_full[valid_idx] if valid_idx is not None else Z0_full
    K, d = int(Z0.shape[0]), int(Z0.shape[1])

    if K != returns.shape[1]:
        raise ValueError(f"K mismatch: embeddings {K}, returns {returns.shape[1]}")

    st = linucb_init(d=d, lam=lam) if policy in ("linucb", "greedy") else None

    rows = []
    cum_reward, cum_regret = 0.0, 0.0
    last_a = None
    repeats = 0
    theta_norm = None
    reward_history: dict = {}

    for t in range(T - 1):
        date_t = dates[t]
        date_next = dates[t + 1]

        X_full = np.load(context_dir / f"{date_t.date().isoformat()}.npy").astype(np.float64)
        X = X_full[valid_idx] if valid_idx is not None else X_full
        assert X.shape == (K, d)

        if policy == "random":
            a = int(rng.integers(0, K))
            mu_a = sigma_a = ucb_a = None
        elif policy == "linucb":
            ucb, mu, sig = ucb_scores(st, X, alpha=alpha)
            a = int(np.argmax(ucb))
            mu_a, sigma_a, ucb_a = float(mu[a]), float(sig[a]), float(ucb[a])
        elif policy == "greedy":
            ucb, mu, sig = ucb_scores(st, X, alpha=0.0)
            a = int(np.argmax(ucb))
            mu_a, sigma_a, ucb_a = float(mu[a]), float(sig[a]), float(ucb[a])
        else:
            raise ValueError(f"Unknown policy: {policy}")

        r_raw = float(returns.loc[date_next, returns.columns[a]])

        if reward_mode == "sharpe":
            r = sharpe_reward(reward_history, a, r_raw, window=12)
        elif reward_mode == "sortino":
            r = sortino_reward(reward_history, a, r_raw, window=12)
        else:
            r = clip_reward(r_raw, reward_clip)

        best_next = float(np.max(returns.loc[date_next].values))
        regret_emp = best_next - float(returns.loc[date_next, returns.columns[a]])

        cum_reward += r
        cum_regret += regret_emp

        if policy in ("linucb", "greedy"):
            linucb_update(st, X[a], r)
            theta_norm = float(np.linalg.norm(theta_hat(st)))

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
            "sigma_a": sigma_a,
            "ucb_a": ucb_a,
            "theta_norm": theta_norm,
        })

    log_df = pd.DataFrame(rows)
    log_path = out_dir / f"logs_{policy}.csv"
    log_df.to_csv(log_path, index=False)

    return {
        "policy": policy,
        "context_dir": str(context_dir),
        "T_effective": int(T - 1),
        "K": int(K),
        "d": int(d),
        "alpha": float(alpha),
        "lambda": float(lam),
        "reward_clip": None if reward_clip is None else float(reward_clip),
        "reward_mode": reward_mode,
        "quality_filter": valid_idx is not None,
        "seed": int(seed),
        "cum_reward": float(cum_reward),
        "cum_regret_emp": float(cum_regret),
        "mean_reward_used": float(log_df["reward_used"].mean()),
        "mean_regret_emp": float(log_df["regret_emp"].mean()),
        "repeat_rate": float(repeats / max(1, (T - 2))),
        "log_path": str(log_path),
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", choices=["embeddings", "X_raw"], default="embeddings")
    ap.add_argument("--returns_path", type=str, default="data/processed/weekly_returns.parquet")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--reward_clip", type=float, default=float('nan'))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward_mode", choices=["raw", "sharpe", "sortino"], default="sharpe",
                    help="Reward signal: raw return, rolling Sharpe, or rolling Sortino.")
    args = ap.parse_args()

    context_dir = (Path("artifacts/embeddings_gat/npy")
                   if args.context == "embeddings"
                   else Path("artifacts/X_raw/npy"))

    returns_raw = load_returns(Path(args.returns_path))
    dates = list_dates_from_dir(context_dir)
    reward_clip = None if math.isnan(args.reward_clip) else float(args.reward_clip)

    # Apply quality filter
    returns_filtered, valid_idx = apply_quality_filter(returns_raw)
    filtered_tickers = [t for t in returns_raw.columns if t not in returns_filtered.columns]
    print(f"[INFO] Universe: {returns_raw.shape[1]} → {returns_filtered.shape[1]} assets after quality filter")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts/linucb") / f"run_{args.context}_filtered_{run_tag}"
    soft_mkdir(out_dir)

    summaries = []
    summaries.append(run_policy("linucb", context_dir, returns_filtered, dates,
                                args.alpha, args.lam, reward_clip, args.seed, out_dir,
                                reward_mode=args.reward_mode, valid_idx=valid_idx))
    summaries.append(run_policy("greedy", context_dir, returns_filtered, dates,
                                0.0, args.lam, reward_clip, args.seed, out_dir,
                                reward_mode=args.reward_mode, valid_idx=valid_idx))
    summaries.append(run_policy("random", context_dir, returns_filtered, dates,
                                0.0, args.lam, reward_clip, args.seed, out_dir,
                                reward_mode="raw", valid_idx=valid_idx))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "runs": summaries,
            "quality_filter": True,
            "filtered_tickers": filtered_tickers,
            "universe_original": returns_raw.shape[1],
            "universe_filtered": returns_filtered.shape[1],
        }, f, indent=2)

    print(f"[OK] saved to: {out_dir}")
    print(json.dumps({"runs": summaries}, indent=2))


if __name__ == "__main__":
    main()