# src/10_linucb_contextual.py
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


def clip_reward(r: float, c: float | None) -> float:
    return float(r) if c is None else float(np.clip(r, -c, c))


def sharpe_reward(
    history: dict,
    asset_idx: int,
    r_raw: float,
    window: int = 12,
    annualize: bool = True,
) -> float:
    """
    Reward = Sharpe rolling del activo seleccionado.
    history: dict {asset_idx: [retornos pasados]}
    """
    if asset_idx not in history:
        history[asset_idx] = []
    history[asset_idx].append(r_raw)
    buf = history[asset_idx][-window:]
    if len(buf) < 4:
        return r_raw  # fallback a retorno crudo al inicio
    arr = np.array(buf)
    std = arr.std()
    if std < 1e-8:
        return 0.0
    sr = arr.mean() / std
    if annualize:
        sr *= np.sqrt(52)
    return float(np.clip(sr, -3.0, 3.0))


@dataclass
class LinUCBState:
    A_inv: np.ndarray  # (d,d)
    b: np.ndarray      # (d,)


def linucb_init(d: int, lam: float) -> LinUCBState:
    A_inv = (1.0 / lam) * np.eye(d, dtype=np.float64)
    b = np.zeros(d, dtype=np.float64)
    return LinUCBState(A_inv=A_inv, b=b)


def theta_hat(st: LinUCBState) -> np.ndarray:
    return st.A_inv @ st.b


def ucb_scores(st: LinUCBState, X: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X: (K,d) contextos para todos los brazos.
    Retorna (ucb, mu, sigma) con shape (K,).
    """
    th = theta_hat(st)
    mu = X @ th
    XA = X @ st.A_inv
    sigma = np.sqrt(np.sum(XA * X, axis=1))
    ucb = mu + alpha * sigma
    return ucb, mu, sigma


def linucb_update(st: LinUCBState, x: np.ndarray, r: float) -> None:
    """
    Sherman-Morrison para A_inv cuando A <- A + x x^T
    """
    x = x.astype(np.float64, copy=False)
    Ainv = st.A_inv
    Ainv_x = Ainv @ x
    denom = 1.0 + float(x.T @ Ainv_x)
    st.A_inv = Ainv - np.outer(Ainv_x, Ainv_x) / denom
    st.b = st.b + float(r) * x


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
) -> dict:
    rng = np.random.default_rng(seed)

    T = len(dates)
    if T < 2:
        raise ValueError("Need at least 2 snapshots to define next-week reward.")

    missing = [d for d in dates if d not in returns.index]
    if missing:
        raise ValueError(f"Returns missing {len(missing)} snapshot dates. Example: {missing[:5]}")

    Z0 = np.load(context_dir / f"{dates[0].date().isoformat()}.npy")
    K, d = int(Z0.shape[0]), int(Z0.shape[1])
    if K != returns.shape[1]:
        raise ValueError(f"K mismatch: context has {K} arms, returns has {returns.shape[1]} columns")

    st = linucb_init(d=d, lam=lam) if policy in ("linucb", "greedy") else None

    rows = []
    cum_reward, cum_regret = 0.0, 0.0
    last_a = None
    repeats = 0
    theta_norm = None

    sharpe_history: dict = {}  # buffer rolling por activo

    for t in range(T - 1):
        date_t = dates[t]
        date_next = dates[t + 1]

        X = np.load(context_dir / f"{date_t.date().isoformat()}.npy").astype(np.float64)
        assert X.shape == (K, d), f"Bad context shape at {date_t}: {X.shape}"

        # Acción
        if policy == "random":
            a = int(rng.integers(0, K))
            mu_a = sigma_a = ucb_a = None
        elif policy == "linucb":
            ucb, mu, sig = ucb_scores(st, X, alpha=alpha)  # type: ignore[arg-type]
            a = int(np.argmax(ucb))
            mu_a, sigma_a, ucb_a = float(mu[a]), float(sig[a]), float(ucb[a])
        elif policy == "greedy":
            ucb, mu, sig = ucb_scores(st, X, alpha=0.0)    # type: ignore[arg-type]
            a = int(np.argmax(ucb))
            mu_a, sigma_a, ucb_a = float(mu[a]), float(sig[a]), float(ucb[a])
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Reward causal: retorno real de la semana siguiente
        r_raw = float(returns.loc[date_next, returns.columns[a]])

        # Reward usado para aprendizaje
        if reward_mode == "sharpe":
            r = sharpe_reward(sharpe_history, a, r_raw, window=12)
        else:
            r = clip_reward(r_raw, reward_clip)

        # Regret empírico: vs mejor activo en hindsight
        best_next = float(np.max(returns.loc[date_next].values))
        regret_emp = best_next - float(returns.loc[date_next, returns.columns[a]])

        cum_reward += r
        cum_regret += regret_emp

        # Update online
        if policy in ("linucb", "greedy"):
            linucb_update(st, X[a], r)  # type: ignore[arg-type]
            th = theta_hat(st)          # type: ignore[arg-type]
            theta_norm = float(np.linalg.norm(th))

        # Colapso (repetición)
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
        "seed": int(seed),
        "cum_reward": float(cum_reward),
        "cum_regret_emp": float(cum_regret),
        "mean_reward_used": float(log_df["reward_used"].mean()),
        "mean_regret_emp": float(log_df["regret_emp"].mean()),
        "repeat_rate": float(repeats / max(1, (T - 2))),
        "log_path": str(log_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", choices=["embeddings", "X_raw"], default="embeddings")
    ap.add_argument("--returns_path", type=str, default="data/processed/weekly_returns.parquet")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--reward_clip", type=float, default=np.nan,
                    help="Clip reward to [-c,c]. Use NaN to disable.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward_mode", choices=["raw", "sharpe"], default="sharpe",
                    help="Reward signal for LinUCB learning: raw return or rolling Sharpe.")
    args = ap.parse_args()

    context_dir = Path("artifacts/embeddings_gat/npy") if args.context == "embeddings" else Path("artifacts/X_raw/npy")

    returns = load_returns(Path(args.returns_path))
    dates = list_dates_from_dir(context_dir)

    reward_clip = None if np.isnan(args.reward_clip) else float(args.reward_clip)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts/linucb") / f"run_{args.context}_{run_tag}"
    soft_mkdir(out_dir)

    summaries = []
    summaries.append(run_policy("linucb", context_dir, returns, dates, args.alpha, args.lam, reward_clip, args.seed, out_dir, reward_mode=args.reward_mode))
    summaries.append(run_policy("greedy", context_dir, returns, dates, 0.0, args.lam, reward_clip, args.seed, out_dir, reward_mode=args.reward_mode))
    summaries.append(run_policy("random", context_dir, returns, dates, 0.0, args.lam, reward_clip, args.seed, out_dir, reward_mode="raw"))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": summaries}, f, indent=2)

    print(f"[OK] saved to: {out_dir}")
    print(json.dumps({"runs": summaries}, indent=2))


if __name__ == "__main__":
    main()