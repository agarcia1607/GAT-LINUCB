"""
dashboard/app.py — GAT-LINUCB Financial Dashboard
Run: streamlit run dashboard/app.py
"""
import os
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="GAT-LINUCB Dashboard",
    page_icon="📈",
    layout="wide",
)

LINUCB_ROOT = Path("artifacts/linucb")
WEEKS = 52

CRISIS_EVENTS = {
    "COVID Crash": "2020-02-21",
    "Bear 2022": "2022-01-03",
    "China/Brexit": "2016-08-01",
}

FILTERED_ASSETS = ['PCG','SMCI','NCLH','WBD','APA','BA','UAL','OXY','RCL','SPG','VTR','WELL','DHR']

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def sharpe(r): return np.sqrt(WEEKS)*r.mean()/r.std() if r.std()>0 else 0
def sortino(r):
    down = r[r<0].std()
    return np.sqrt(WEEKS)*r.mean()/down if down>0 else 0
def ann_return(r): return (1+r).prod()**(WEEKS/len(r))-1
def max_dd(r):
    cum=(1+r).cumprod()
    return ((cum-cum.cummax())/(cum.cummax()+1)).min()
def volatility(r): return r.std()*np.sqrt(WEEKS)

def compute_metrics(r, label):
    return {
        "Policy": label,
        "Ann. Return": f"{ann_return(r):.1%}",
        "Sharpe": f"{sharpe(r):.3f}",
        "Sortino": f"{sortino(r):.3f}",
        "Max Drawdown": f"{max_dd(r):.1%}",
        "Volatility": f"{volatility(r):.1%}",
    }

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_run(pattern, min_weeks=400):
    runs = sorted([p for p in LINUCB_ROOT.glob(f"{pattern}/") if p.is_dir()])
    for run in reversed(runs):
        try:
            df = pd.read_csv(run / "logs_linucb.csv")
            if len(df) >= min_weeks:
                return {
                    "linucb": df,
                    "greedy": pd.read_csv(run / "logs_greedy.csv"),
                    "random": pd.read_csv(run / "logs_random.csv"),
                    "run_dir": str(run),
                    "run_name": run.name,
                }
        except: continue
    return None

@st.cache_data(ttl=3600)
def load_sp500(start, end):
    sp = yf.download("^GSPC", start=start, end=end, interval="1wk", progress=False)
    return sp["Close"].pct_change().dropna().squeeze()

def get_crisis_indices(dates):
    crisis_idx = {}
    dates_ts = pd.to_datetime(dates)
    for name, date in CRISIS_EVENTS.items():
        idx = int((dates_ts - pd.Timestamp(date)).abs().argmin())
        crisis_idx[name] = idx
    return crisis_idx

def recovery_periods(r, dates, threshold=-0.10):
    cum = (1+r).cumprod().values
    peak = cum[0]; peak_idx = 0
    in_dd = False; trough_idx = None; trough_val = None
    events = []
    for i in range(1, len(cum)):
        if cum[i] > peak:
            if in_dd and trough_idx is not None:
                dd_pct = (trough_val - peak) / peak
                events.append({
                    "Peak": pd.to_datetime(dates.iloc[peak_idx]).strftime("%Y-%m-%d"),
                    "Trough": pd.to_datetime(dates.iloc[trough_idx]).strftime("%Y-%m-%d"),
                    "Recovery": pd.to_datetime(dates.iloc[i]).strftime("%Y-%m-%d"),
                    "Drawdown": f"{dd_pct:.1%}",
                    "Wks to bottom": trough_idx - peak_idx,
                    "Wks to recover": i - trough_idx,
                    "Total wks": i - peak_idx,
                })
                in_dd = False
            peak = cum[i]; peak_idx = i
        else:
            dd = (cum[i] - peak) / peak
            if dd < threshold:
                if not in_dd:
                    in_dd = True; trough_idx = i; trough_val = cum[i]
                elif cum[i] < trough_val:
                    trough_idx = i; trough_val = cum[i]
    return pd.DataFrame(events)

# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
st.title("📈 GAT-LINUCB — Financial Dashboard")
st.caption("Graph Attention Networks + Contextual Bandits for Asset Selection · Universidad Nacional de Colombia")

# Load filtered (primary) and unfiltered (comparison)
data = load_run("run_embeddings_filtered_*", min_weeks=400)
data_raw = load_run("run_embeddings_2*", min_weeks=400)

if data is None:
    st.error("No filtered runs found. Run: python -m src.11_linucb_filtered")
    st.stop()

r = data["linucb"]["reward_raw"].reset_index(drop=True)
dates = pd.to_datetime(data["linucb"]["date_t"]).reset_index(drop=True)
T = len(r)
t1, t2 = T//3, 2*T//3

sp500_raw = load_sp500(dates.min(), dates.max())
sp_r = pd.Series(sp500_raw.values[:T])
crisis_idx = get_crisis_indices(dates)

st.caption(f"Period: {dates.iloc[0].strftime('%Y-%m-%d')} → {dates.iloc[-1].strftime('%Y-%m-%d')} · {T} weeks · K=453 assets (quality-filtered)")

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Performance",
    "🎲 Combinatorial",
    "🔬 Ablation Study",
    "🧹 Quality Filter",
    "📉 Crisis Analysis",
    "⚙️ How It Works",
    "❓ FAQ",
])

# ===================================================================
# TAB 1 — Performance
# ===================================================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ann. Return", f"{ann_return(r):.1%}", f"vs S&P {ann_return(sp_r):.1%}")
    col2.metric("Sharpe Ratio", f"{sharpe(r):.3f}", f"vs S&P {sharpe(sp_r):.3f}")
    col3.metric("Sortino Ratio", f"{sortino(r):.3f}", f"vs S&P {sortino(sp_r):.3f}")
    col4.metric("Max Drawdown", f"{max_dd(r):.1%}", f"vs S&P {max_dd(sp_r):.1%}", delta_color="inverse")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Cumulative Return vs S&P 500")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot((1+r).cumprod().values, label="LinUCB+GAT (filtered)", color="steelblue", linewidth=2)
        ax.plot((1+sp_r).cumprod().values, label="S&P 500", color="gray", linestyle="--", linewidth=1.5)
        ax.axvline(t1, color="orange", linestyle=":", alpha=0.6, label="Phase boundaries")
        ax.axvline(t2, color="orange", linestyle=":", alpha=0.6)
        for name, idx in crisis_idx.items():
            if idx < T:
                ax.axvline(idx, color="red", linestyle=":", alpha=0.5)
                ax.text(idx+1, ax.get_ylim()[1]*0.9, name.split()[0], fontsize=7, color="red", rotation=90)
        ax.set_xlabel("Weeks")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.subheader("Rolling Sharpe (12-week window)")
        fig, ax = plt.subplots(figsize=(8, 4))
        rs = r.rolling(12).apply(lambda x: np.sqrt(52)*x.mean()/x.std() if x.std()>0 else 0)
        rs_sp = sp_r.rolling(12).apply(lambda x: np.sqrt(52)*x.mean()/x.std() if x.std()>0 else 0)
        ax.plot(rs.values, color="steelblue", linewidth=1.5, label="LinUCB (filtered)")
        ax.plot(rs_sp.values, color="gray", linestyle="--", linewidth=1, label="S&P 500")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(t1, color="orange", linestyle=":", alpha=0.6)
        ax.axvline(t2, color="orange", linestyle=":", alpha=0.6)
        for name, idx in crisis_idx.items():
            if idx < T:
                ax.axvline(idx, color="red", linestyle=":", alpha=0.5)
        ax.set_xlabel("Weeks")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Phase analysis
    st.subheader("Convergence Phase Analysis")
    phase_rows = []
    for label, s, e in [("🔴 Early — exploration", 0, t1), ("🟡 Mid — convergence", t1, t2), ("🟢 Late — exploitation", t2, T)]:
        ry = r.iloc[s:e]
        sp_y = sp_r.iloc[s:e]
        phase_rows.append({
            "Phase": label,
            "Period": f"t={s}–{e}",
            "LinUCB Return": f"{ann_return(ry):.1%}",
            "S&P 500 Return": f"{ann_return(sp_y):.1%}",
            "LinUCB Sharpe": f"{sharpe(ry):.3f}",
            "LinUCB MaxDD": f"{max_dd(ry):.1%}",
            "Unique Assets": data["linucb"]["asset"].iloc[s:e].nunique(),
        })
    st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Annual performance
    st.subheader("Annual Performance vs S&P 500")
    data["linucb"]["year"] = dates.dt.year
    annual_rows = []
    for year, group in data["linucb"].groupby("year"):
        ry = group["reward_raw"].reset_index(drop=True)
        idx = group.index - data["linucb"].index[0]
        sp_y = sp_r.iloc[idx].reset_index(drop=True) if idx.max() < len(sp_r) else sp_r.iloc[:len(ry)].reset_index(drop=True)
        if len(ry) > 4:
            winner = "✅ LinUCB" if ann_return(ry) > ann_return(sp_y) else "📉 S&P500"
            annual_rows.append({
                "Year": year,
                "LinUCB Return": f"{ann_return(ry):.1%}",
                "S&P 500 Return": f"{ann_return(sp_y):.1%}",
                "LinUCB Sharpe": f"{sharpe(ry):.3f}",
                "Winner": winner,
            })
    st.dataframe(pd.DataFrame(annual_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Recent windows
    st.subheader("Recent Windows — Converged System")
    window_rows = []
    for label, n in [("Last 8w", 8), ("Last 12w", 12), ("Last 26w", 26), ("Last 52w", 52)]:
        ry = r.tail(n)
        window_rows.append({
            "Window": label,
            "Ann. Return": f"{ann_return(ry):.1%}",
            "Sharpe": f"{sharpe(ry):.3f}",
            "Sortino": f"{sortino(ry):.3f}",
            "Max Drawdown": f"{max_dd(ry):.1%}",
        })
    st.dataframe(pd.DataFrame(window_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Last 12 selections
    st.subheader("Last 12 Weeks — Asset Selections")
    last12 = data["linucb"].tail(12)[["date_t","asset","reward_raw","mu_a","sigma_a"]].copy()
    last12.columns = ["Date","Asset","Weekly Return","μ expected","σ uncertainty"]
    last12["Weekly Return"] = last12["Weekly Return"].map("{:.2%}".format)
    last12["μ expected"] = last12["μ expected"].map("{:.4f}".format)
    last12["σ uncertainty"] = last12["σ uncertainty"].map("{:.4f}".format)
    st.dataframe(last12, use_container_width=True, hide_index=True)


# ===================================================================
# TAB 2 — Combinatorial
# ===================================================================
with tab2:
    st.subheader("🎲 Combinatorial LinUCB — k Assets per Week")
    st.markdown("""
    Instead of selecting **1 asset**, CombLinUCB selects the **top-k assets by UCB score** simultaneously.
    The reward signal is the **rolling Sharpe of the equally-weighted k-asset portfolio** — forcing
    the system to learn diversification, not just momentum.
    """)

    # Load combinatorial runs
    @st.cache_data(ttl=3600)
    def load_combinatorial():
        results = {}
        for run in reversed(sorted([p for p in LINUCB_ROOT.glob("run_combinatorial_*/") if p.is_dir()])):
            try:
                summary = json.load(open(run / "summary.json"))
                for s in summary.get("runs", []):
                    k = s.get("k")
                    if k and k not in results:
                        log_path = Path(s["log_path"])
                        if log_path.exists():
                            df = pd.read_csv(log_path)
                            results[k] = {"summary": s, "df": df}
                if len(results) >= 3:
                    break
            except: continue
        return results

    comb_data = load_combinatorial()

    if not comb_data:
        st.warning("No combinatorial runs found. Run: `python3 -m src.12_linucb_combinatorial --k_values 3,5,10`")
    else:
        # Metrics table
        st.subheader("Results vs Single Asset vs S&P 500")
        comb_rows = []

        # Add k=1 (filtered single asset)
        comb_rows.append({
            "Policy": "LinUCB k=1 (single asset)",
            "Ann. Return": f"{ann_return(r):.1%}",
            "Sharpe": f"{sharpe(r):.3f}",
            "Sortino": f"{sortino(r):.3f}",
            "Max Drawdown": f"{max_dd(r):.1%}",
        })

        for k in sorted(comb_data.keys()):
            d = comb_data[k]
            r_k = d["df"]["portfolio_return"]
            comb_rows.append({
                "Policy": f"CombLinUCB k={k}",
                "Ann. Return": f"{ann_return(r_k):.1%}",
                "Sharpe": f"{sharpe(r_k):.3f}",
                "Sortino": f"{sortino(r_k):.3f}",
                "Max Drawdown": f"{max_dd(r_k):.1%}",
            })

        comb_rows.append({
            "Policy": "S&P 500 (benchmark)",
            "Ann. Return": f"{ann_return(sp_r):.1%}",
            "Sharpe": f"{sharpe(sp_r):.3f}",
            "Sortino": f"{sortino(sp_r):.3f}",
            "Max Drawdown": f"{max_dd(sp_r):.1%}",
        })

        st.dataframe(pd.DataFrame(comb_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Charts
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Cumulative Return — All k values")
            fig, ax = plt.subplots(figsize=(8, 4))
            colors_k = {1: "steelblue", 3: "orange", 5: "green", 10: "purple"}
            ax.plot((1+r).cumprod().values, label="k=1 (single)", color="steelblue", linewidth=2)
            for k in sorted(comb_data.keys()):
                r_k = comb_data[k]["df"]["portfolio_return"]
                ax.plot((1+r_k).cumprod().values, label=f"k={k}", color=colors_k.get(k, "gray"), linewidth=1.5, linestyle="--")
            ax.plot((1+sp_r).cumprod().values, label="S&P 500", color="gray", linestyle=":", linewidth=1.5)
            for name, idx in crisis_idx.items():
                if idx < T:
                    ax.axvline(idx, color="red", linestyle=":", alpha=0.4)
            ax.set_xlabel("Weeks")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col_r:
            st.subheader("Rolling Sharpe — All k values")
            fig, ax = plt.subplots(figsize=(8, 4))
            rs_k1 = r.rolling(12).apply(lambda x: np.sqrt(52)*x.mean()/x.std() if x.std()>0 else 0)
            ax.plot(rs_k1.values, label="k=1", color="steelblue", linewidth=2)
            for k in sorted(comb_data.keys()):
                r_k = comb_data[k]["df"]["portfolio_return"]
                rs = r_k.rolling(12).apply(lambda x: np.sqrt(52)*x.mean()/x.std() if x.std()>0 else 0)
                ax.plot(rs.values, label=f"k={k}", color=colors_k.get(k, "gray"), linewidth=1.5, linestyle="--")
            rs_sp = sp_r.rolling(12).apply(lambda x: np.sqrt(52)*x.mean()/x.std() if x.std()>0 else 0)
            ax.plot(rs_sp.values, label="S&P 500", color="gray", linestyle=":", linewidth=1)
            ax.axhline(0, color="black", linewidth=0.5)
            for name, idx in crisis_idx.items():
                if idx < T:
                    ax.axvline(idx, color="red", linestyle=":", alpha=0.4)
            ax.set_xlabel("Weeks")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

        st.divider()

        # Drawdown comparison
        st.subheader("Drawdown — Diversification Effect")
        fig, ax = plt.subplots(figsize=(12, 4))
        cum_k1 = (1+r).cumprod()
        dd_k1 = (cum_k1 - cum_k1.cummax()) / cum_k1.cummax()
        ax.plot(dd_k1.values, label="k=1", color="steelblue", linewidth=2)
        for k in sorted(comb_data.keys()):
            r_k = comb_data[k]["df"]["portfolio_return"]
            cum_k = (1+r_k).cumprod()
            dd_k = (cum_k - cum_k.cummax()) / cum_k.cummax()
            ax.plot(dd_k.values, label=f"k={k}", color=colors_k.get(k, "gray"), linewidth=1.5, linestyle="--")
        cum_sp = (1+sp_r).cumprod()
        dd_sp = (cum_sp - cum_sp.cummax()) / cum_sp.cummax()
        ax.plot(dd_sp.values, label="S&P 500", color="gray", linestyle=":", linewidth=1)
        for name, idx in crisis_idx.items():
            if idx < T:
                ax.axvline(idx, color="red", linestyle=":", alpha=0.5)
                ax.text(idx+1, -0.05, name.split()[0], fontsize=7, color="red", rotation=90)
        ax.set_xlabel("Weeks")
        ax.set_ylabel("Drawdown")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

        st.divider()

        # Last 12 weeks per k
        st.subheader("Last 12 Weeks — Portfolio Compositions")
        for k in sorted(comb_data.keys()):
            st.markdown(f"**k={k}**")
            df_k = comb_data[k]["df"].tail(12)[["date_t","assets","portfolio_return"]].copy()
            df_k.columns = ["Date","Assets Selected","Portfolio Return"]
            df_k["Portfolio Return"] = df_k["Portfolio Return"].map("{:.2%}".format)
            st.dataframe(df_k, use_container_width=True, hide_index=True)

        st.info("""
        **Key finding:** k=10 achieves **Sharpe 0.867** — beating S&P 500 (0.843) — while reducing
        max drawdown from -43% (k=1) to -32%. Diversification across correlated assets learned
        via GAT embeddings reduces portfolio volatility without sacrificing return.
        """)


# ===================================================================
# TAB 3 — Ablation Study (was tab2)
# ===================================================================
with tab3:
    st.subheader("🔬 Ablation Study — Does GAT Actually Help?")
    st.markdown("Same algorithm (LinUCB), same hyperparameters (α=2.0, Sharpe reward, 10 years), different context:")

    abl_data = {
        "GAT embeddings (d=16)": None,
        "Raw features (d=2)": None,
        "Random embeddings (d=16)": None,
    }

    for run in reversed(sorted([p for p in LINUCB_ROOT.glob("run_embeddings_2*/") if p.is_dir()])):
        try:
            df = pd.read_csv(run / "logs_linucb.csv")
            if len(df) > 400 and abl_data["GAT embeddings (d=16)"] is None:
                abl_data["GAT embeddings (d=16)"] = df["reward_raw"].reset_index(drop=True)
        except: continue

    for run in reversed(sorted([p for p in LINUCB_ROOT.glob("run_X_raw_*/") if p.is_dir()])):
        try:
            df = pd.read_csv(run / "logs_linucb.csv")
            if len(df) > 400 and abl_data["Raw features (d=2)"] is None:
                abl_data["Raw features (d=2)"] = df["reward_raw"].reset_index(drop=True)
        except: continue

    for run in reversed(sorted([p for p in LINUCB_ROOT.glob("run_random_*/") if p.is_dir()])):
        try:
            df = pd.read_csv(run / "logs_linucb.csv")
            if len(df) > 400 and abl_data["Random embeddings (d=16)"] is None:
                abl_data["Random embeddings (d=16)"] = df["reward_raw"].reset_index(drop=True)
        except: continue

    abl_rows = []
    for label, rv in abl_data.items():
        if rv is not None:
            abl_rows.append({
                "Context": f"LinUCB + {label}",
                "Ann. Return": f"{ann_return(rv):.1%}",
                "Sharpe": f"{sharpe(rv):.3f}",
                "Max Drawdown": f"{max_dd(rv):.1%}",
            })
    if abl_rows:
        st.dataframe(pd.DataFrame(abl_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Cumulative Return — All Contexts")
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = {"GAT embeddings (d=16)": "steelblue", "Raw features (d=2)": "orange", "Random embeddings (d=16)": "green"}
        for label, rv in abl_data.items():
            if rv is not None:
                ax.plot((1+rv).cumprod().values, label=f"LinUCB + {label}", color=colors.get(label,"gray"), linewidth=2)
        ax.plot((1+sp_r).cumprod().values, label="S&P 500", color="gray", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Weeks")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.info("**Key finding:** GAT embeddings (60% ann.) vastly outperform raw features (23.4%) and random embeddings (2.7%). The 57pp gap cannot be explained by chance — the graph correlation structure contains genuine market information.")


# ===================================================================
# TAB 4 — Quality Filter (was tab3)
# ===================================================================
with tab4:
    st.subheader("🧹 Quality Filter — Removing Distressed Assets")

    st.markdown("""
    ### The Problem

    Without filtering, the system achieves **60% annualized** — but this is inflated by extreme volatility
    from corporate distress events, not genuine market signal:

    - **PCG (PG&E)** — selected 46 weeks during its 2019 **bankruptcy process**. Rolling Sharpe appeared high
      because large positive weeks dominated the 12-week window during its court-driven recovery.
    - **SMCI (Super Micro Computer)** — selected 66 weeks around its 2024 **accounting fraud investigation**.
    - **NCLH (Norwegian Cruise)** — selected 32 weeks during COVID **near-insolvency** rebound.

    LinUCB correctly optimized its objective. The problem is that rolling Sharpe cannot distinguish between:
    - ✅ **Genuine momentum** — asset growing due to business performance
    - ❌ **Distress rebound** — asset recovering from near-collapse
    """)

    st.divider()

    st.markdown("""
    ### The Filter

    Assets are excluded if they exhibit extreme historical volatility patterns:

    ```python
    # Exclude if:
    # 1. Single week return > 50%  →  crisis/bankruptcy event
    # 2. OR (>5 weeks with |return|>30%) AND max_drawdown < -75%  →  sustained distress

    filtered = (max_weekly_return > 0.50) or
               (extreme_weeks > 5 and max_drawdown < -0.75)
    ```
    """)

    st.subheader("Filtered Assets (13 of 466)")
    filter_data = []
    filter_reasons = {
        "PCG": "PG&E — bankruptcy 2019",
        "SMCI": "Super Micro — accounting fraud 2024",
        "NCLH": "Norwegian Cruise — COVID near-insolvency",
        "WBD": "Warner Bros Discovery — extreme debt load",
        "APA": "APA Corp — oil price collapse",
        "BA": "Boeing — 737 MAX crisis",
        "UAL": "United Airlines — COVID near-insolvency",
        "OXY": "Occidental — oil price collapse",
        "RCL": "Royal Caribbean — COVID near-insolvency",
        "SPG": "Simon Property — COVID real estate crisis",
        "VTR": "Ventas — COVID healthcare REIT crisis",
        "WELL": "Welltower — COVID healthcare REIT crisis",
        "DHR": "Danaher — extreme single-week moves",
    }
    for ticker in FILTERED_ASSETS:
        filter_data.append({
            "Ticker": ticker,
            "Reason": filter_reasons.get(ticker, "Extreme volatility"),
        })
    st.dataframe(pd.DataFrame(filter_data), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Impact of the Filter")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Without filter (466 assets)**")
        if data_raw:
            r_raw = data_raw["linucb"]["reward_raw"].reset_index(drop=True)
            rows_raw = [
                compute_metrics(r_raw, "LinUCB (no filter)"),
                compute_metrics(sp_r, "S&P 500"),
            ]
            st.dataframe(pd.DataFrame(rows_raw), use_container_width=True, hide_index=True)

            # Year 2019
            df_raw = data_raw["linucb"].copy()
            df_raw["year"] = pd.to_datetime(df_raw["date_t"]).dt.year
            r2019 = df_raw[df_raw["year"]==2019]["reward_raw"]
            if len(r2019) > 4:
                st.metric("2019 (PCG bankruptcy)", f"{ann_return(r2019):.1%}", "Artificially inflated")

    with col2:
        st.markdown("**With filter (453 assets)**")
        rows_filt = [
            compute_metrics(r, "LinUCB (filtered)"),
            compute_metrics(sp_r, "S&P 500"),
        ]
        st.dataframe(pd.DataFrame(rows_filt), use_container_width=True, hide_index=True)

        df_filt = data["linucb"].copy()
        df_filt["year"] = pd.to_datetime(df_filt["date_t"]).dt.year
        r2019f = df_filt[df_filt["year"]==2019]["reward_raw"]
        if len(r2019f) > 4:
            st.metric("2019 (filtered)", f"{ann_return(r2019f):.1%}", "Genuine signal")

    st.info("""
    **Key insight:** The filter reduces headline return from 60% to 22.5%, but reveals the genuine signal.
    Most importantly, the filtered system **beats the market in 2022** (+7.4% vs -14.5%) —
    something the unfiltered system failed to do (-26.2%).
    """)


# ===================================================================
# TAB 5 — Crisis Analysis (was tab4)
# ===================================================================
with tab5:
    st.subheader("📉 Crisis Analysis — Recovery Periods")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**LinUCB + GAT (filtered)**")
        ev_l = recovery_periods(r, dates, threshold=-0.10)
        st.dataframe(ev_l, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("**S&P 500**")
        ev_s = recovery_periods(sp_r, dates, threshold=-0.10)
        st.dataframe(ev_s, use_container_width=True, hide_index=True)

    st.divider()

    # COVID comparison
    st.subheader("🦠 COVID-19 Crash — Head to Head")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**LinUCB + GAT**")
        covid_l = ev_l[ev_l["Peak"].str.startswith("2020-02")] if len(ev_l) else pd.DataFrame()
        if len(covid_l):
            row = covid_l.iloc[0]
            st.metric("Drawdown", row["Drawdown"])
            st.metric("Weeks to bottom", row["Wks to bottom"])
            st.metric("Weeks to recover", row["Wks to recover"])
            st.metric("Total weeks under water", row["Total wks"])
    with col2:
        st.markdown("**S&P 500**")
        covid_s = ev_s[ev_s["Peak"].str.startswith("2020-01")] if len(ev_s) else pd.DataFrame()
        if len(covid_s):
            row = covid_s.iloc[0]
            st.metric("Drawdown", row["Drawdown"])
            st.metric("Weeks to bottom", row["Wks to bottom"])
            st.metric("Weeks to recover", row["Wks to recover"])
            st.metric("Total weeks under water", row["Total wks"])

    st.divider()

    # Drawdown chart
    st.subheader("Drawdown — Full Period")
    fig, ax = plt.subplots(figsize=(12, 4))
    cum_l = (1+r).cumprod()
    dd_l = (cum_l - cum_l.cummax()) / cum_l.cummax()
    cum_s = (1+sp_r).cumprod()
    dd_s = (cum_s - cum_s.cummax()) / cum_s.cummax()
    ax.fill_between(range(len(dd_l)), dd_l.values, 0, alpha=0.5, color="steelblue", label="LinUCB+GAT (filtered)")
    ax.fill_between(range(len(dd_s)), dd_s.values, 0, alpha=0.4, color="gray", label="S&P 500")
    for name, idx in crisis_idx.items():
        if idx < T:
            ax.axvline(idx, color="red", linestyle=":", alpha=0.7)
            ax.text(idx+1, -0.05, name, fontsize=7, color="red", rotation=90)
    ax.set_xlabel("Weeks")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()


# ===================================================================
# TAB 6 — How It Works (was tab5)
# ===================================================================
with tab6:
    st.subheader("⚙️ How the System Works")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Input Variables")
        st.markdown("""
        The system uses **only historical price data** — no fundamentals, no news, no macro.

        **Node features (2 per asset):**
        - **Momentum** — average return over last 4 weeks
        - **Volatility** — standard deviation over last 4 weeks

        **Graph edges:**
        - **Rolling Pearson correlation** (24-week window)
        - kNN connectivity (k=8)
        - Updated weekly — dynamic market structure

        **Universe:** 453 S&P 500 companies (after quality filter)
        """)

        st.markdown("### Causal Integrity — No Look-Ahead Bias")
        st.markdown("""
        Every decision at week `t` uses **exclusively data up to `t-1``:
        ```
        F_{t-1} → embedding(t) → action(t) → reward(t+1)
        ```
        Enforced by explicit assertions in `src/lib/filtration.py`.
        """)

        st.markdown("### Key Design Decisions")
        st.markdown("""
        | Decision | Choice | Validated by |
        |---|---|---|
        | Reward | Rolling Sharpe | Grid search vs raw, Sortino |
        | Alpha | 2.0 | Grid search [0.1–3.0] |
        | Embedding dim | d=16 | Ablation study |
        | Quality filter | max_wk>50% or distress | Manual inspection of PCG/SMCI |
        """)

    with col_r:
        st.markdown("### Pipeline")
        st.markdown("""
        ```
        Yahoo Finance prices (2016–2026)
                │
                ▼
        Quality Filter
        (remove 13 distressed assets)
                │
                ▼
        Weekly returns (453 assets)
                │
                ▼
        Rolling correlation graph
        (24-week window, kNN k=8)
                │
                ▼
        Graph Attention Network (GAT)
        → 16-dim embedding per asset
                │
                ▼
        LinUCB contextual bandit
        → selects 1 asset per week
        → reward: rolling Sharpe (w=12)
                │
                ▼
        Online update (Sherman-Morrison)
        O(d²) per step — instantaneous
        ```
        """)

        st.markdown("### Convergence — Why Phases Exist")
        st.markdown(f"""
        With K=453 assets and random initialization, LinUCB needs time to explore.
        Regret bound: **O(d√T log T)** — sublinear, guarantees eventual convergence.

        | Phase | Weeks | LinUCB | S&P 500 |
        |---|---|---|---|
        | 🔴 Exploration | 0–{t1} | 5.2% | 11.2% |
        | 🟡 Convergence | {t1}–{t2} | 29.6% | 8.4% |
        | 🟢 Exploitation | {t2}–{T} | **34.6%** | 20.2% |

        **Solution:** walk-forward — train on 2016–2022, deploy from 2023 already converged.
        """)


# ===================================================================
# TAB 7 — FAQ (was tab6)
# ===================================================================
with tab7:
    st.subheader("❓ Frequently Asked Questions")

    with st.expander("What assets does the system select?"):
        st.markdown("""
        **Universe:** 453 of the 466 S&P 500 companies (13 quality-filtered).
        Includes Apple, Microsoft, Google, all major sectors.

        Recent selections concentrate in **semiconductors and technology:**
        SNPS (Synopsys), STX (Seagate), MU (Micron), WDC (Western Digital), TPL (Texas Pacific Land).

        The system learns to concentrate on assets with consistent weekly Sharpe — typically
        technology and growth companies in bull markets.
        """)

    with st.expander("Does it buy or sell? How long does it hold?"):
        st.markdown("""
        **Selects one asset per week**, holds exactly one week, then rotates entirely.

        No stop-loss, no position sizing, no leverage.

        **This is a research prototype.** In production, weekly full rotation would incur
        significant transaction costs (commissions + slippage) that are not modeled here.
        """)

    with st.expander("Why does it pick distressed assets without the filter?"):
        st.markdown("""
        LinUCB optimizes rolling Sharpe — which appears high during distressed-asset rebounds
        because large positive weeks dominate the 12-week window.

        **Example:** PG&E (PCG) during its 2019 bankruptcy had weeks of +15-20% as courts
        approved recovery plans. The 12-week rolling Sharpe looked excellent. LinUCB selected
        it 46 times that year — correctly optimizing its objective, but on noise, not signal.

        The quality filter removes assets where this pattern is structurally likely:
        max weekly return > 50% or sustained distress (>5 extreme weeks + deep drawdown).
        """)

    with st.expander("How did it perform during COVID-19?"):
        st.markdown("""
        **COVID crash (Feb–May 2020) — filtered system:**
        - Drawdown: -47.6% vs S&P 500 -28.6%
        - Weeks to bottom: 3 vs 5
        - **Weeks to full recovery: 15 vs 27 (S&P 500)**

        Fell more but recovered **2x faster.** Full year 2020: +6.6% vs +16.2% (S&P 500).

        Note: without filter, 2020 showed +126.5% — mostly due to NCLH (Norwegian Cruise)
        COVID rebound. The filtered result of +6.6% is the genuine signal.
        """)

    with st.expander("Is this luck or real signal?"):
        st.markdown("""
        The ablation study provides the strongest evidence of real signal:

        | Context | 10-Year Return | Sharpe |
        |---|---|---|
        | **LinUCB + GAT** | 60.0% | 1.072 |
        | LinUCB + Raw features | 23.4% | 0.655 |
        | LinUCB + Random embeddings | 2.7% | 0.256 |

        Same algorithm, same hyperparameters — only the context changes.
        A 57pp gap between GAT and Random cannot be explained by luck.

        **Honest caveat:** validated on 2016–2026, mostly bull market.
        The 2022 bear market (+7.4% vs -14.5%) is the strongest stress test available.
        20+ year validation with historical S&P 500 universe is the rigorous next step.
        """)

    with st.expander("What are the next steps?"):
        st.markdown("""
        1. **Walk-forward initialization** — eliminate cold-start
        2. **Combinatorial bandits** — select k=10 assets simultaneously
        3. **Broader universe** — international ETFs, bonds, commodities
        4. **Transaction cost modeling** — realistic performance
        5. **20+ year validation** — multiple full market cycles
        6. **Multi-seed evaluation** — confidence intervals
        """)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.divider()
st.caption("**Andrés García** · Computer Scientist · Universidad Nacional de Colombia · [GitHub](https://github.com/agarcia1607/GAT-LINUCB)")