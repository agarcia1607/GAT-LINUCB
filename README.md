# GAT-LINUCB: Graph Attention Networks + Contextual Bandits for Asset Selection

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

A reproducible ML pipeline combining **Graph Attention Networks (GAT)** and **LinUCB contextual bandits** for sequential asset selection in financial markets.

**Core hypothesis:** graph-aware embeddings from market correlation structure produce better bandit context than raw price features — validated empirically via ablation study.

---

## Key Results (10-Year Backtest, Quality-Filtered Universe)

### System vs S&P 500 — 2016–2026

| | LinUCB + GAT (filtered) | S&P 500 |
|---|---|---|
| **Ann. Return** | **22.5%** | 13.1% |
| Sharpe | 0.677 | 0.843 |
| **Sortino** | **1.032** | 0.958 |
| Max Drawdown | -43.0% | -18.1% |
| **2022 bear market** | **+7.4%** | -14.5% |

The system generates **1.7x the benchmark return** over 10 years with better downside-adjusted performance (Sortino). The cost is higher max drawdown — a consequence of single-asset concentration.

### Convergence Phase Analysis

Performance improves consistently as LinUCB learns market structure:

| Phase | Ann. Return | Sharpe | MaxDD | vs S&P 500 |
|---|---|---|---|---|
| Early — exploration (t=0–177) | 5.2% | 0.324 | -27.1% | Loses (11.2%) |
| Mid — convergence (t=178–354) | 29.6% | 0.741 | -33.6% | **Wins (8.4%)** |
| **Late — exploitation (t=355–532)** | **34.6%** | **0.962** | -18.5% | **Wins (20.2%)** |

Once converged, the system generates **34.6% annualized vs 20.2% benchmark** with Sharpe 0.962.

### Recent Windows (Fully Converged)

| Window | Ann. Return | Sharpe | Sortino | Max Drawdown |
|---|---|---|---|---|
| Last 52w | 69.7% | 1.397 | 2.053 | -16.2% |
| Last 12w | 38.5% | 0.835 | 1.140 | -10.0% |

### Annual Performance

| Year | LinUCB | S&P 500 | Winner |
|---|---|---|---|
| 2016 | 69.6% | 16.5% | ✅ LinUCB |
| 2017 | -16.3% | 21.9% | S&P 500 |
| 2018 | -13.3% | -6.2% | S&P 500 |
| 2019 | 16.0% | 26.1% | S&P 500 |
| 2020 | 6.6% | 16.2% | S&P 500 |
| 2021 | 117.8% | 22.0% | ✅ LinUCB |
| **2022** | **+7.4%** | **-14.5%** | **✅ LinUCB** |
| 2023 | 21.6% | 20.0% | ✅ LinUCB |
| 2024 | 19.9% | 23.8% | S&P 500 |
| 2025 | 44.7% | 17.0% | ✅ LinUCB |

**5 of 10 years beat the benchmark.** The years that lose (2017–2020) correspond to the exploration and early convergence phases — the cold-start problem. Walk-forward initialization would eliminate this.

### COVID-19 Recovery — Head to Head

| | LinUCB | S&P 500 |
|---|---|---|
| Max drawdown | -47.6% | -28.6% |
| Weeks to bottom | 3 | 5 |
| **Weeks to full recovery** | **15** | **27** |

The system fell more but recovered **2x faster** than the market.

---

## Quality Filter

### The Problem

Without filtering, the system achieves 60% annualized — but 219pp of that comes from **PG&E (PCG) during its 2019 bankruptcy** and **Norwegian Cruise Line (NCLH) during COVID recovery**. These are not genuine market signals — they are extreme volatility from corporate distress events.

LinUCB correctly optimized its objective (rolling Sharpe), but rolling Sharpe appears high during distressed-asset rebounds because large positive returns dominate the 12-week window. The algorithm cannot distinguish between:
- Genuine momentum (asset growing due to business performance)
- Distress rebound (asset recovering from near-insolvency)

### The Solution

A pre-filtering step excludes assets with extreme historical volatility patterns:

```python
# Exclude asset if:
# 1. max_weekly_return > 50%  (single-week crisis event)
# 2. OR (extreme_weeks > 5 AND max_drawdown < -75%)  (sustained distress)

filtered = (max_wk > 0.50) or (extreme_wks > 5 and dd < -0.75)
```

### Filtered Assets (13 of 466)

`PCG` `SMCI` `NCLH` `WBD` `APA` `BA` `UAL` `OXY` `RCL` `SPG` `VTR` `WELL` `DHR`

These include: PG&E (bankruptcy 2019), Super Micro Computer (accounting fraud 2024), Norwegian Cruise Line (COVID near-insolvency), Boeing (737 MAX crisis), airlines (COVID).

### Impact

| | Without filter | With filter |
|---|---|---|
| Ann. Return | 60.0% | **22.5%** |
| Sharpe | 1.072 | 0.677 |
| 2019 return | 235.3% | 16.0% |
| 2022 return | -26.2% | **+7.4%** |
| Signal quality | Inflated by distress | **Genuine** |

The filter reduces headline return but **reveals the genuine signal** — and critically, the system now beats the market in the 2022 bear market.

---

## Ablation Study — Does GAT Actually Help?

LinUCB with three context types, same hyperparameters (alpha=2.0, Sharpe reward, 10 years):

| Context | Ann. Return | Sharpe | MaxDD |
|---|---|---|---|
| **LinUCB + GAT embeddings** | **60.0%** | **1.072** | -60.8% |
| LinUCB + Raw features (d=2) | 23.4% | 0.655 | -73.0% |
| LinUCB + Random embeddings (d=16) | 2.7% | 0.256 | -35.3% |

**GAT embeddings are the difference** — 60% vs 2.7% with identical algorithm and hyperparameters. Graph-aware market representations capture correlation structure that raw price features cannot represent.

---

## How the System Works

```
Yahoo Finance prices (2016–2026, 466 S&P 500 assets)
        │
        ▼
Quality Filter — remove 13 distressed assets
        │
        ▼
Weekly returns + rolling correlation matrices
        │
        ▼
Graph snapshots — strictly causal (F_{t-1} only)
        │
        ▼
Graph Attention Network (GAT) → 16-dim embeddings
        │
        ▼
LinUCB contextual bandit
→ selects 1 asset per week
→ reward: rolling Sharpe (window=12)
→ online update via Sherman-Morrison
```

### Input Variables

**Node features (2 per asset):**
- **Momentum** — average return over last 4 weeks
- **Volatility** — standard deviation over last 4 weeks

**Graph edges:**
- **Rolling Pearson correlation** over last 24 weeks
- kNN connectivity (k=8) on correlation strength
- Updated every week — dynamic market structure

### Causal Integrity

No look-ahead bias. Snapshot at date `t` uses **only returns up to `t-1`**:

```python
# src/lib/filtration.py
assert window_corr.index.max() == hist_until  # hist_until = t-1
assert window_mom.index.max() == hist_until
```

Full causal chain: `F_{t-1}` → `embedding(t)` → `action(t)` → `reward(t+1)`

---

## Known Limitations

- **Single asset per week** — no diversification, 100% concentration
- **No transaction costs** — weekly rotation incurs real commissions
- **Cold-start: ~55 weeks** of exploration before convergence with K=453 assets
- **Survivorship bias** — uses current S&P 500 universe back-projected (companies that failed are excluded)
- **Single seed** — no confidence intervals
- **Bull market dominated** — 2016–2026 mostly favorable; bear market of 2022 is the main stress test

---

## Experiment Setup

- **Period:** January 2016 – March 2026 (532 weeks)
- **Universe:** 453 S&P 500 assets (466 minus 13 quality-filtered)
- **Reward:** rolling Sharpe ratio (window=12 weeks)
- **Alpha:** 2.0 (grid search over [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0])
- **Graph:** kNN (k=8) on rolling Pearson correlation (W=24 weeks)
- **Embeddings:** GAT, d=16

---

## Quick Start

```bash
git clone https://github.com/agarcia1607/GAT-LINUCB
cd GAT-LINUCB
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python run_pipeline.py                                    # build graphs + embeddings
python -m src.11_linucb_filtered --reward_mode sharpe    # run with quality filter
python -m src.10_linucb_contextual --reward_mode sharpe  # run without filter
python reports/financial_metrics.py                       # evaluate vs S&P 500
python reports/recovery_analysis.py                      # crisis recovery analysis
streamlit run dashboard/app.py                           # launch dashboard
```

---

## Project Structure

```
GAT-LINUCB/
├── src/
│   ├── lib/
│   │   ├── filtration.py          # Causal data filtration
│   │   ├── quality_filter.py      # Asset quality filter
│   │   └── ...
│   ├── block3/                    # GAT encoder
│   ├── 10_linucb_contextual.py    # LinUCB (no filter)
│   └── 11_linucb_filtered.py      # LinUCB (with quality filter)
├── reports/
│   ├── financial_metrics.py       # Full evaluation vs S&P 500
│   ├── recovery_analysis.py       # Crisis recovery periods
│   └── figures/
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── run_pipeline.py
├── run_bandits.py
└── Dockerfile
```

---

## Research Roadmap

**Completed:**
- ✅ Sharpe reward (14.1% → 38.1% vs raw return)
- ✅ Alpha grid search (optimal: 2.0)
- ✅ Ablation study (GAT vs Raw vs Random — validated)
- ✅ Convergence phase analysis
- ✅ Warm start experiment (cold start outperforms — documented)
- ✅ Sortino reward (Sharpe remains optimal)
- ✅ Quality filter (removes distressed assets, reveals genuine signal)
- ✅ 10-year backtest with crisis recovery analysis
- ✅ AWS S3 deployment + Streamlit dashboard

**Next:**
- Walk-forward initialization — train on 2016–2022, deploy from 2023 already converged
- Combinatorial bandits — select k=10 assets simultaneously (real portfolio)
- Reduce K — faster convergence, less cold-start cost
- Multi-seed evaluation — confidence intervals
- Validation on 20+ years with historical S&P 500 universe

---

## Stack

`Python` · `PyTorch` · `PyTorch Geometric` · `LinUCB` · `AWS S3` · `Streamlit` · `Docker` · `Yahoo Finance`

---

## Author

**Andrés García** · Computer Scientist · Universidad Nacional de Colombia  
[GitHub](https://github.com/agarcia1607) · [LinkedIn](https://www.linkedin.com/in/andrés-felipe-garcía-orrego-17965b218)

## License

MIT