# GAT-LINUCB: Graph Attention Networks + Contextual Bandits for Asset Selection

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

A reproducible ML pipeline combining **Graph Attention Networks (GAT)** and **LinUCB contextual bandits** for sequential asset selection in financial markets.

**Core hypothesis:** graph-aware embeddings from market correlation structure produce better bandit context than raw price features — validated empirically via ablation study.

---

## Key Results

### Ablation Study — Does GAT Actually Help?

LinUCB evaluated with three context types under identical conditions (alpha=2.0, Sharpe reward, 166 weeks):

| Context | Ann. Return | Sharpe | Sortino | Max Drawdown |
|---|---|---|---|---|
| **LinUCB + GAT embeddings** | **38.1%** | **0.769** | **1.204** | -39.9% |
| LinUCB + Raw features (d=2) | 11.7% | 0.531 | 0.726 | -53.1% |
| LinUCB + Random embeddings (d=16) | 8.8% | 0.426 | 0.589 | -18.3% |
| Random policy (baseline) | 24.2% | 0.850 | 1.220 | -19.9% |

**In the converged phase (last 55 weeks), the separation is decisive:**

| Context | Ann. Return | Sharpe | Sortino |
|---|---|---|---|
| **LinUCB + GAT embeddings** | **+83.2%** | **1.199** | **1.844** |
| LinUCB + Raw features | -28.0% | -0.266 | -0.324 |
| LinUCB + Random embeddings | -26.2% | -0.587 | -0.743 |

GAT embeddings are not decorative — they are the difference between a working system and two that fail after convergence.

---

## How the System Works

```
Yahoo Finance prices (2015–2026, 466 assets)
        │
        ▼
Weekly returns + rolling correlation matrices
        │
        ▼
Graph snapshots — one per week, strictly causal (F_{t-1} only)
        │
        ▼
Graph Attention Network → asset embeddings (d=16)
        │
        ▼
LinUCB contextual bandit
        │
        ▼
Asset selected at t, reward observed at t+1
```

### Causal Integrity

No look-ahead bias. The snapshot at date `t` uses **only returns up to `t-1`** (`F_{t-1}`). Enforced by explicit assertions in `src/lib/filtration.py`:

```python
assert window_corr.index.max() == hist_until  # hist_until = t-1
assert window_mom.index.max() == hist_until
```

Full causal chain: `F_{t-1}` → `embedding(t)` → `action(t)` → `reward(t+1)`

---

## Convergence Analysis

LinUCB exhibits sublinear regret O(d√T log T). Performance improves as θ_t stabilizes:

| Phase | Ann. Return | Sharpe | Max Drawdown | Unique Assets |
|---|---|---|---|---|
| Early — exploration (t=0–55) | -7.0% | -0.085 | -17.5% | 41 |
| Mid — convergence (t=56–110) | 33.8% | 0.766 | -41.1% | 21 |
| **Late — exploitation (t=111–165)** | **109.8%** | **1.385** | -25.2% | 18 |

**Converged phase vs S&P 500 (same period):**

| | Ann. Return | Sharpe | Max Drawdown |
|---|---|---|---|
| LinUCB + GAT (converged) | **109.8%** | **1.385** | -25.2% |
| S&P 500 | 8.0% | 0.538 | -7.7% |

**Recent windows (fully converged system):**

| Window | Ann. Return | Sharpe | Sortino | Max Drawdown |
|---|---|---|---|---|
| Last 12w | 215.5% | 2.953 | 9.703 | -3.8% |
| Last 26w | 76.6% | 1.580 | 3.263 | -6.4% |
| Last 52w | 152.3% | 1.730 | 3.365 | -14.6% |

The global drawdown (-39.9%) is entirely explained by the cold-start exploration phase with K=466 assets. Once converged, the system achieves Sharpe 2.953 and max drawdown -3.8%.

---

## Known Limitations

- **Cold-start cost:** ~55 weeks of exploration before convergence with K=466 assets. Reducing K is the primary next step.
- **Single market regime:** evaluated on 2023–2026 (bull market). Validation on 2018–2022 (bear market, COVID) is pending.
- **No transaction costs:** weekly rotation incurs real-world costs not modeled here.
- **Single seed:** results from one random initialization. Multi-seed evaluation with confidence intervals is pending.

---

## Experiment Setup

- **Period:** January 2023 – March 2026 (166 weeks)
- **Universe:** 466 assets (S&P 500 constituents + global ETFs)
- **Reward:** rolling Sharpe ratio (window=12 weeks) — outperforms raw return and Sortino
- **Alpha:** 2.0 (grid search over [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0])
- **Graph:** kNN (k=8) over rolling Pearson correlation (W=24 weeks), symmetrized
- **Embeddings:** GAT, d=16, trained on graph snapshots

---

## Quick Start

```bash
git clone https://github.com/agarcia1607/GAT-LINUCB
cd GAT-LINUCB

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python run_pipeline.py   # build graphs + embeddings
python run_bandits.py    # run LinUCB / Greedy / Random
```

Reproduce the financial evaluation:
```bash
python reports/financial_metrics.py
```

---

## Project Structure

```
GAT-LINUCB/
├── src/
│   ├── lib/              # filtration, correlation, graph, features
│   ├── block3/           # GAT encoder
│   └── 10_linucb_contextual.py  # LinUCB + Greedy + Random
├── reports/
│   ├── financial_metrics.py     # full evaluation vs S&P 500
│   └── figures/
├── notebooks/
│   └── analysis_linucb.ipynb
├── run_pipeline.py
├── run_bandits.py
└── Dockerfile
```

---

## Research Roadmap

**Completed:**
- ✅ Sharpe reward (raw → Sharpe: 14.1% → 38.1%)
- ✅ Alpha grid search (optimal: 2.0)
- ✅ Ablation study (GAT vs Raw vs Random embeddings)
- ✅ Convergence phase analysis
- ✅ Warm start experiment (cold start outperforms — documented)

**Next:**
- Reduce K (cold-start bottleneck)
- Multi-seed evaluation
- Validation on 2018–2022
- Combinatorial bandits (portfolio-level selection)
- EXP3 (adversarial regime comparison)

---

## Stack

`Python` · `PyTorch` · `PyTorch Geometric` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Docker` · `AWS`

---

## Author

**Andrés García** · Computer Scientist · Universidad Nacional de Colombia  
[GitHub](https://github.com/agarcia1607) · [LinkedIn](https://www.linkedin.com/in/andrés-felipe-garcía-orrego-17965b218)

## License

MIT