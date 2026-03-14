# Initial Experimental Results

This document records early experimental results for the **GAT-LINUCB** pipeline. The goal is to evaluate whether graph-based embeddings improve contextual bandit performance for asset selection.

---

# Experimental Setup

## Data

* Source: Yahoo Finance
* Frequency: Weekly
* Asset universe: ~466 assets
* Time horizon: ~166 weeks

Returns are computed from adjusted closing prices and aligned across the asset universe.

---

# Graph Construction

Each week a graph snapshot is constructed using **rolling correlations of returns**.

Steps:

1. Compute rolling window correlations
2. Build adjacency matrix
3. Apply k-NN filtering
4. Normalize edge weights

The resulting sequence of graphs represents the **dynamic market structure over time**.

---

# Node Features

For each asset and week:

Features include:

* momentum
* volatility

These features form the base input for the graph neural network.

---

# Embeddings

Graph Attention Networks (GAT) are used to produce **low-dimensional embeddings** for each asset in each snapshot.

Example configuration:

* embedding dimension: 16
* attention heads: configurable
* training snapshots: sequential weekly graphs

The embeddings capture **structural relationships between assets** beyond raw features.

---

# Bandit Policies Evaluated

The contextual bandit stage compares several policies:

* LinUCB
* Greedy
* Random

Context vectors are derived from **GAT embeddings**.

Rewards are computed from subsequent asset returns.

---

# Evaluation Metrics

The following metrics are monitored:

* cumulative reward
* average reward
* regret relative to oracle

These metrics help determine whether **graph-based context improves decision quality**.

---

# Preliminary Observations

Initial experiments suggest:

* graph embeddings capture sector-like clustering
* asset relationships evolve over time
* contextual bandits can exploit these representations

Further experimentation is required to quantify performance improvements.

---

# Next Steps

Planned improvements include:

* EXP3 and adversarial bandits
* combinatorial bandit formulations
* portfolio-level rewards
* transaction cost modeling
* risk-adjusted metrics (Sharpe, drawdown)

---

# Reproducibility

To reproduce the pipeline:

Run the data + graph + embedding pipeline:

```
python run_pipeline.py
```

Then run bandit experiments:

```
python run_bandits.py
```

All generated outputs are written to the `artifacts/` directory.

---

# Results Table (Example)

| Policy | Avg Reward | Cumulative Reward | Notes                               |
| ------ | ---------- | ----------------- | ----------------------------------- |
| LinUCB | TBD        | TBD               | Uses GAT embeddings as context      |
| Greedy | TBD        | TBD               | Exploits immediate estimated reward |
| Random | baseline   | baseline          | Control policy                      |

These values will be filled as experiments are repeated and stabilized.

---

# Pipeline Diagram

```
Price Data
   │
   ▼
Weekly Returns
   │
   ▼
Rolling Correlations
   │
   ▼
Graph Snapshots
   │
   ▼
GAT Encoder
   │
   ▼
Asset Embeddings
   │
   ▼
Contextual Bandits
   │
   ▼
Sequential Asset Selection
```

---

# Research Questions

This project explores several research questions:

1. Do **graph neural network embeddings** capture useful structure in financial markets?
2. Does using **graph-aware context** improve contextual bandit performance?
3. Can dynamic correlation graphs reveal **latent sector structures**?
4. How stable are graph embeddings across time?
5. Do contextual bandits benefit from **market structure information**?

These questions guide further experiments and extensions of the project.
