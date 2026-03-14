# GAT-LINUCB: Graph Attention Networks + Contextual Bandits for Asset Selection

This project builds a reproducible machine learning pipeline that combines **graph neural networks** and **contextual bandits** to model financial markets and perform asset selection.

The idea is to represent the market as a **dynamic graph of assets**, learn **graph-aware embeddings using Graph Attention Networks (GAT)**, and then use those embeddings as context for **bandit algorithms such as LinUCB**.

---

# Overview

The pipeline performs the following steps:

1. Download historical asset prices
2. Build weekly returns
3. Construct rolling correlation graphs
4. Generate graph snapshots
5. Train GAT embeddings for assets
6. Run contextual bandit policies

These steps produce representations of assets that capture **market structure** rather than treating each asset independently.

---

# Project Structure

```
GAT-LINUCB
│
├── src/
│   ├── block3/
│   │   └── embed.py
│   ├── 02_prepare_weekly_adjclose.py
│   └── ...
│
├── run_pipeline.py        # end-to-end pipeline
├── run_bandits.py         # contextual bandit experiments
├── config.py              # configuration variables
├── requirements.txt
├── .gitignore
│
├── artifacts/             # generated outputs (ignored)
├── data/                  # intermediate data (ignored)
├── logs/                  # logs (ignored)
```

The repository only tracks **code and configuration**. Generated data and artifacts are excluded to keep the repository lightweight and reproducible.

---

# Pipeline

```
Yahoo Finance prices
        │
        ▼
Weekly adjusted prices
        │
        ▼
Weekly returns
        │
        ▼
Rolling correlation matrices
        │
        ▼
Graph snapshots
        │
        ▼
Graph Attention Network (GAT)
        │
        ▼
Asset embeddings
        │
        ▼
Contextual bandits (LinUCB / Greedy / Random)
```

The embeddings encode **structural relationships between assets** derived from the market network.

---

# Main Components

## Graph Construction

Assets are connected based on **rolling correlations of returns**. Each week produces a graph snapshot representing the current market structure.

## Node Features

Typical features include:

* momentum
* volatility

These features are combined with graph structure through GAT.

## GAT Embeddings

Graph Attention Networks produce embeddings that capture:

* asset co-movement
* sector-like clustering
* dynamic relationships

## Contextual Bandits

Embeddings are used as context for decision-making algorithms such as:

* LinUCB
* Greedy
* Random

The bandit selects assets over time and receives rewards based on returns.

---

# Running the Project

Run the full pipeline:

```bash
python run_pipeline.py
```

Run bandit experiments:

```bash
python run_bandits.py
```

---

# Notes

The following directories are intentionally excluded from version control:

* artifacts/
* data/
* logs/

They contain generated outputs that can be recreated by running the pipeline.

---

# Future Work

Possible extensions include:

* EXP3 bandits
* combinatorial bandits
* portfolio-level reward functions
* risk-adjusted evaluation metrics
* dynamic graph models

---

# Research Context

This project sits at the intersection of:

* Graph Neural Networks
* Financial network modeling
* Online learning and contextual bandits

It explores whether **graph-aware representations improve sequential asset selection** compared to traditional feature-based approaches.
