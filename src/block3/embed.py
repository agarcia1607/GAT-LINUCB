from __future__ import annotations

import os
import csv
from typing import Dict, List

import numpy as np
import torch

from .dataset import SnapshotDataset
from .model import GATEncoder
from .checks import assert_snapshot_contract, embedding_stats, sensitivity_tests


def ensure_dirs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "npy"), exist_ok=True)


@torch.no_grad()
def run_block3(
    snapshots_root: str = "artifacts/snapshots/npz",
    out_dir: str = "artifacts/embeddings_gat",
    device: str = "cpu",
    out_dim: int = 16,
    limit: int | None = None,
):
    ensure_dirs(out_dir)

    ds = SnapshotDataset(snapshots_root)

    model = GATEncoder(out_dim=out_dim)
    model.to(device)
    model.eval()

    diag_path = os.path.join(out_dir, "embeddings_diag.csv")
    index_path = os.path.join(out_dir, "embeddings_index.csv")

    # index
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "path_npy", "N", "d"])
        dates = ds.get_dates() if limit is None else ds.get_dates()[:limit]
        for date in dates:
            w.writerow([date, f"npy/{date}.npy", 48, out_dim])

    diag_rows: List[Dict] = []
    n = len(ds) if limit is None else min(limit, len(ds))

    for i in range(n):
        data = ds[i]
        date = data.date

        assert_snapshot_contract(data, num_nodes=48)

        data = data.to(device)
        z = model(data.x, data.edge_index, data.edge_attr)

        st = embedding_stats(z)
        sens = sensitivity_tests(model, data)

        np.save(os.path.join(out_dir, "npy", f"{date}.npy"), z.cpu().numpy().astype(np.float32))

        diag_rows.append({"date": date, **st, **sens})

    cols = ["date", "z_mean_abs", "z_std", "z_max_abs", "z_nan_count",
            "delta_edgeattr0", "delta_negedgeattr"]
    with open(diag_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in diag_rows:
            w.writerow(r)

    print(f"[OK] embeddings saved: {out_dir}/npy/*.npy ({n} snapshots)")
    print(f"[OK] diag saved: {diag_path}")
    print(f"[OK] index saved: {index_path}")


if __name__ == "__main__":
    run_block3(limit=3)  # start small
