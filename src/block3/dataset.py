from __future__ import annotations

import glob
import os
import json
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class SnapshotDataset(Dataset):
    """
    Loads weekly snapshots from artifacts/snapshots/npz/YYYY-MM-DD.npz

    Keys per npz:
      - X: (48,2) float32
      - edge_index: (2,E) int64
      - edge_attr: (E,1) float32  (z-score clipped)
      - edge_attr_raw: (E,1) float32 (signed corr)
      - stats_json: scalar string JSON
    """

    def __init__(self, root: str = "artifacts/snapshots/npz"):
        self.root = root
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz found under: {root}")
        self.dates = [os.path.splitext(os.path.basename(f))[0] for f in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        path = self.files[idx]
        date = self.dates[idx]
        z = np.load(path, allow_pickle=False)

        X = z["X"].astype(np.float32)
        edge_index = z["edge_index"].astype(np.int64)
        edge_attr = z["edge_attr"].astype(np.float32)
        edge_attr_raw = z["edge_attr_raw"].astype(np.float32)

        stats_json = str(z["stats_json"].item())
        try:
            stats: Dict = json.loads(stats_json)
        except Exception:
            stats = {"raw": stats_json}

        data = Data(
            x=torch.from_numpy(X),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
        )
        data.edge_attr_raw = torch.from_numpy(edge_attr_raw)
        data.date = date
        data.stats = stats
        return data

    def get_dates(self) -> List[str]:
        return list(self.dates)
