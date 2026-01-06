from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATEncoder(nn.Module):
    """
    Minimal GAT encoder that *uses edge_attr* via edge_dim=1.

    Input:  x (N,2), edge_index (2,E), edge_attr (E,1)
    Output: z (N,d_out)
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 32,
        out_dim: int = 16,
        heads1: int = 4,
        heads2: int = 1,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.dropout = float(dropout)

        self.conv1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads1,
            concat=False,         # output: (N, hidden_dim)
            dropout=dropout,
            edge_dim=1,           # <-- critical
            add_self_loops=False, # keep your graph unchanged
        )
        self.ln1 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=out_dim,
            heads=heads2,
            concat=False,         # output: (N, out_dim)
            dropout=dropout,
            edge_dim=1,
            add_self_loops=False,
        )
        self.ln2 = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.ln1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        z = self.conv2(h, edge_index, edge_attr=edge_attr)
        z = self.ln2(z)
        return z
