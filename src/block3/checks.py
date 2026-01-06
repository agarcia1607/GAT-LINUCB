from __future__ import annotations

from typing import Dict
import torch


def assert_snapshot_contract(data, num_nodes: int = 48) -> None:
    x = data.x
    ei = data.edge_index
    ea = data.edge_attr

    assert x.shape == (num_nodes, 2), f"Bad X shape: {x.shape}"
    assert ei.dim() == 2 and ei.shape[0] == 2, f"Bad edge_index shape: {ei.shape}"
    assert ea.dim() == 2 and ea.shape[1] == 1, f"Bad edge_attr shape: {ea.shape}"
    assert ei.shape[1] == ea.shape[0], f"E mismatch: {ei.shape[1]} vs {ea.shape[0]}"
    assert int(ei.max()) < num_nodes and int(ei.min()) >= 0, "edge_index out of bounds"
    assert torch.isfinite(x).all(), "Non-finite in X"
    assert torch.isfinite(ea).all(), "Non-finite in edge_attr"


@torch.no_grad()
def embedding_stats(z: torch.Tensor) -> Dict[str, float]:
    abs_z = z.abs()
    return {
        "z_mean_abs": float(abs_z.mean().item()),
        "z_std": float(z.std().item()),
        "z_max_abs": float(abs_z.max().item()),
        "z_nan_count": float((~torch.isfinite(z)).sum().item()),
    }


@torch.no_grad()
def frob_relative(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.norm(a - b, p="fro")
    den = torch.norm(a, p="fro").clamp_min(eps)
    return float((num / den).item())


@torch.no_grad()
def sensitivity_tests(model, data) -> Dict[str, float]:
    model.eval()
    x, ei, ea = data.x, data.edge_index, data.edge_attr

    z = model(x, ei, ea)
    z0 = model(x, ei, torch.zeros_like(ea))
    zneg = model(x, ei, -ea)

    return {
        "delta_edgeattr0": frob_relative(z, z0),
        "delta_negedgeattr": frob_relative(z, zneg),
    }
