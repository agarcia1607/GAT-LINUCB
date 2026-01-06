from __future__ import annotations

def symmetrize_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Simetriza un conjunto de aristas dirigidas.
    Si i->j o j->i existe, incluye ambas.
    """
    edge_set = set(edges)
    sym_edges = set()

    for i, j in edge_set:
        if i == j:
            continue
        sym_edges.add((i, j))
        sym_edges.add((j, i))

    return sorted(sym_edges)
