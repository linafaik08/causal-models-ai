"""Adjacency matrix decoding for causal-learn graphs."""

from __future__ import annotations

import numpy as np

from .models import EdgeInput


def decode_adj_matrix(adj: np.ndarray, node_names: list[str]) -> list[EdgeInput]:
    """
    Convert a causal-learn adjacency matrix to a list of EdgeInput objects.

    Iterates the upper triangle to avoid double-counting symmetric edges.
    Encoding (adj[i,j], adj[j,i]):
        (-1,  1) → i --> j
        ( 1, -1) → j --> i  (stored as j --> i with node_from=j)
        (-1, -1) → i --- j
        ( 1,  1) → i <-> j
        ( 2, -1) → i o-> j
        (-1,  2) → j o-> i  (stored with node_from=j)
        ( 2,  2) → i o-o j
    """
    edges = []
    n = len(node_names)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = int(adj[i, j]), int(adj[j, i])
            if a == 0 and b == 0:
                continue

            if a == -1 and b == 1:
                edges.append(EdgeInput(node_from=node_names[i], node_to=node_names[j],
                                       mark="-->", idx_from=i, idx_to=j))
            elif a == 1 and b == -1:
                edges.append(EdgeInput(node_from=node_names[j], node_to=node_names[i],
                                       mark="-->", idx_from=j, idx_to=i))
            elif a == -1 and b == -1:
                edges.append(EdgeInput(node_from=node_names[i], node_to=node_names[j],
                                       mark="---", idx_from=i, idx_to=j))
            elif a == 1 and b == 1:
                edges.append(EdgeInput(node_from=node_names[i], node_to=node_names[j],
                                       mark="<->", idx_from=i, idx_to=j))
            elif a == 2 and b == -1:
                edges.append(EdgeInput(node_from=node_names[i], node_to=node_names[j],
                                       mark="o->", idx_from=i, idx_to=j))
            elif a == -1 and b == 2:
                edges.append(EdgeInput(node_from=node_names[j], node_to=node_names[i],
                                       mark="o->", idx_from=j, idx_to=i))
            elif a == 2 and b == 2:
                edges.append(EdgeInput(node_from=node_names[i], node_to=node_names[j],
                                       mark="o-o", idx_from=i, idx_to=j))
    return edges
