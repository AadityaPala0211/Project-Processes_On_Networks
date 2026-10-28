#!/usr/bin/env python3
"""Estimate local clustering on the undirected projection of the full network.

The production network is directed. This script defines an undirected adjacency
where i--j exists if either i->j or j->i is present, then calculates the local
clustering coefficient on that projection. Sampling is the default to avoid
costly neighborhood subgraphs for a dense 4,050-node network; pass --all-nodes
for coefficients on every node.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_nodes(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["node"] for row in csv.DictReader(handle)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-dir", type=Path, default=Path("results_full_network"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sample-nodes", type=int, default=100,
                    help="Number of nodes to sample unless --all-nodes is used")
    ap.add_argument("--all-nodes", action="store_true",
                    help="Calculate exact local clustering coefficients for all nodes")
    ap.add_argument("--seed", type=int, default=2022)
    args = ap.parse_args()
    root = args.matrix_dir.resolve()
    out = (args.out or root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    directed = sparse.load_npz(root / "adjacency_binary_full.npz").tocsr()
    nodes = read_nodes(root / "node_index.csv")
    n = len(nodes)
    if directed.shape != (n, n):
        raise ValueError("Adjacency dimensions do not agree with node index.")

    undirected = directed.maximum(directed.T).tocsr().astype(np.uint8)
    undirected.setdiag(0)
    undirected.eliminate_zeros()
    degree = np.asarray(undirected.sum(axis=1)).ravel().astype(np.int64)

    rng = np.random.default_rng(args.seed)
    if args.all_nodes:
        sampled = np.arange(n, dtype=np.int64)
    else:
        count = max(1, min(args.sample_nodes, n))
        sampled = np.sort(rng.choice(n, size=count, replace=False))

    values = []
    rows = []
    print(f"Calculating clustering for {len(sampled)} of {n} nodes...", flush=True)
    for position, i in enumerate(sampled, start=1):
        k = int(degree[i])
        if k < 2:
            coefficient = 0.0
        else:
            neighbors = undirected.indices[undirected.indptr[i]:undirected.indptr[i + 1]]
            induced = undirected[neighbors, :][:, neighbors]
            coefficient = float(induced.nnz / (k * (k - 1)))
        values.append(coefficient)
        rows.append({
            "matrix_index": int(i),
            "node": nodes[int(i)],
            "undirected_projection_degree": k,
            "local_clustering_coefficient": coefficient,
            "exact_or_sampled": "exact_all_nodes" if args.all_nodes else "sampled_node",
        })
        if position % 25 == 0 or position == len(sampled):
            print(f"Processed {position}/{len(sampled)} nodes", flush=True)

    with (out / "clustering_coefficients_full.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    values_array = np.asarray(values, dtype=float)
    summary = {
        "network_node_count": n,
        "directed_edge_count_in_input": int(directed.nnz),
        "undirected_projection_edge_count": int(undirected.nnz // 2),
        "calculated_node_count": int(len(sampled)),
        "all_nodes_calculated": bool(args.all_nodes),
        "seed": args.seed,
        "definition": "local clustering coefficient on undirected projection A_u=max(A,A.T)",
        "mean_local_clustering": float(values_array.mean()) if values_array.size else None,
        "median_local_clustering": float(np.median(values_array)) if values_array.size else None,
        "standard_deviation_local_clustering": float(values_array.std(ddof=1)) if values_array.size > 1 else 0.0,
        "minimum_local_clustering": float(values_array.min()) if values_array.size else None,
        "maximum_local_clustering": float(values_array.max()) if values_array.size else None,
        "method_note": (
            "Summary statistics describe the sampled nodes unless all_nodes_calculated is true. "
            "Directed edges are symmetrized only for this clustering measure; all other analyses "
            "retain the supplier-to-buyer direction."
        ),
    }
    (out / "clustering_summary_full.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values_array, bins=30, range=(0, 1))
    ax.set_xlabel("Local clustering coefficient")
    ax.set_ylabel("Node count in sample" if not args.all_nodes else "Node count")
    ax.set_title("Clustering on the undirected projection")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "clustering_distribution_full.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
