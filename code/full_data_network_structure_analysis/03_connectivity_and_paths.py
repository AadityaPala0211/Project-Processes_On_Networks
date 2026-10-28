#!/usr/bin/env python3
"""Connectivity and sampled shortest-path analysis for the full sparse network.

All paths use unweighted link count. To avoid allocating a 4,050 x 4,050
all-pairs distance matrix, shortest paths are calculated from a reproducible
sample of source nodes. Component counts are exact.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_nodes(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [r["node"] for r in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-dir", type=Path, default=Path("results_full_network"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sample-sources", type=int, default=64,
                    help="Number of source nodes for shortest-path sampling (default: 64)")
    ap.add_argument("--seed", type=int, default=2022)
    args = ap.parse_args()
    root = args.matrix_dir.resolve()
    out = (args.out or root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    A = sparse.load_npz(root / "adjacency_binary_full.npz").tocsr()
    nodes = read_nodes(root / "node_index.csv")
    n = len(nodes)
    if A.shape != (n, n):
        raise ValueError("Binary adjacency dimensions disagree with node index.")

    weak_count, weak_labels = connected_components(A, directed=True, connection="weak")
    strong_count, strong_labels = connected_components(A, directed=True, connection="strong")
    weak_sizes = np.bincount(weak_labels)
    strong_sizes = np.bincount(strong_labels)

    weak_component_rows = [
        {"component_id": i, "size": int(size), "type": "weakly_connected"}
        for i, size in enumerate(sorted(weak_sizes, reverse=True), start=1)
    ]
    weak_component_rows += [
        {"component_id": i, "size": int(size), "type": "strongly_connected"}
        for i, size in enumerate(sorted(strong_sizes, reverse=True), start=1)
    ]
    write_csv(out / "component_size_distributions.csv", weak_component_rows)

    rng = np.random.default_rng(args.seed)
    number_sources = max(1, min(args.sample_sources, n))
    source_ids = np.sort(rng.choice(n, size=number_sources, replace=False))
    print(f"Computing exact component statistics and shortest paths from {number_sources} sampled sources...", flush=True)
    distances = shortest_path(A, directed=True, unweighted=True, indices=source_ids, method="auto")
    if distances.ndim == 1:
        distances = distances[np.newaxis, :]

    source_rows = []
    all_finite_distances = []
    pair_rows = []
    for row_number, source in enumerate(source_ids):
        d = distances[row_number]
        reachable_mask = np.isfinite(d) & (np.arange(n) != source)
        reachable = d[reachable_mask]
        all_finite_distances.extend(reachable.astype(int).tolist())
        source_rows.append({
            "source_index": int(source),
            "source_node": nodes[int(source)],
            "reachable_other_nodes": int(reachable.size),
            "reachable_fraction": float(reachable.size / max(n - 1, 1)),
            "mean_shortest_path_to_reachable_nodes": float(reachable.mean()) if reachable.size else None,
            "maximum_shortest_path_to_reachable_nodes": int(reachable.max()) if reachable.size else None,
            "unreachable_other_nodes": int((n - 1) - reachable.size),
        })
        for target in np.where(np.isfinite(d))[0]:
            if target != source:
                pair_rows.append({
                    "source_node": nodes[int(source)],
                    "target_node": nodes[int(target)],
                    "shortest_path_length": int(d[target]),
                })
    write_csv(out / "sampled_source_reachability.csv", source_rows)
    write_csv(out / "sampled_shortest_path_pairs.csv", pair_rows)

    finite = np.asarray(all_finite_distances, dtype=float)
    summary = {
        "node_count": n,
        "directed_edge_count": int(A.nnz),
        "weakly_connected_component_count": int(weak_count),
        "largest_weak_component_size": int(weak_sizes.max()),
        "largest_weak_component_fraction": float(weak_sizes.max() / n),
        "strongly_connected_component_count": int(strong_count),
        "largest_strong_component_size": int(strong_sizes.max()),
        "largest_strong_component_fraction": float(strong_sizes.max() / n),
        "isolated_nodes": int(np.sum(np.asarray(A.sum(axis=0)).ravel() + np.asarray(A.sum(axis=1)).ravel() == 0)),
        "shortest_path_method": "exact directed unweighted distances from a reproducible random source sample",
        "sampled_source_count": number_sources,
        "sample_seed": args.seed,
        "sampled_pair_count_reachable": int(finite.size),
        "sampled_mean_shortest_path_reachable_pairs": float(finite.mean()) if finite.size else None,
        "sampled_median_shortest_path_reachable_pairs": float(np.median(finite)) if finite.size else None,
        "sampled_90th_percentile_shortest_path": float(np.quantile(finite, 0.90)) if finite.size else None,
        "sampled_maximum_shortest_path": int(finite.max()) if finite.size else None,
        "note": "Mean/path percentiles are estimates from sampled sources, not exact all-pairs statistics.",
    }
    (out / "connectivity_path_summary_full.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(weak_sizes, bins=min(50, max(5, len(weak_sizes))))
    axes[0].set_title("Weak component-size distribution")
    axes[0].set_xlabel("Nodes in component")
    axes[0].set_ylabel("Component count")
    axes[1].hist(strong_sizes, bins=min(50, max(5, len(strong_sizes))))
    axes[1].set_title("Strong component-size distribution")
    axes[1].set_xlabel("Nodes in component")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "component_size_distributions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if finite.size:
        bins = np.arange(int(finite.min()) - 0.5, int(finite.max()) + 1.5, 1)
        ax.hist(finite, bins=bins)
    ax.set_title(f"Shortest-path distribution from {number_sources} sampled sources")
    ax.set_xlabel("Directed shortest-path length (edges)")
    ax.set_ylabel("Reachable ordered pairs sampled")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "sampled_shortest_path_distribution.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved connectivity outputs under {out}")


if __name__ == "__main__":
    main()
