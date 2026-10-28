#!/usr/bin/env python3
"""Compute full-network node metrics from the saved sparse adjacency matrices.

Run after 01_build_adjacency_matrix.py. Structural edges are directed:
supplier i -> purchasing industry j. Weighted strengths are sums of monetary
intermediate input flows (million current USD), excluding self-flows by default.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(np.maximum(x, 0.0))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    n = x.size
    return float((2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def weighted_pagerank(weights: sparse.csr_matrix, alpha: float = 0.85,
                      max_iter: int = 500, tol: float = 1e-11) -> tuple[np.ndarray, int]:
    n = weights.shape[0]
    out_strength = np.asarray(weights.sum(axis=1)).ravel()
    inverse = np.zeros(n)
    non_dangling = out_strength > 0
    inverse[non_dangling] = 1.0 / out_strength[non_dangling]
    transition = sparse.diags(inverse) @ weights
    dangling = ~non_dangling
    rank = np.full(n, 1.0 / n)
    for iteration in range(1, max_iter + 1):
        updated = (1.0 - alpha) / n + alpha * (
            np.asarray(transition.T @ rank).ravel() + rank[dangling].sum() / n
        )
        updated /= updated.sum()
        if np.linalg.norm(updated - rank, ord=1) < tol:
            return updated, iteration
        rank = updated
    return rank, max_iter


def load_nodes(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["node"] for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=Path("results_full_network"),
                        help="Folder containing adjacency_weighted_full.npz and node_index.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder (defaults to --matrix-dir)")
    parser.add_argument("--topn", type=int, default=30)
    args = parser.parse_args()
    matrix_dir = args.matrix_dir.resolve()
    out = (args.out or matrix_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    weighted = sparse.load_npz(matrix_dir / "adjacency_weighted_full.npz").tocsr()
    binary = sparse.load_npz(matrix_dir / "adjacency_binary_full.npz").tocsr()
    nodes = load_nodes(matrix_dir / "node_index.csv")
    if weighted.shape[0] != len(nodes) or weighted.shape != binary.shape:
        raise ValueError("Matrix dimensions do not match node_index.csv.")
    n = len(nodes)

    out_degree = np.asarray(binary.sum(axis=1)).ravel().astype(np.int64)
    in_degree = np.asarray(binary.sum(axis=0)).ravel().astype(np.int64)
    out_strength = np.asarray(weighted.sum(axis=1)).ravel()
    in_strength = np.asarray(weighted.sum(axis=0)).ravel()
    density = float(binary.nnz / (n * (n - 1))) if n > 1 else 0.0
    rank, pagerank_iterations = weighted_pagerank(weighted)

    gross_output_path = matrix_dir / "gross_output_million_usd.csv"
    gross_output = np.loadtxt(gross_output_path, delimiter=",", skiprows=1, dtype=float)
    if gross_output.shape[0] != n:
        raise ValueError("Gross output vector length does not match node count.")

    rows = []
    for i, node in enumerate(nodes):
        country, industry = node.split("_", 1)
        rows.append({
            "matrix_index": i, "node": node, "country_code": country,
            "industry_code": industry, "out_degree": int(out_degree[i]),
            "in_degree": int(in_degree[i]),
            "out_strength_million_current_usd": float(out_strength[i]),
            "in_strength_million_current_usd": float(in_strength[i]),
            "weighted_pagerank": float(rank[i]),
            "gross_output_million_current_usd": float(gross_output[i]),
        })
    write_rows(out / "node_metrics_full.csv", rows)

    summary = {
        "node_count": n,
        "directed_edge_count_excluding_self_loops": int(binary.nnz),
        "possible_directed_edges_excluding_self_loops": int(n * (n - 1)),
        "directed_density": density,
        "out_degree": {
            "mean": float(out_degree.mean()), "median": float(np.median(out_degree)),
            "p90": float(np.quantile(out_degree, 0.90)), "max": int(out_degree.max()),
        },
        "in_degree": {
            "mean": float(in_degree.mean()), "median": float(np.median(in_degree)),
            "p90": float(np.quantile(in_degree, 0.90)), "max": int(in_degree.max()),
        },
        "out_strength_gini": gini(out_strength),
        "in_strength_gini": gini(in_strength),
        "total_off_diagonal_flow_million_current_usd": float(out_strength.sum()),
        "total_gross_output_million_current_usd": float(np.maximum(gross_output, 0).sum()),
        "weighted_pagerank_iterations": pagerank_iterations,
        "interpretation": (
            "Out-strength is value of intermediate flows supplied to other nodes; "
            "in-strength is value of intermediate inputs purchased. Neither is GDP, "
            "exports, nor a measure of actual disruption."
        ),
    }
    correlations = {}
    for key, x, y in [
        ("out_degree_vs_out_strength", out_degree, out_strength),
        ("in_degree_vs_in_strength", in_degree, in_strength),
        ("out_strength_vs_weighted_pagerank", out_strength, rank),
        ("out_strength_vs_gross_output", out_strength, gross_output),
        ("in_strength_vs_gross_output", in_strength, gross_output),
    ]:
        stat = spearmanr(x, y)
        correlations[key] = {"spearman_rho": float(stat.statistic), "p_value": float(stat.pvalue)}
    summary["spearman_correlations"] = correlations
    (out / "structural_summary_full.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    metric_specs = [
        ("out_strength_million_current_usd", out_strength, "Top supplier nodes by out-strength", "Flows supplied (million current USD)"),
        ("in_strength_million_current_usd", in_strength, "Top buyer nodes by in-strength", "Inputs purchased (million current USD)"),
        ("out_degree", out_degree, "Top nodes by out-degree", "Number of buyer nodes supplied"),
        ("in_degree", in_degree, "Top nodes by in-degree", "Number of supplier nodes"),
        ("weighted_pagerank", rank, "Top nodes by weighted PageRank", "Weighted PageRank"),
    ]
    topn = max(1, min(args.topn, n))
    for metric, values, title, xlabel in metric_specs:
        top_ids = np.argsort(values)[::-1][:topn]
        top_rows = [rows[int(i)] for i in top_ids]
        write_rows(out / f"top_{metric}.csv", top_rows)
        ids = top_ids[::-1]
        fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(ids))))
        labels = [nodes[int(i)] for i in ids]
        ax.barh(labels, np.asarray(values)[ids])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(out_degree, bins=50)
    axes[0, 0].set_title("Out-degree distribution")
    axes[0, 1].hist(in_degree, bins=50)
    axes[0, 1].set_title("In-degree distribution")
    axes[1, 0].hist(np.log10(out_strength[out_strength > 0]), bins=50)
    axes[1, 0].set_title("log10(out-strength) for positive-flow nodes")
    axes[1, 1].hist(np.log10(in_strength[in_strength > 0]), bins=50)
    axes[1, 1].set_title("log10(in-strength) for positive-flow nodes")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "degree_and_strength_distributions.png", dpi=180)
    plt.close(fig)

    print(f"Analysed {n:,} nodes and {binary.nnz:,} directed edges.")
    print(f"Metrics: {out / 'node_metrics_full.csv'}")
    print(f"Summary: {out / 'structural_summary_full.json'}")


if __name__ == "__main__":
    main()
