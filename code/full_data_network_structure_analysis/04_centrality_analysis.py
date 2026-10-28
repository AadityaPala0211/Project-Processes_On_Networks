#!/usr/bin/env python3
"""Compute influence centralities on the full sparse supply-chain network.

Degree/strength/PageRank/eigenvector centrality are computed on all nodes.
Betweenness uses a reproducible source-sampled Brandes estimator because exact
betweenness over a 4,050-node, dense directed graph is unnecessarily expensive.
Shortest-path betweenness treats all positive-flow edges as length 1; it is not
flow-weighted economic influence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_nodes(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [r["node"] for r in csv.DictReader(handle)]


def pagerank(weights: sparse.csr_matrix, alpha=0.85, tol=1e-11, max_iter=500):
    n = weights.shape[0]
    out = np.asarray(weights.sum(axis=1)).ravel()
    inv = np.zeros(n)
    valid = out > 0
    inv[valid] = 1.0 / out[valid]
    trans = sparse.diags(inv) @ weights
    dangling = ~valid
    rank = np.full(n, 1.0 / n)
    for iteration in range(1, max_iter + 1):
        nxt = (1 - alpha) / n + alpha * (
            np.asarray(trans.T @ rank).ravel() + rank[dangling].sum() / n
        )
        nxt /= nxt.sum()
        if np.linalg.norm(nxt - rank, 1) < tol:
            return nxt, iteration
        rank = nxt
    return rank, max_iter


def eigenvector_centrality(weights: sparse.csr_matrix, tol=1e-10, max_iter=20000):
    """Perron eigenvector of W.T, returned as a nonnegative L1-normalized vector."""
    n = weights.shape[0]
    if weights.nnz == 0:
        return np.full(n, 1.0 / n), 0, "zero_matrix"
    # Normalize magnitude; eigenvectors are unchanged by a common positive scale.
    scaled = weights / max(float(weights.data.max()), 1.0)
    try:
        values, vectors = eigs(scaled.T.astype(float), k=1, which="LM",
                               maxiter=max_iter, tol=tol)
        vector = np.abs(np.real(vectors[:, 0]))
        if not np.isfinite(vector).all() or vector.sum() == 0:
            raise RuntimeError("Invalid eigenvector returned by eigs.")
        return vector / vector.sum(), 0, f"scipy_eigs_eigenvalue={values[0]}"
    except Exception as exc:
        # Power iteration on the nonnegative transpose; normalized every step.
        vector = np.full(n, 1.0 / np.sqrt(n))
        for iteration in range(1, max_iter + 1):
            nxt = np.asarray(scaled.T @ vector).ravel()
            norm = np.linalg.norm(nxt)
            if norm == 0 or not np.isfinite(norm):
                return np.full(n, 1.0 / n), iteration, f"fallback_failed:{exc}"
            nxt /= norm
            if np.linalg.norm(nxt - vector) < tol or np.linalg.norm(nxt + vector) < tol:
                vector = nxt
                break
            vector = nxt
        vector = np.abs(vector)
        vector /= vector.sum()
        return vector, iteration, f"power_iteration_fallback:{exc}"


def brandes_for_source(source: int, indptr: np.ndarray, indices: np.ndarray, n: int):
    """Unweighted directed Brandes dependency accumulation from one source."""
    stack = []
    predecessors = [[] for _ in range(n)]
    sigma = np.zeros(n, dtype=np.float64)
    distance = np.full(n, -1, dtype=np.int32)
    sigma[source] = 1.0
    distance[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        stack.append(v)
        start, end = indptr[v], indptr[v + 1]
        for w in indices[start:end]:
            if distance[w] < 0:
                distance[w] = distance[v] + 1
                queue.append(int(w))
            if distance[w] == distance[v] + 1:
                sigma[w] += sigma[v]
                predecessors[w].append(v)
    dependency = np.zeros(n, dtype=np.float64)
    centrality = np.zeros(n, dtype=np.float64)
    while stack:
        w = stack.pop()
        if sigma[w] > 0:
            coefficient = (1.0 + dependency[w]) / sigma[w]
            for v in predecessors[w]:
                dependency[v] += sigma[v] * coefficient
        if w != source:
            centrality[w] = dependency[w]
    return centrality


def sampled_betweenness(A: sparse.csr_matrix, sample_count: int, seed: int):
    n = A.shape[0]
    count = max(0, min(sample_count, n))
    result = np.zeros(n, dtype=np.float64)
    if count == 0 or n < 3:
        return result, 0
    rng = np.random.default_rng(seed)
    sources = np.sort(rng.choice(n, size=count, replace=False))
    for idx, source in enumerate(sources, start=1):
        result += brandes_for_source(int(source), A.indptr, A.indices, n)
        if idx % 5 == 0 or idx == count:
            print(f"Betweenness source samples: {idx}/{count}", flush=True)
    # Estimate total-source Brandes sum and use directed normalized centrality.
    result *= n / count
    result /= max((n - 1) * (n - 2), 1)
    return result, count


def write_rows(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-dir", type=Path, default=Path("results_full_network"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--betweenness-sources", type=int, default=12,
                    help="Number of sampled source nodes for betweenness; set 0 to skip")
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--topn", type=int, default=30)
    args = ap.parse_args()
    root = args.matrix_dir.resolve()
    out = (args.out or root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    A = sparse.load_npz(root / "adjacency_binary_full.npz").tocsr()
    W = sparse.load_npz(root / "adjacency_weighted_full.npz").tocsr()
    nodes = read_nodes(root / "node_index.csv")
    n = len(nodes)
    if A.shape != W.shape or A.shape != (n, n):
        raise ValueError("Adjacency dimensions do not agree with the node index.")

    print("Computing weighted PageRank...", flush=True)
    pr, pr_iterations = pagerank(W)
    print("Computing weighted eigenvector centrality...", flush=True)
    ev, ev_iterations, ev_method = eigenvector_centrality(W)
    print("Estimating normalized directed betweenness from sampled sources...", flush=True)
    betweenness, sampled_count = sampled_betweenness(A, args.betweenness_sources, args.seed)

    out_degree = np.asarray(A.sum(axis=1)).ravel()
    in_degree = np.asarray(A.sum(axis=0)).ravel()
    out_strength = np.asarray(W.sum(axis=1)).ravel()
    in_strength = np.asarray(W.sum(axis=0)).ravel()
    rows = []
    for i, node in enumerate(nodes):
        rows.append({
            "matrix_index": i, "node": node,
            "out_degree": int(out_degree[i]), "in_degree": int(in_degree[i]),
            "out_strength_million_current_usd": float(out_strength[i]),
            "in_strength_million_current_usd": float(in_strength[i]),
            "weighted_pagerank": float(pr[i]),
            "weighted_eigenvector_centrality": float(ev[i]),
            "sampled_directed_betweenness": float(betweenness[i]),
            "betweenness_source_samples": sampled_count,
        })
    write_rows(out / "centrality_metrics_full.csv", rows)
    summary = {
        "nodes": n,
        "weighted_pagerank_iterations": pr_iterations,
        "eigenvector_method": ev_method,
        "eigenvector_power_iterations": ev_iterations,
        "betweenness_method": "unweighted directed Brandes; sampled sources; scaled estimate",
        "betweenness_sample_count": sampled_count,
        "betweenness_seed": args.seed,
        "caveat": (
            "Betweenness is an approximation from sampled source nodes. It uses hop-count "
            "shortest paths, not monetary flow weights. The other weighted centralities use "
            "the monetary-flow adjacency matrix."
        ),
    }
    (out / "centrality_summary_full.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metric_specs = [
        ("weighted_pagerank", "Weighted PageRank"),
        ("weighted_eigenvector_centrality", "Weighted eigenvector centrality"),
        ("sampled_directed_betweenness", "Sampled directed betweenness"),
        ("out_degree", "Out-degree"),
        ("in_degree", "In-degree"),
    ]
    topn = max(1, min(args.topn, n))
    for key, title in metric_specs:
        ranked = sorted(rows, key=lambda r: float(r[key]), reverse=True)[:topn]
        write_rows(out / f"centrality_top_{key}.csv", ranked)
        vals = np.asarray([float(r[key]) for r in ranked[::-1]])
        labels = [r["node"] for r in ranked[::-1]]
        fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(labels))))
        ax.barh(labels, vals)
        ax.set_title(f"Top {len(labels)} nodes: {title}")
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"centrality_top_{key}.png", dpi=180)
        plt.close(fig)

    print(f"Wrote centrality_metrics_full.csv for {n:,} nodes.", flush=True)


if __name__ == "__main__":
    main()
