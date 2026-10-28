"""Full-network structural metrics and reduced-network centrality."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr


def weighted_pagerank(A: sparse.csr_matrix, damping: float = 0.85, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Weighted directed PageRank, where A[i,j] is flow i -> j."""
    A = A.tocsr(copy=True).astype(np.float64)
    A.data = np.maximum(A.data, 0.0)
    A.eliminate_zeros()
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    out_strength = np.asarray(A.sum(axis=1)).ravel()
    inv_out = np.divide(1.0, out_strength, out=np.zeros_like(out_strength), where=out_strength > 0)
    transition = (sparse.diags(inv_out) @ A).tocsr()
    dangling = out_strength <= 0
    rank = np.full(n, 1.0 / n)
    teleport = np.full(n, (1.0 - damping) / n)
    for _ in range(max_iter):
        dangling_mass = rank[dangling].sum()
        updated = teleport + damping * (transition.T @ rank + dangling_mass / n)
        total = float(updated.sum())
        if total > 0:
            updated /= total
        if np.abs(updated - rank).sum() < tol:
            rank = updated
            break
        rank = updated
    return rank


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _gini(values: np.ndarray) -> float:
    x = np.maximum(np.asarray(values, dtype=float), 0.0)
    if x.size == 0 or float(x.sum()) == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2.0 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def build_full_metrics(nodes: list[str], Z: sparse.csr_matrix, output: np.ndarray, results_dir: Path) -> dict:
    """Analyse all 4,050 detailed nodes and write per-node and threshold results."""
    Z = Z.tocsr(copy=True)
    Z.data = np.maximum(Z.data, 0.0)
    Z.eliminate_zeros()
    total_flow = float(Z.sum())
    diag = Z.diagonal()
    self_flow = float(diag.sum())
    Z_cross = Z.copy()
    Z_cross.setdiag(0.0)
    Z_cross.eliminate_zeros()
    n = len(nodes)
    out_strength = np.asarray(Z_cross.sum(axis=1)).ravel()
    in_strength = np.asarray(Z_cross.sum(axis=0)).ravel()
    out_degree = np.asarray((Z_cross > 0).sum(axis=1)).ravel().astype(int)
    in_degree = np.asarray((Z_cross > 0).sum(axis=0)).ravel().astype(int)
    pagerank = weighted_pagerank(Z_cross)
    off_diagonal_edge_count = int(Z_cross.nnz)
    possible_edges = n * max(n - 1, 0)

    full_metrics = []
    for i, label in enumerate(nodes):
        country, activity = label.split("_", 1)
        full_metrics.append({
            "node": label,
            "country_code": country,
            "activity_code": activity,
            "out_degree": int(out_degree[i]),
            "in_degree": int(in_degree[i]),
            "out_strength_million_USD": float(out_strength[i]),
            "in_strength_million_USD": float(in_strength[i]),
            "weighted_pagerank": float(pagerank[i]),
            "gross_output_million_USD": float(output[i]),
        })
    write_csv(results_dir / "structural_node_metrics_full.csv", full_metrics)

    thresholds = [0.0, 1.0, 10.0, 100.0]  # flow units are million USD
    threshold_rows = []
    for cutoff in thresholds:
        mask = Z_cross.copy()
        mask.data = (mask.data >= cutoff).astype(np.int8)
        mask.eliminate_zeros()
        edge_count = int(mask.nnz)
        weak_count, weak_labels = connected_components(mask, directed=True, connection="weak")
        strong_count, strong_labels = connected_components(mask, directed=True, connection="strong")
        weak_sizes = np.bincount(weak_labels)
        strong_sizes = np.bincount(strong_labels)
        threshold_rows.append({
            "flow_cutoff_million_USD": cutoff,
            "flow_cutoff_USD": cutoff * 1_000_000,
            "positive_edges": edge_count,
            "possible_off_diagonal_edges": int(possible_edges),
            "directed_density": float(edge_count / possible_edges) if possible_edges else 0.0,
            "weak_component_count": int(weak_count),
            "largest_weak_component_nodes": int(weak_sizes.max(initial=0)),
            "strong_component_count": int(strong_count),
            "largest_strong_component_nodes": int(strong_sizes.max(initial=0)),
        })
    write_csv(results_dir / "edge_threshold_sensitivity.csv", threshold_rows)

    top_out = sorted(full_metrics, key=lambda row: row["out_strength_million_USD"], reverse=True)[:20]
    top_in = sorted(full_metrics, key=lambda row: row["in_strength_million_USD"], reverse=True)[:20]
    top_pr = sorted(full_metrics, key=lambda row: row["weighted_pagerank"], reverse=True)[:20]
    top_gross = sorted(full_metrics, key=lambda row: row["gross_output_million_USD"], reverse=True)[:20]
    write_csv(results_dir / "top20_out_strength_full.csv", top_out)
    write_csv(results_dir / "top20_in_strength_full.csv", top_in)
    write_csv(results_dir / "top20_weighted_pagerank_full.csv", top_pr)

    active = np.asarray([out_strength[i] + in_strength[i] > 0 for i in range(n)])
    rho1 = spearmanr(out_strength, in_strength).statistic
    rho2 = spearmanr(out_strength, pagerank).statistic
    rho3 = spearmanr(out_strength, np.maximum(output, 0.0)).statistic
    summary = {
        "node_count": int(n),
        "raw_positive_intermediate_flow_total_million_USD": total_flow,
        "self_flow_total_million_USD": self_flow,
        "self_flow_share_of_positive_intermediate_flow": self_flow / total_flow if total_flow else 0.0,
        "off_diagonal_flow_total_million_USD": float(Z_cross.sum()),
        "positive_off_diagonal_edge_count": off_diagonal_edge_count,
        "possible_off_diagonal_edge_count": int(possible_edges),
        "positive_off_diagonal_density": float(off_diagonal_edge_count / possible_edges) if possible_edges else 0.0,
        "out_strength_gini": _gini(out_strength),
        "in_strength_gini": _gini(in_strength),
        "active_node_count": int(active.sum()),
        "spearman_out_vs_in_strength": float(rho1) if np.isfinite(rho1) else None,
        "spearman_out_strength_vs_weighted_pagerank": float(rho2) if np.isfinite(rho2) else None,
        "spearman_out_strength_vs_gross_output": float(rho3) if np.isfinite(rho3) else None,
        "top20_out_strength": top_out,
        "top20_in_strength": top_in,
        "top20_weighted_pagerank": top_pr,
        "top20_gross_output": top_gross,
        "edge_threshold_sensitivity": threshold_rows,
        "interpretation": "Structural measures are derived from recorded annual intermediate flows; they are not direct measures of failure probability or realised economic loss.",
    }
    (results_dir / "structural_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_aggregate_metrics(aggregate_nodes: list[str], Z_aggregate: sparse.csr_matrix, X_aggregate: np.ndarray, results_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute metrics used for source selection in the 12-sector dynamical model."""
    Z = Z_aggregate.tocsr(copy=True)
    Z.data = np.maximum(Z.data, 0.0)
    Z.setdiag(0.0)
    Z.eliminate_zeros()
    out_strength = np.asarray(Z.sum(axis=1)).ravel()
    in_strength = np.asarray(Z.sum(axis=0)).ravel()
    degree_out = np.asarray((Z > 0).sum(axis=1)).ravel().astype(int)
    degree_in = np.asarray((Z > 0).sum(axis=0)).ravel().astype(int)
    pagerank = weighted_pagerank(Z)
    rows = []
    for idx, label in enumerate(aggregate_nodes):
        country, sector = label.split("_", 1)
        rows.append({
            "node": label,
            "country_code": country,
            "broad_sector": sector,
            "out_degree": int(degree_out[idx]),
            "in_degree": int(degree_in[idx]),
            "out_strength_million_USD": float(out_strength[idx]),
            "in_strength_million_USD": float(in_strength[idx]),
            "weighted_pagerank": float(pagerank[idx]),
            "gross_output_million_USD": float(max(X_aggregate[idx], 0.0)),
        })
    write_csv(results_dir / "structural_node_metrics_country_sector.csv", rows)

    # Weighted betweenness is approximate and deliberately uses a $10m edge cutoff
    # so tiny accounting flows do not make the graph appear fully connected.
    threshold_million_usd = 10.0
    eligible = Z.copy()
    eligible.data = (eligible.data >= threshold_million_usd).astype(np.int8)
    eligible.eliminate_zeros()
    try:
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(range(len(aggregate_nodes)))
        coo = Z.tocoo()
        for i, j, flow in zip(coo.row, coo.col, coo.data):
            if flow >= threshold_million_usd and i != j:
                # The larger the observed flow, the shorter the path cost.
                G.add_edge(int(i), int(j), distance=float(1.0 / max(np.log1p(float(flow)), 1e-9)))
        k = min(64, max(1, len(aggregate_nodes) - 1))
        betweenness_raw = nx.betweenness_centrality(G, k=k, normalized=True, weight="distance", seed=2022)
        betweenness = np.asarray([betweenness_raw.get(i, 0.0) for i in range(len(aggregate_nodes))])
        betweenness_method = f"NetworkX approximate weighted directed betweenness, k={k}, edge cutoff US$10m"
    except Exception as exc:
        # Keep the analysis runnable even if optional graph functionality is unavailable.
        betweenness = np.zeros(len(aggregate_nodes), dtype=float)
        betweenness_method = f"Unavailable; fallback zeros ({type(exc).__name__}: {exc})"

    for idx, row in enumerate(rows):
        row["weighted_betweenness_approx"] = float(betweenness[idx])
    write_csv(results_dir / "structural_node_metrics_country_sector.csv", rows)
    summary = {
        "aggregate_node_count": len(aggregate_nodes),
        "country_count": len({node.split("_", 1)[0] for node in aggregate_nodes}),
        "broad_sector_count": len({node.split("_", 1)[1] for node in aggregate_nodes}),
        "flow_conservation_error_relative": None,
        "output_conservation_error_relative": None,
        "betweenness_method": betweenness_method,
        "betweenness_edge_cutoff_million_USD": threshold_million_usd,
    }
    return summary, pagerank, betweenness
