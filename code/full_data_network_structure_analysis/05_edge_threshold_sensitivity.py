#!/usr/bin/env python3
"""Flow-threshold sensitivity for structural metrics on the full ICIO network.

ICIO monetary tables can contain many tiny positive flows. Binary metrics and
density therefore depend strongly on what counts as an edge. This script
recomputes topology at multiple minimum-flow thresholds. Threshold units are
million current USD (e.g. threshold=1 means US$1 million).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-dir", type=Path, default=Path("results_full_network"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.0, 0.01, 0.1, 1.0, 10.0, 100.0],
                    help="Minimum retained edge flow in million current USD")
    args = ap.parse_args()
    root = args.matrix_dir.resolve()
    out = (args.out or root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    W = sparse.load_npz(root / "adjacency_weighted_full.npz").tocsr()
    n = W.shape[0]
    possible = n * (n - 1)
    base_total = float(W.data[W.data > 0].sum())
    rows = []
    for threshold in sorted(set(args.thresholds)):
        if threshold < 0:
            raise ValueError("Thresholds cannot be negative.")
        if threshold == 0:
            mask = W.copy()
            mask.data = (mask.data > 0).astype(np.uint8)
            mask.eliminate_zeros()
        else:
            mask = W.copy()
            mask.data = (mask.data >= threshold).astype(np.uint8)
            mask.eliminate_zeros()
        edge_count = int(mask.nnz)
        out_degree = np.asarray(mask.sum(axis=1)).ravel()
        in_degree = np.asarray(mask.sum(axis=0)).ravel()
        component_count, labels = connected_components(mask, directed=True, connection="weak")
        component_sizes = np.bincount(labels)
        retained_flow = float(W.data[W.data >= threshold].sum()) if threshold > 0 else base_total
        rows.append({
            "minimum_flow_million_current_usd": threshold,
            "directed_edge_count": edge_count,
            "possible_directed_edges": possible,
            "directed_density": edge_count / possible if possible else 0.0,
            "mean_out_degree": float(out_degree.mean()),
            "mean_in_degree": float(in_degree.mean()),
            "isolated_nodes": int(np.sum((out_degree == 0) & (in_degree == 0))),
            "weak_component_count": int(component_count),
            "largest_weak_component_nodes": int(component_sizes.max()),
            "largest_weak_component_fraction": float(component_sizes.max() / n),
            "retained_positive_flow_million_current_usd": retained_flow,
            "retained_flow_fraction": retained_flow / base_total if base_total else 0.0,
        })
    write_csv(out / "edge_threshold_sensitivity_full.csv", rows)
    (out / "edge_threshold_sensitivity_full.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    thresholds = np.asarray([r["minimum_flow_million_current_usd"] for r in rows])
    density = 100 * np.asarray([r["directed_density"] for r in rows])
    retained = 100 * np.asarray([r["retained_flow_fraction"] for r in rows])
    x = thresholds.copy()
    x[x == 0] = min([v for v in thresholds if v > 0], default=1.0) / 10.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, density, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Minimum flow (million current USD; log scale)")
    axes[0].set_ylabel("Directed binary density (%)")
    axes[0].set_title("Topology depends on flow cutoff")
    axes[1].plot(x, retained, marker="o")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Minimum flow (million current USD; log scale)")
    axes[1].set_ylabel("Positive flow retained (%)")
    axes[1].set_title("Economic flow retained after thresholding")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "edge_threshold_sensitivity_full.png", dpi=190)
    plt.close(fig)
    print(f"Wrote {len(rows)} threshold scenarios to {out / 'edge_threshold_sensitivity_full.csv'}")


if __name__ == "__main__":
    main()
