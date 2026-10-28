#!/usr/bin/env python3
"""Aggregate and visualize the full network without converting all 4,050 nodes to a graph object.

Writes country-level and broad-industry flow summaries, a sector-to-sector heatmap,
and a legible subnetwork of the strongest-flow country-industry nodes.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

SECTOR_GROUPS = [
    "Agriculture/forestry/fishing", "Mining/quarrying", "Food/textiles/wood/paper",
    "Chemicals/petroleum/minerals", "Metals/machinery/electronics/transport",
    "Utilities", "Construction", "Trade", "Transport/accommodation",
    "Information/communications", "Finance/real estate", "Professional/admin services",
    "Public/social/other services",
]


def industry_group(code: str) -> str:
    c = code.strip().upper()
    if c in {"A01", "A02", "A03"}:
        return SECTOR_GROUPS[0]
    if c.startswith("B"):
        return SECTOR_GROUPS[1]
    if c in {"C10T12", "C13T15", "C16", "C17_18"}:
        return SECTOR_GROUPS[2]
    if c in {"C19", "C20", "C21", "C22", "C23"}:
        return SECTOR_GROUPS[3]
    if c.startswith("C"):
        return SECTOR_GROUPS[4]
    if c.startswith(("D", "E")):
        return SECTOR_GROUPS[5]
    if c.startswith("F"):
        return SECTOR_GROUPS[6]
    if c.startswith("G"):
        return SECTOR_GROUPS[7]
    if c.startswith("H") or c == "I":
        return SECTOR_GROUPS[8]
    if c.startswith("J"):
        return SECTOR_GROUPS[9]
    if c in {"K", "L"}:
        return SECTOR_GROUPS[10]
    if c in {"M", "N"}:
        return SECTOR_GROUPS[11]
    if c in {"O", "P", "Q", "R", "S", "T", "U"}:
        return SECTOR_GROUPS[12]
    raise ValueError(f"Industry code {code!r} has no broad-sector mapping.")


def read_nodes(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [r["node"] for r in csv.DictReader(handle)]


def write_csv(path: Path, headers: list[str], rows: list[list]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-dir", type=Path, default=Path("results_full_network"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--top-nodes", type=int, default=25,
                    help="Number of high total-strength nodes in the network plot")
    ap.add_argument("--top-edges", type=int, default=120,
                    help="Maximum flow edges shown in the subnetwork plot")
    args = ap.parse_args()
    root = args.matrix_dir.resolve()
    out = (args.out or root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    W = sparse.load_npz(root / "adjacency_weighted_full.npz").tocsr()
    nodes = read_nodes(root / "node_index.csv")
    n = len(nodes)
    if W.shape != (n, n):
        raise ValueError("Adjacency shape disagrees with node_index.csv.")

    countries = [label.split("_", 1)[0] for label in nodes]
    industries = [label.split("_", 1)[1] for label in nodes]
    groups = [industry_group(c) for c in industries]
    country_names = list(dict.fromkeys(countries))
    group_names = SECTOR_GROUPS
    country_idx = {v: i for i, v in enumerate(country_names)}
    group_idx = {v: i for i, v in enumerate(group_names)}

    # Sparse membership matrices avoid dense, hand-built graph objects.
    country_membership = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), [country_idx[v] for v in countries])),
        shape=(n, len(country_names)),
    )
    group_membership = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), [group_idx[v] for v in groups])),
        shape=(n, len(group_names)),
    )
    country_flows = (country_membership.T @ W @ country_membership).toarray()
    group_flows = (group_membership.T @ W @ group_membership).toarray()

    write_csv(out / "country_flow_summary_full.csv",
              ["country_code", "out_strength_million_current_usd", "in_strength_million_current_usd"],
              [[country_names[i], float(country_flows[i, :].sum()), float(country_flows[:, i].sum())]
               for i in range(len(country_names))])
    write_csv(out / "broad_sector_flow_summary_full.csv",
              ["broad_sector", "out_strength_million_current_usd", "in_strength_million_current_usd"],
              [[group_names[i], float(group_flows[i, :].sum()), float(group_flows[:, i].sum())]
               for i in range(len(group_names))])
    write_csv(out / "broad_sector_flow_matrix_full.csv",
              ["supplier_sector"] + group_names,
              [[group_names[i]] + [float(v) for v in group_flows[i, :]]
               for i in range(len(group_names))])

    # Sector-to-sector heatmap: rows supply, columns purchase.
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(np.log10(group_flows / 1000.0 + 1e-3), aspect="auto")
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels([x.replace("/", "/\n") for x in group_names], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names, fontsize=8)
    ax.set_xlabel("Purchasing sector")
    ax.set_ylabel("Supplying sector")
    ax.set_title("Country-aggregated intermediate flows by broad sector (log scale)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log10(flow in billion current USD + 0.001)")
    fig.tight_layout()
    fig.savefig(out / "broad_sector_flow_heatmap_full.png", dpi=190)
    plt.close(fig)

    out_strength = np.asarray(W.sum(axis=1)).ravel()
    in_strength = np.asarray(W.sum(axis=0)).ravel()
    total_strength = out_strength + in_strength
    top_nodes = np.argsort(total_strength)[::-1][:max(2, min(args.top_nodes, n))]
    selected = set(int(i) for i in top_nodes)
    edges = []
    coo = W.tocoo()
    for i, j, value in zip(coo.row, coo.col, coo.data):
        if int(i) in selected and int(j) in selected and i != j:
            edges.append((int(i), int(j), float(value)))
    edges.sort(key=lambda row: row[2], reverse=True)
    edges = edges[:max(1, args.top_edges)]

    # Circular layout is deterministic and avoids force-layout memory costs.
    m = len(top_nodes)
    coords = {}
    for rank, node_id in enumerate(top_nodes):
        angle = 2 * math.pi * rank / m - math.pi / 2
        coords[int(node_id)] = (math.cos(angle), math.sin(angle))

    fig, ax = plt.subplots(figsize=(13, 11))
    max_edge_weight = max((edge[2] for edge in edges), default=1.0)
    for i, j, value in edges:
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        width = 0.25 + 1.8 * math.log1p(value) / max(math.log1p(max_edge_weight), 1e-12)
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=7, linewidth=width, alpha=0.28,
                                color="slategray", connectionstyle="arc3,rad=0.08",
                                shrinkA=9, shrinkB=9)
        ax.add_patch(arrow)

    positive = total_strength[top_nodes]
    sizes = 100 + 900 * np.sqrt(positive / max(float(positive.max()), 1e-12))
    palette = plt.get_cmap("tab20")
    country_order = {c: i for i, c in enumerate(dict.fromkeys(countries[int(i)] for i in top_nodes))}
    for node_id, size in zip(top_nodes, sizes):
        x, y = coords[int(node_id)]
        node_country = countries[int(node_id)]
        ax.scatter([x], [y], s=size, color=palette(country_order[node_country] % 20),
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.text(x * 1.09, y * 1.09, nodes[int(node_id)], fontsize=7,
                ha="center", va="center", zorder=4)
    ax.set_title(f"Strongest {m} country-industry nodes and up to {len(edges)} largest internal flows")
    ax.text(0.02, 0.02,
            "Arrow: supplier → buyer. Node size: total in- plus out-strength. "
            "Only edges among selected nodes are shown.",
            transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "top_strength_subnetwork_full.png", dpi=200)
    plt.close(fig)

    print(f"Wrote country/sector flow summaries and figures to {out}")


if __name__ == "__main__":
    main()
