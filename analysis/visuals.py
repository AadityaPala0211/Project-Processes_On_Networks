"""Figures for structural results of the 2022 OECD ICIO network."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import sparse


def create_structural_figures(nodes, Z_full, groups, structural_summary, results_dir: Path, image_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_dir.mkdir(parents=True, exist_ok=True)
    out_top = structural_summary["top20_out_strength"]
    in_top = structural_summary["top20_in_strength"]
    pr_top = structural_summary["top20_weighted_pagerank"]
    for records, key, filename, title, xlabel, divisor in [
        (out_top, "out_strength_million_USD", "top20_out_strength.png", "Top 20 country-industry suppliers", "Outgoing intermediate flows (billion current USD)", 1000.0),
        (in_top, "in_strength_million_USD", "top20_in_strength.png", "Top 20 country-industry input buyers", "Incoming intermediate inputs (billion current USD)", 1000.0),
        (pr_top, "weighted_pagerank", "top20_weighted_pagerank.png", "Top 20 weighted PageRank nodes", "Weighted PageRank", 1.0),
    ]:
        ordered = list(reversed(records))
        fig, ax = plt.subplots(figsize=(9.2, 6.2))
        ax.barh([r["node"] for r in ordered], [r[key] / divisor for r in ordered])
        ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=0.25)
        fig.tight_layout(); fig.savefig(image_dir / filename, dpi=180); plt.close(fig)

    Zcross = Z_full.tocsr(copy=True)
    Zcross.setdiag(0.0); Zcross.eliminate_zeros()
    out_degree = np.asarray((Zcross > 0).sum(axis=1)).ravel()
    in_degree = np.asarray((Zcross > 0).sum(axis=0)).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].hist(out_degree, bins=40); axes[0].set_title("Out-degree distribution")
    axes[0].set_xlabel("Number of distinct purchasing nodes"); axes[0].set_ylabel("Node count")
    axes[1].hist(in_degree, bins=40); axes[1].set_title("In-degree distribution")
    axes[1].set_xlabel("Number of distinct suppliers")
    fig.tight_layout(); fig.savefig(image_dir / "degree_distributions_full.png", dpi=180); plt.close(fig)

    from analysis.data import SECTORS
    sector_index = {s: i for i, s in enumerate(SECTORS)}
    G = sparse.csr_matrix((np.ones(len(nodes)), (np.arange(len(nodes)), [sector_index[g] for g in groups])),
                          shape=(len(nodes), len(SECTORS)))
    sector_flow = (G.T @ Z_full @ G).toarray()
    # Values are millions of USD; divide by 1,000,000 for trillions of USD.
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    values_trillion = sector_flow / 1_000_000.0
    im = ax.imshow(np.log10(values_trillion + 1e-4), aspect="auto", cmap="viridis")
    labels = [x.replace("_", " ") for x in SECTORS]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Purchasing sector"); ax.set_ylabel("Supplying sector")
    ax.set_title("Global sector-to-sector intermediate flows (log scale)")
    fig.colorbar(im, ax=ax, label="log10(flow, trillion USD + 0.0001)")
    fig.tight_layout(); fig.savefig(image_dir / "sector_flow_heatmap_full.png", dpi=180); plt.close(fig)

    thresholds = structural_summary["edge_threshold_sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot([r["flow_cutoff_USD"] for r in thresholds], [100*r["directed_density"] for r in thresholds], marker="o")
    axes[0].set_xscale("symlog", linthresh=1e6); axes[0].set_xlabel("Minimum flow edge cutoff (USD)")
    axes[0].set_ylabel("Directed density (%)"); axes[0].set_title("Edge-cutoff sensitivity"); axes[0].grid(alpha=.25)
    axes[1].plot([r["flow_cutoff_USD"] for r in thresholds], [r["largest_weak_component_nodes"] for r in thresholds], marker="o", label="Largest weak component")
    axes[1].plot([r["flow_cutoff_USD"] for r in thresholds], [r["largest_strong_component_nodes"] for r in thresholds], marker="s", label="Largest strong component")
    axes[1].set_xscale("symlog", linthresh=1e6); axes[1].set_xlabel("Minimum flow edge cutoff (USD)")
    axes[1].set_ylabel("Nodes in largest component"); axes[1].set_title("Connectivity sensitivity")
    axes[1].grid(alpha=.25); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(image_dir / "edge_threshold_sensitivity_full.png", dpi=180); plt.close(fig)

    sector_rows = []
    for i, sector in enumerate(SECTORS):
        sector_rows.append({
            "sector": sector,
            "out_strength_million_USD": float(sector_flow[i, :].sum()),
            "in_strength_million_USD": float(sector_flow[:, i].sum()),
        })
    import csv
    path = results_dir / "sector_flow_summary_full.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sector_rows[0].keys()))
        writer.writeheader(); writer.writerows(sector_rows)
    return sector_flow
