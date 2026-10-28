"""Single entry point for the full OECD ICIO 2022 analysis.

Run only: uv run master.py
This script locates data, creates output folders, runs every stage, and writes results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
import scipy

from analysis.data import SECTORS, aggregate_country_sector, load_icio
from analysis.dynamics import BASELINE, run_baseline_census
from analysis.structural import build_aggregate_metrics, build_full_metrics, write_csv
from analysis.visuals import create_structural_figures

ROOT = Path(__file__).resolve().parent
DATA_DEFAULT = ROOT / "data" / "2022_SML.csv"
RESULTS = ROOT / "results" / "linear_nonlinear"
IMAGES = ROOT / "images" / "linear_nonlinear"


def find_data() -> Path:
    if DATA_DEFAULT.is_file():
        return DATA_DEFAULT
    found = [p for p in ROOT.rglob("2022_SML.csv") if p.is_file()
             and not any(part in {".git", ".venv", "__pycache__"} for part in p.parts)]
    if found:
        return sorted(found, key=lambda p: (len(p.parts), str(p)))[0]
    raise FileNotFoundError(f"Expected OECD input at {DATA_DEFAULT}. Add data/2022_SML.csv and rerun uv run master.py.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_error(a: float, b: float) -> float:
    return abs(a-b) / max(abs(a), abs(b), 1e-12)


def sensitivity_summary(rows):
    summary = []
    for model in ("linear", "nonlinear"):
        model_rows = [r for r in rows if r["model"] == model]
        for factor in ("initial_disruption", "beta", "recovery_rate"):
            values = sorted({float(r[factor]) for r in model_rows})
            for value in values:
                sub = [r for r in model_rows if np.isclose(float(r[factor]), value)]
                summary.append({
                    "model": model, "factor": factor, "factor_value": value, "scenario_count": len(sub),
                    "mean_peak_disruption": float(np.mean([float(r["peak_output_weighted_disruption"]) for r in sub])),
                    "mean_cumulative_disruption_share_steps": float(np.mean([float(r["cumulative_output_weighted_disruption_share_steps"]) for r in sub])),
                    "mean_final_disruption": float(np.mean([float(r["final_output_weighted_disruption"]) for r in sub])),
                    "mean_max_affected_nodes_at_1pct": float(np.mean([float(r["max_affected_nodes_at_1pct"]) for r in sub])),
                })
    write_csv(RESULTS / "sensitivity_parameter_summary.csv", summary)
    return summary


def make_report(structural, aggregate, comparison, conservation, metadata):
    lin, non = comparison["linear_model"], comparison["nonlinear_model"]
    reps = comparison["representative_nodes"]
    lines = [
        "# OECD ICIO 2022: Structural and Linear-versus-Nonlinear Dynamics",
        "",
        "## Executive summary",
        f"- Full detailed network: **{structural['node_count']:,} nodes**, **{structural['positive_off_diagonal_edge_count']:,} positive off-diagonal edges**, density **{100*structural['positive_off_diagonal_density']:.2f}%**.",
        f"- Dynamical network: **{aggregate['aggregate_node_count']:,} country-sector nodes** across {aggregate['broad_sector_count']} sectors.",
        f"- Flow conservation relative error: {conservation['flow_relative_error']:.3e}; gross-output conservation relative error: {conservation['output_relative_error']:.3e}.",
        f"- Each model ran {lin['source_scenario_count']:,} source scenarios with the same baseline parameters.",
        f"- Mean peak output-weighted disruption: linear **{100*lin['mean_peak_disruption']:.3f}%**, nonlinear **{100*non['mean_peak_disruption']:.3f}%**.",
        f"- Mean cumulative disruption: linear **{lin['mean_cumulative_disruption_share_steps']:.4f} share-steps**, nonlinear **{non['mean_cumulative_disruption_share_steps']:.4f} share-steps**.",
        "",
        "The numerical dynamics are counterfactual outcomes, not forecasts. Annual ICIO has no observed, time-stamped disruption labels, so the model comparison is about simulated behaviour rather than predictive accuracy.",
        "",
        "## 1. Structural analysis",
        "",
        "| Analysis | What it measures | Why it is useful |",
        "|---|---|---|",
        "| Out-strength | Total intermediate sales supplied by a node | Finds large suppliers |",
        "| In-strength | Total intermediate inputs purchased | Identifies nodes with large input dependence in monetary terms |",
        "| In/out degree | Number of distinct supplier/buyer links | Summarizes topology; sensitive to tiny positive flows |",
        "| Weighted PageRank | Recursive influence on the directed weighted network | Candidate proxy for downstream influence |",
        "| Flow Gini | Concentration of supplier/buyer flow totals | Shows whether flows are evenly distributed |",
        "| Component and edge-cutoff analysis | Connectivity as small flows are excluded | Tests whether raw connectivity is driven by tiny transactions |",
        "| Sector-flow matrix | Global flows between 12 broad sectors | Gives a more readable production-system overview |",
        "",
        f"Out-strength Gini is {structural['out_strength_gini']:.3f}; in-strength Gini is {structural['in_strength_gini']:.3f}. Raw density is highly sensitive to the edge threshold, so do not treat every tiny positive transaction as equally important.",
        "",
        "### Top 10 detailed suppliers by out-strength",
        "",
        "| Rank | Node | Out-strength (billion current USD) |",
        "|---:|---|---:|",
    ]
    for k, r in enumerate(structural["top20_out_strength"][:10], 1):
        lines.append(f"| {k} | {r['node']} | {r['out_strength_million_USD']/1000:,.2f} |")
    lines += ["", "### Top 10 by in-strength", "", "| Rank | Node | In-strength (billion current USD) |", "|---:|---|---:|"]
    for k, r in enumerate(structural["top20_in_strength"][:10], 1):
        lines.append(f"| {k} | {r['node']} | {r['in_strength_million_USD']/1000:,.2f} |")
    lines += [
        "",
        "## 2. Dynamical network",
        "",
        "For tractable repeated dynamics, the 50 detailed activities are grouped into 12 broad activity groups for each economy. This produces about 972 country-sector nodes. Flow and gross-output totals are checked after aggregation.",
        "",
        r"$$P_{ij}=\\frac{Z^{cross}_{ij}}{\\sum_k Z^{all}_{kj}}, \\qquad e_j(t)=\\sum_i P_{ij}d_i(t).$$",
        "",
        "P gives each buyer's intermediate input share from each supplier. Its denominator includes all intermediate purchases, including flows internal to an aggregated sector; cross-node propagation removes the diagonal.",
        "",
        "### Linear approximation",
        r"$$d_j(t+1)=\\operatorname{clip}_{[0,1]}\\left[(1-\\rho)d_j(t)+\\beta e_j(t)\\right].$$",
        "",
        "The update is linear before clipping. Clipping bounds severity between zero and one, but means the implemented baseline is a clipped-linear model.",
        "",
        "### Nonlinear SIS-inspired model",
        r"$$d_j(t+1)=\\operatorname{clip}_{[0,1]}\\left[(1-\\rho)d_j(t)+(1-d_j(t))(1-e^{-\\beta e_j(t)})\\right].$$",
        "",
        "The nonlinear response saturates: when a node is already highly disrupted, the remaining opportunity for it to become more disrupted is smaller.",
        "",
        "| Parameter | Shared baseline value | Meaning |",
        "|---|---:|---|",
        "| Initial source disruption | 0.20 | Same 20% severity for each source |",
        "| Propagation strength beta | 0.30 | Shared influence of disrupted supplier exposure |",
        "| Recovery rate rho | 0.10 | Share of existing disruption that recovers per step |",
        "| Source hold | 3 steps | Externally maintain the source shock for three rounds |",
        "| Maximum length | 50 steps | Stop earlier if state change meets the convergence tolerance |",
        "",
        "These values are not estimated from ICIO. Shared parameters ensure that baseline differences mainly reflect source-node location, input dependencies, and gross-output weights.",
        "",
        "The baseline experiment shocks every country-sector node once, with all other nodes initially undisturbed, for each of the two models.",
        "",
        "## 3. Linear versus nonlinear results",
        "",
        "| Metric | Linear | Nonlinear |",
        "|---|---:|---:|",
        f"| Mean peak disruption | {100*lin['mean_peak_disruption']:.4f}% | {100*non['mean_peak_disruption']:.4f}% |",
        f"| Median peak disruption | {100*lin['median_peak_disruption']:.4f}% | {100*non['median_peak_disruption']:.4f}% |",
        f"| 95th percentile peak disruption | {100*lin['p95_peak_disruption']:.4f}% | {100*non['p95_peak_disruption']:.4f}% |",
        f"| Mean cumulative disruption (share-steps) | {lin['mean_cumulative_disruption_share_steps']:.4f} | {non['mean_cumulative_disruption_share_steps']:.4f} |",
        f"| Mean final disruption | {100*lin['mean_final_disruption']:.4f}% | {100*non['mean_final_disruption']:.4f}% |",
        f"| Mean maximum nodes at >=1% disruption | {lin['mean_max_affected_nodes_at_1pct']:.2f} | {non['mean_max_affected_nodes_at_1pct']:.2f} |",
        "",
        f"Mean paired peak difference (nonlinear minus linear): {100*comparison['mean_paired_nonlinear_minus_linear_peak']:.4f} percentage points. Mean paired cumulative difference: {comparison['mean_paired_nonlinear_minus_linear_cumulative']:.4f} share-steps.",
        "",
        "Interpret these numbers as differences between mathematical models under matched assumptions. Without a separate dataset of observed outages and recovery trajectories, neither model can be called more accurate in the predictive sense.",
        "",
        "## 4. Five representative sources",
        "",
        "| Case | Selection rule | Data-selected node |",
        "|---|---|---|",
    ]
    for r in reps:
        lines.append(f"| {r['case']} | {r['selection_rule']} | {r['source_node']} |")
    lines += [
        "",
        "A: highest out-strength. B: highest weighted PageRank, excluding A. C: highest approximate weighted betweenness, excluding A and B. D: highest cumulative nonlinear impact in the all-source census, excluding A-C. E: seeded random control from the middle 60% of out-strength, avoiding earlier choices.",
        "",
        "Betweenness is approximated on the aggregate graph after filtering edges below US$10 million; this limits the influence of very small flows. The random control uses a fixed seed.",
        "",
        "Each of the five sources is tested over 27 combinations: initial severity 0.10/0.20/0.40, beta 0.10/0.30/0.50, and recovery 0.05/0.10/0.20. Both models are run for all 27 combinations (270 runs total).",
        "",
        "## 5. Metrics and caveats",
        "",
        "- Peak disruption: maximum over time of gross-output-weighted node severity.",
        "- Cumulative disruption: sum of the weighted severity index over model rounds; represents magnitude and persistence.",
        "- Affected nodes: count with severity at least 1%, a reporting convention.",
        "- Final disruption: weighted index at the end of the simulation.",
        "",
        "The output-weighted index is not actual production lost, GDP loss, or welfare loss. Gross output counts intermediate production at multiple stages. The model also does not include spare capacity, inventories, price changes, or explicit positive substitution gains. Those require data and equations beyond these two requested disruption models.",
        "",
        f"Input: {metadata['source_file']}",
        f"Input SHA-256: {metadata['sha256']}",
        f"Run timestamp UTC: {metadata['run_timestamp_utc']}",
        f"Python {metadata['python_version']}; NumPy {metadata['numpy_version']}; SciPy {metadata['scipy_version']}.",
        "",
        "Reproduce all stages by running only: uv run master.py",
        "",
    ]
    (RESULTS / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("OECD ICIO 2022: full structural + linear/nonlinear analysis")
    print("One entry point: uv run master.py")
    print("=" * 72)

    data_path = find_data()
    print(f"[1/6] Data found: {data_path}")
    print("[2/6] Load and validate ICIO matrix...")
    nodes, dense_flows, output, metadata = load_icio(data_path)
    Z_full = sparse.csr_matrix(dense_flows)
    del dense_flows
    Z_full.sum_duplicates()
    Z_full.eliminate_zeros()
    print(f"      {len(nodes):,} detailed country-industry nodes loaded.")

    print("[3/6] Structural measures on the full detailed network...")
    structural = build_full_metrics(nodes, Z_full, output, RESULTS)
    aggregate_nodes, detail_map, Z_agg, X_agg, groups, S = aggregate_country_sector(nodes, Z_full, output)
    raw_flow = float(Z_full.sum())
    aggregated_flow = float(Z_agg.sum())
    raw_output = float(np.maximum(output, 0.0).sum())
    aggregated_output = float(np.maximum(X_agg, 0.0).sum())
    conservation = {
        "raw_flow_total_million_USD": raw_flow,
        "aggregated_flow_total_million_USD": aggregated_flow,
        "flow_relative_error": rel_error(raw_flow, aggregated_flow),
        "raw_gross_output_total_million_USD": raw_output,
        "aggregated_gross_output_total_million_USD": aggregated_output,
        "output_relative_error": rel_error(raw_output, aggregated_output),
    }
    if conservation["flow_relative_error"] > 1e-8 or conservation["output_relative_error"] > 1e-8:
        raise RuntimeError(f"Aggregation conservation check failed: {conservation}")
    aggregate_summary, pagerank, betweenness = build_aggregate_metrics(aggregate_nodes, Z_agg, X_agg, RESULTS)
    aggregate_summary.update({
        "flow_conservation_error_relative": conservation["flow_relative_error"],
        "output_conservation_error_relative": conservation["output_relative_error"],
        "sector_groups": SECTORS,
    })
    (RESULTS / "aggregation_validation.json").write_text(json.dumps(conservation, indent=2), encoding="utf-8")
    create_structural_figures(nodes, Z_full, groups, structural, RESULTS, IMAGES)
    print(f"      {len(aggregate_nodes):,} country-sector nodes after aggregation.")

    Z_cross = Z_agg.copy()
    Z_cross.setdiag(0.0)
    Z_cross.eliminate_zeros()
    out_strength = np.asarray(Z_cross.sum(axis=1)).ravel()
    input_share = build_input_share_matrix(Z_agg)
    Q = input_share.T.tocsr()
    print("[4/6] Run baseline shocks from every country-sector node in both models...")
    model_summary = run_baseline_census(
        aggregate_nodes, Q, X_agg, out_strength, pagerank, betweenness, RESULTS, IMAGES
    )
    print(f"      Linear mean peak disruption: {100*model_summary['linear_model']['mean_peak_disruption']:.3f}%")
    print(f"      Nonlinear mean peak disruption: {100*model_summary['nonlinear_model']['mean_peak_disruption']:.3f}%")

    print("[5/6] Summarize 5-node parameter sensitivity...")
    with (RESULTS / "representative_node_sensitivity.csv").open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    sensitivity_summary = sensitivity_summary_fn(rows)
    metadata.update({
        "sha256": file_sha256(data_path),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "detailed_node_count": len(nodes),
    })
    (RESULTS / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    complete = {
        "dataset": metadata, "structural_summary": structural, "aggregation": aggregate_summary,
        "conservation": conservation, "model_comparison": model_summary,
        "baseline_parameters": BASELINE, "sensitivity_summary": sensitivity_summary,
    }
    (RESULTS / "full_analysis_summary.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print("[6/6] Write report and execution manifest...")
    make_report(structural, aggregate_summary, model_summary, conservation, metadata)
    manifest = {
        "entry_point": "master.py", "run_command": "uv run master.py",
        "input_file": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
        "results_directory": str(RESULTS.relative_to(ROOT)), "images_directory": str(IMAGES.relative_to(ROOT)),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "result_files": sorted(p.name for p in RESULTS.iterdir() if p.is_file()),
        "image_files": sorted(p.name for p in IMAGES.iterdir() if p.is_file()),
    }
    (RESULTS / "execution_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print(f"Results: {RESULTS.relative_to(ROOT)}")
    print(f"Images:  {IMAGES.relative_to(ROOT)}")
    print(f"Report:  {(RESULTS / 'analysis_report.md').relative_to(ROOT)}")
    print(f"Elapsed: {manifest['elapsed_seconds']:.1f}s")
    print("Reminder: these are counterfactual dynamics, not forecasts of real-world losses.")
    print("=" * 72)


def sensitivity_summary_fn(rows):
    numeric = ("initial_disruption", "beta", "recovery_rate", "peak_output_weighted_disruption",
               "cumulative_output_weighted_disruption_share_steps", "final_output_weighted_disruption",
               "max_affected_nodes_at_1pct")
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return sensitivity_summary(rows)


if __name__ == "__main__":
    main()
