#!/usr/bin/env python3
"""Signed counterfactual shock propagation on the OECD regular ICIO network.

The state x_i is a bounded dimensionless pressure/opportunity index, not GDP or
observed output loss. Annual ICIO flows define the network; dynamic coefficients
are explicit teaching assumptions. Run with the raw ICIO CSV for new results.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COUNTRY_NAMES = {"USA":"United States", "CHN":"China", "RUS":"Russia", "ROW":"Rest of World", "IND":"India", "SAU":"Saudi Arabia", "NOR":"Norway", "VEN":"Venezuela"}
INDUSTRY_NAMES = {"B06":"Crude petroleum and natural gas extraction", "C301":"Building of ships and boats", "G":"Wholesale and retail trade; motor vehicle repair"}

def split_node(label: str):
    return label.split("_", 1)

def flow_matrices(Z: np.ndarray):
    """Return positive cross-node flows, total inputs (including self-use), and out-strength."""
    positive = np.maximum(np.asarray(Z, dtype=float), 0.0)
    inputs = positive.sum(axis=0)
    np.fill_diagonal(positive, 0.0)
    Zcross = sparse.csr_matrix(positive)
    Zcross.eliminate_zeros()
    out_strength = np.asarray(Zcross.sum(axis=1)).ravel()
    return Zcross, inputs, out_strength

def dependency_matrix(Zcross, input_totals):
    inv = np.zeros_like(input_totals, dtype=float)
    valid = input_totals > 0
    inv[valid] = 1.0 / input_totals[valid]
    D = (Zcross @ sparse.diags(inv)).tocsr()
    D.eliminate_zeros()
    return D

def substitution_matrix(Zcross, nodes, out_strength):
    """C[alternative, disrupted] weights competitors in the same detailed industry.

    Competitor weights use cosine similarity in their supplier-to-buyer flow rows.
    If there is no shared buyer pattern, observed output-flow strength is a fallback.
    This models a possible reallocation opportunity; it does not identify spare capacity.
    """
    codes = [split_node(node)[1] for node in nodes]
    by_code = {}
    for i, code in enumerate(codes):
        by_code.setdefault(code, []).append(i)
    rows, cols, vals = [], [], []
    for ids_list in by_code.values():
        if len(ids_list) < 2:
            continue
        ids = np.asarray(ids_list, dtype=int)
        block = Zcross[ids, :].tocsr()
        gram = (block @ block.T).toarray()
        norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
        denom = np.outer(norms, norms)
        sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        np.fill_diagonal(sim, 0.0)
        for j_local, source in enumerate(ids):
            weights = sim[:, j_local].copy()
            weights[j_local] = 0.0
            if weights.sum() <= 0:
                weights = np.maximum(out_strength[ids], 0.0).astype(float)
                weights[j_local] = 0.0
            if weights.sum() <= 0:
                continue
            weights /= weights.sum()
            for i_local in np.flatnonzero(weights > 0):
                rows.append(int(ids[i_local])); cols.append(int(source)); vals.append(float(weights[i_local]))
    C = sparse.csr_matrix((vals, (rows, cols)), shape=Zcross.shape)
    C.eliminate_zeros()
    return C

def simulate(D, C, output, source: int, severity=1.0, steps=12, persistence=0.55, transmission=0.35, substitution=0.25):
    """Discrete signed linear approximation with clipping to [-1,1].

    x(t+1) = clip[p*x(t) + a*D.T*x(t) + b*C*max(-x(t),0), -1, 1].
    Negative x means adverse supply/production pressure; positive x means possible
    opportunity for same-industry alternative suppliers.
    """
    if not 0 < severity <= 1:
        raise ValueError("severity must be in (0, 1].")
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    n = len(output)
    state = np.zeros(n, dtype=float)
    state[source] = -severity
    weights = np.maximum(np.asarray(output, dtype=float), 0.0)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n) / n
    path = [state.copy()]
    records = []
    def metric(t, x):
        return {"step": t,
                "output_weighted_negative_pressure_share": float(weights @ np.maximum(-x, 0.0)),
                "output_weighted_positive_opportunity_share": float(weights @ np.maximum(x, 0.0)),
                "net_signed_index": float(weights @ x),
                "negative_nodes": int(np.sum(x < -1e-10)),
                "positive_nodes": int(np.sum(x > 1e-10)),
                "max_negative_magnitude": float(np.max(np.maximum(-x, 0.0), initial=0.0)),
                "max_positive_magnitude": float(np.max(np.maximum(x, 0.0), initial=0.0))}
    records.append(metric(0, state))
    for t in range(1, steps + 1):
        xnext = persistence * state + transmission * (D.T @ state) + substitution * (C @ np.maximum(-state, 0.0))
        state = np.clip(np.asarray(xnext).ravel(), -1.0, 1.0)
        path.append(state.copy())
        records.append(metric(t, state))
    return records, np.vstack(path), weights

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

def group_industry_totals(nodes, out_strength):
    out = {}
    for node, strength in zip(nodes, out_strength):
        _, code = split_node(node)
        row = out.setdefault(code, {"industry_code": code, "industry_name": INDUSTRY_NAMES.get(code, code), "global_out_strength_million_USD": 0.0, "positive_country_nodes": 0})
        row["global_out_strength_million_USD"] += float(strength)
        row["positive_country_nodes"] += int(strength > 0)
    return sorted(out.values(), key=lambda r: r["global_out_strength_million_USD"], reverse=True)

def pick_scenarios(nodes, out_strength, output):
    by_code = {}
    for i, node in enumerate(nodes):
        _, code = split_node(node)
        by_code.setdefault(code, []).append(i)
    totals = group_industry_totals(nodes, out_strength)
    ranked = [r for r in totals if r["industry_code"] != "T" and r["global_out_strength_million_USD"] > 0]
    scenarios = []
    if ranked:
        for key, row, rule in [("top_global_industry", ranked[0], "Highest detailed activity by global outgoing intermediate-flow strength"),
                               ("lowest_nonzero_industry", ranked[-1], "Lowest positive detailed activity by global outgoing intermediate-flow strength; household-use activity T excluded")]:
            code = row["industry_code"]
            ids = by_code[code]
            source = max(ids, key=lambda i: out_strength[i])
            scenarios.append({"scenario": key, "source_index": int(source), "source_node": nodes[source], "industry_code": code,
                              "industry_name": INDUSTRY_NAMES.get(code, code), "industry_global_out_strength_million_USD": float(row["global_out_strength_million_USD"]),
                              "source_out_strength_million_USD": float(out_strength[source]), "source_gross_output_million_USD": float(output[source]), "selection_rule": rule})
    if "RUS_B06" in nodes:
        source = nodes.index("RUS_B06")
        scenarios.append({"scenario":"russia_oil_extraction", "source_index":source, "source_node":"RUS_B06", "industry_code":"B06",
                          "industry_name":INDUSTRY_NAMES["B06"], "industry_global_out_strength_million_USD": next((r["global_out_strength_million_USD"] for r in totals if r["industry_code"] == "B06"), 0.0),
                          "source_out_strength_million_USD":float(out_strength[source]), "source_gross_output_million_USD":float(output[source]), "selection_rule":"User-specified illustrative node; B06 combines crude petroleum and natural gas extraction"})
    return scenarios, totals

def export_156_example(base, Z, nodes, output, outdir, countries_n=12):
    regions, agg_nodes, _, Za, Xa, _ = base.make_aggregate(Z, nodes, output)
    regional_output = {c: 0.0 for c in regions}
    for label, val in zip(agg_nodes, Xa):
        regional_output[split_node(label)[0]] += float(val)
    selected = sorted(regions, key=lambda c: regional_output[c], reverse=True)[:countries_n]
    if "ROW" in regions and "ROW" not in selected:
        selected[-1] = "ROW"
    wanted = set(selected)
    ids = [i for i, label in enumerate(agg_nodes) if split_node(label)[0] in wanted]
    local = {old: new for new, old in enumerate(ids)}
    node_rows = []
    for old in ids:
        c, sector = split_node(agg_nodes[old])
        node_rows.append({"example_node_id": local[old], "node": agg_nodes[old], "country_code": c,
                          "country_name": COUNTRY_NAMES.get(c, c), "sector_group": sector,
                          "gross_output_million_USD": float(Xa[old])})
    edge_rows = []
    sub = Za[ids, :][:, ids].tocoo()
    for i, j, value in zip(sub.row, sub.col, sub.data):
        if i != j and value > 0:
            edge_rows.append({"source_node_id": int(i), "target_node_id": int(j), "source_node": agg_nodes[ids[int(i)]],
                              "target_node": agg_nodes[ids[int(j)]], "intermediate_flow_million_USD": float(value)})
    write_csv(outdir / "worked_example_nodes.csv", node_rows)
    write_csv(outdir / "worked_example_edges.csv", edge_rows)
    return len(node_rows), len(edge_rows)

def main():
    parser = argparse.ArgumentParser(description="Run signed positive/negative shock propagation on OECD regular ICIO.")
    parser.add_argument("csv", type=Path, help="Path to the raw OECD ICIO 2025 edition CSV for reference year 2022 (e.g. 2022_SML.csv)")
    parser.add_argument("--out", type=Path, default=Path("results_signed"))
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--severity", type=float, default=1.0, help="Initial negative state magnitude in (0,1]")
    parser.add_argument("--persistence", type=float, default=0.55)
    parser.add_argument("--transmission", type=float, default=0.35)
    parser.add_argument("--substitution", type=float, default=0.25)
    parser.add_argument("--example-countries", type=int, default=12, help="12 countries x 13 broad sectors = up to 156 teaching nodes")
    args = parser.parse_args()
    if not 1 <= args.example_countries <= 81:
        parser.error("--example-countries must be between 1 and 81")
    outdir = args.out.resolve(); outdir.mkdir(parents=True, exist_ok=True)
    sp = importlib.util.spec_from_file_location("scanalysis", HERE / "supply_chain_complete_analysis.py")
    if sp is None or sp.loader is None:
        raise RuntimeError("Could not import supply_chain_complete_analysis.py from the code directory")
    base = importlib.util.module_from_spec(sp); sys.modules[sp.name] = base; sp.loader.exec_module(base)
    print("Loading raw ICIO matrix…", flush=True)
    _, nodes, Z, output = base.load_icio(args.csv.resolve())
    output = np.maximum(np.nan_to_num(output, nan=0.0), 0.0)
    Zcross, input_totals, out_strength = flow_matrices(Z)
    D = dependency_matrix(Zcross, input_totals)
    C = substitution_matrix(Zcross, nodes, out_strength)
    scenarios, industry_totals = pick_scenarios(nodes, out_strength, np.maximum(np.nan_to_num(output), 0.0))
    ranked = sorted(industry_totals, key=lambda r: r["global_out_strength_million_USD"], reverse=True)
    write_csv(outdir / "signed_industry_ranking.csv", ranked)
    in_strength = np.asarray(Zcross.sum(axis=0)).ravel()
    clean_output = np.maximum(np.nan_to_num(output, nan=0.0), 0.0)
    node_ranking = []
    for rank, idx in enumerate(np.argsort(out_strength)[::-1], start=1):
        country, code = split_node(nodes[int(idx)])
        node_ranking.append({"rank_by_out_strength": rank, "node": nodes[int(idx)],
            "country_code": country, "country_name": COUNTRY_NAMES.get(country, country),
            "industry_code": code, "industry_name": INDUSTRY_NAMES.get(code, code),
            "out_strength_million_USD": float(out_strength[idx]),
            "in_strength_million_USD": float(in_strength[idx]),
            "gross_output_million_USD": float(clean_output[idx])})
    write_csv(outdir / "signed_node_ranking.csv", node_ranking)
    summaries, series_rows, effects = [], [], []
    severities = sorted(set([float(args.severity), float(args.severity) / 2.0]))
    for scenario in scenarios:
        for severity in severities:
            records, path, out_weights = simulate(D, C, output, scenario["source_index"], severity=severity, steps=args.steps,
                persistence=args.persistence, transmission=args.transmission, substitution=args.substitution)
            for r in records:
                series_rows.append({"scenario":scenario["scenario"], "source_node":scenario["source_node"], "severity":severity, **r})
            negpeak = np.max(np.maximum(-path, 0.0), axis=0)
            pospeak = np.max(np.maximum(path, 0.0), axis=0)
            final = path[-1]
            negatives = [float(out_weights @ np.maximum(-p, 0.0)) for p in path]
            positives = [float(out_weights @ np.maximum(p, 0.0)) for p in path]
            peak_neg = max(negatives, default=0.0); peak_pos = max(positives, default=0.0)
            summaries.append({**scenario, "severity":severity, "steps":args.steps, "persistence":args.persistence, "transmission":args.transmission,
                "substitution":args.substitution, "peak_output_weighted_negative_pressure_index":peak_neg,
                "peak_output_weighted_positive_opportunity_index":peak_pos, "final_output_weighted_negative_pressure_index":negatives[-1],
                "final_output_weighted_positive_opportunity_index":positives[-1], "negative_nodes_at_peak":int(np.sum(negpeak > 1e-10)),
                "positive_nodes_at_peak":int(np.sum(pospeak > 1e-10)), "largest_positive_opportunity_node":nodes[int(np.argmax(pospeak))] if pospeak.max(initial=0)>0 else "none",
                "largest_negative_pressure_node":nodes[int(np.argmax(negpeak))] if negpeak.max(initial=0)>0 else "none",
                "interpretation":"Dimensionless model index; not lost dollars, GDP loss, or forecast."})
            for i, node in enumerate(nodes):
                if negpeak[i] <= 1e-10 and pospeak[i] <= 1e-10 and abs(final[i]) <= 1e-10:
                    continue
                c, code = split_node(node)
                effects.append({"scenario":scenario["scenario"], "source_node":scenario["source_node"], "severity":severity, "node":node,
                    "country_code":c, "country_name":COUNTRY_NAMES.get(c,c), "industry_code":code, "industry_name":INDUSTRY_NAMES.get(code,code),
                    "gross_output_million_USD":float(max(output[i],0.0)), "peak_negative_pressure_index":float(negpeak[i]),
                    "peak_positive_opportunity_index":float(pospeak[i]), "final_signed_index":float(final[i])})
    write_csv(outdir / "signed_shock_scenario_summary.csv", summaries)
    write_csv(outdir / "signed_shock_time_series.csv", series_rows)
    write_csv(outdir / "signed_shock_node_effects.csv", effects)
    ex_nodes, ex_edges = export_156_example(base, Z, nodes, output, outdir, args.example_countries)
    make_figures(series_rows, summaries, outdir)
    meta = {"dataset_edition":"OECD Inter-Country Input-Output 2025 edition (regular); reference year 2022 if raw file is 2022_SML.csv",
            "raw_node_count":len(nodes), "raw_flow_shape":list(Z.shape), "worked_example_node_count":ex_nodes, "worked_example_edge_count":ex_edges,
            "dependency_definition":"D[i,j] = Zcross[i,j] / total intermediate inputs of buyer j; total input denominator includes self-use while propagation edges exclude self-links.",
            "substitution_definition":"C[alternative, disrupted] uses cosine similarity of supplier-to-buyer flow rows, only within the same detailed activity code; if similarities are all zero, outgoing strength shares are used.",
            "model":"x(t+1)=clip(p*x(t)+a*D.T*x(t)+b*C*max(-x(t),0), -1, 1)",
            "parameters":{"persistence":args.persistence,"transmission":args.transmission,"substitution":args.substitution,"steps":args.steps,"severities":severities},
            "assumptions":["Annual ICIO flows do not identify shock speed, substitution elasticities, inventories, spare capacity, or price effects.",
                "Positive effects are relative opportunity signals for same-activity alternative suppliers, not predicted export gains.",
                "Country-sector node B06 combines crude petroleum and natural gas extraction.",
                "If an economy is only included in the Rest-of-World aggregate, it cannot be separated as a named node."],
            "generated_at_runtime":True}
    (outdir / "signed_shock_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Finished signed-shock run: {len(nodes):,} detailed nodes; worked example {ex_nodes} nodes / {ex_edges} edges.")
    print("The results are counterfactual indices, not observed economic losses.")

def make_figures(series_rows, summaries, outdir):
    fig_dir = outdir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for r in series_rows:
        if float(r["severity"]) == max(float(s["severity"]) for s in summaries if s["scenario"] == r["scenario"]):
            grouped.setdefault(r["scenario"], []).append(r)
    if grouped:
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        for label, rows in grouped.items():
            rows = sorted(rows, key=lambda r: int(r["step"]))
            ax.plot([int(r["step"]) for r in rows], [100*float(r["output_weighted_negative_pressure_share"]) for r in rows], marker="o", markersize=2, label=f"{label}: negative")
            ax.plot([int(r["step"]) for r in rows], [100*float(r["output_weighted_positive_opportunity_share"]) for r in rows], linestyle="--", label=f"{label}: positive")
        ax.set_xlabel("Model step (not calendar time)"); ax.set_ylabel("Output-weighted index (%)")
        ax.set_title("Signed shock propagation: downside and substitution opportunity")
        ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
        fig.savefig(fig_dir / "signed_shock_time_series.png", dpi=180); plt.close(fig)
    rows = [r for r in summaries if float(r["severity"]) == 1.0]
    if rows:
        labels = [str(r["scenario"]) for r in rows]
        x = np.arange(len(rows)); width = .38
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.bar(x-width/2, [100*float(r["peak_output_weighted_negative_pressure_index"]) for r in rows], width, label="Negative pressure")
        ax.bar(x+width/2, [100*float(r["peak_output_weighted_positive_opportunity_index"]) for r in rows], width, label="Positive opportunity")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel("Peak output-weighted index (%)"); ax.set_title("Targeted shock comparison (model indices)")
        ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "signed_shock_target_comparison.png", dpi=180); plt.close(fig)

if __name__ == "__main__":
    main()
