"""Linear and nonlinear disruption propagation on the aggregated IO network."""
from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.stats import spearmanr

BASELINE = {"initial_disruption": 0.20, "beta": 0.30, "recovery_rate": 0.10,
            "source_hold_steps": 3, "max_steps": 50, "convergence_tolerance": 1e-7}
SENSITIVITY = {"initial_disruption": [0.10, 0.20, 0.40], "beta": [0.10, 0.30, 0.50],
               "recovery_rate": [0.05, 0.10, 0.20], "source_hold_steps": 3, "max_steps": 50}
MODEL_LABELS = {"linear": "Clipped linear approximation", "nonlinear": "SIS-inspired saturating nonlinear model"}


def estimate_spectral_radius(Q: sparse.csr_matrix) -> float:
    """Estimate spectral radius of the nonnegative dependency influence matrix."""
    Q = Q.tocsr()
    if Q.shape[0] == 0 or Q.nnz == 0:
        return 0.0
    try:
        values = eigs(Q, k=1, which="LM", return_eigenvectors=False, tol=1e-7, maxiter=10000)
        return float(np.max(np.abs(values)))
    except Exception:
        # Deterministic power iteration fallback for unusual sparse eigensolver failures.
        v = np.ones(Q.shape[0], dtype=float)
        v /= np.linalg.norm(v)
        previous = 0.0
        for _ in range(10000):
            w = np.asarray(Q @ v).ravel()
            norm = float(np.linalg.norm(w))
            if norm <= 1e-15:
                return 0.0
            v = w / norm
            if abs(norm - previous) < 1e-10 * max(1.0, norm):
                return norm
            previous = norm
        return previous


def build_input_share_matrix(Z_aggregate: sparse.csr_matrix) -> sparse.csr_matrix:
    """P[i,j] = supplier i's share of buyer j's total intermediate purchases."""
    Zall = Z_aggregate.tocsr(copy=True).astype(np.float64)
    Zall.data = np.maximum(Zall.data, 0.0)
    Zall.eliminate_zeros()
    totals = np.asarray(Zall.sum(axis=0)).ravel()  # includes flows internal to an aggregate sector
    Zcross = Zall.copy()
    Zcross.setdiag(0.0)
    Zcross.eliminate_zeros()
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    P = (Zcross @ sparse.diags(inverse)).tocsr()
    P.eliminate_zeros()
    return P


def simulate_batch(Q, output_weights, source_indices, severities, betas, recovery_rates, model,
                   hold_steps=3, max_steps=50, tolerance=1e-7, keep_trajectories=False):
    """Run independent source shocks in parallel columns of a state matrix."""
    sources = np.asarray(source_indices, dtype=np.int64)
    severities = np.asarray(severities, dtype=float)
    betas = np.asarray(betas, dtype=float)
    recovery_rates = np.asarray(recovery_rates, dtype=float)
    count, n = len(sources), Q.shape[0]
    if not all(len(x) == count for x in (severities, betas, recovery_rates)):
        raise ValueError("Each scenario needs a source, severity, beta, and recovery rate.")
    if np.any((severities < 0) | (severities > 1)) or np.any(betas < 0) or np.any((recovery_rates < 0) | (recovery_rates > 1)):
        raise ValueError("Invalid severity, beta, or recovery rate.")
    state = np.zeros((n, count), dtype=np.float64)
    state[sources, np.arange(count)] = severities
    weights = np.maximum(np.asarray(output_weights, dtype=float), 0.0)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(n, 1.0 / max(n, 1))
    shares = np.asarray(weights @ state).ravel()
    peak = shares.copy()
    peak_step = np.zeros(count, dtype=int)
    cum = shares.copy()
    affected_max = (state >= 0.01).sum(axis=0).astype(int)
    affected_peak = affected_max.copy()
    affected_final = affected_max.copy()
    converged_each = np.zeros(count, dtype=bool)
    actual_steps = np.zeros(count, dtype=int)
    traces = []

    def append_trace(step, values):
        if not keep_trajectories:
            return
        L = np.asarray(weights @ values).ravel()
        for j in range(count):
            traces.append({
                "model": model, "source_index": int(sources[j]), "step": int(step),
                "source_severity": float(severities[j]), "beta": float(betas[j]),
                "recovery_rate": float(recovery_rates[j]), "output_weighted_disruption": float(L[j]),
                "affected_node_count_at_1pct": int(np.count_nonzero(values[:, j] >= 0.01)),
                "mean_node_disruption": float(values[:, j].mean()),
            })
    append_trace(0, state)

    for step in range(1, max_steps + 1):
        was_converged = converged_each.copy()
        exposure = np.asarray(Q @ state)
        if model == "linear":
            next_state = (1.0 - recovery_rates)[None, :] * state + betas[None, :] * exposure
        elif model == "nonlinear":
            hazard = -np.expm1(-betas[None, :] * np.maximum(exposure, 0.0))
            next_state = (1.0 - recovery_rates)[None, :] * state + (1.0 - state) * hazard
        else:
            raise ValueError(f"Unknown model: {model}")
        next_state = np.clip(next_state, 0.0, 1.0)
        # Externally hold the selected source during rounds 0..hold_steps-1.
        if step < hold_steps:
            next_state[sources, np.arange(count)] = severities
        # Freeze columns that have already reached a stable state. We still count
        # their stationary state in the cumulative metric through the common max_steps
        # horizon, so cumulative values remain comparable across source scenarios.
        if np.any(was_converged):
            next_state[:, was_converged] = state[:, was_converged]
        delta_by_source = np.max(np.abs(next_state - state), axis=0)
        actual_steps[~was_converged] = step
        newly_converged = (~was_converged) & (step >= hold_steps) & (delta_by_source < tolerance)
        converged_each |= newly_converged

        current_share = np.asarray(weights @ next_state).ravel()
        newly_affected = (next_state >= 0.01).sum(axis=0).astype(int)
        bigger = current_share > peak
        peak = np.maximum(peak, current_share)
        peak_step[bigger] = step
        affected_peak[bigger] = newly_affected[bigger]
        affected_max = np.maximum(affected_max, newly_affected)
        affected_final = newly_affected
        cum += current_share
        append_trace(step, next_state)
        state = next_state

    final_share = np.asarray(weights @ state).ravel()
    metrics = []
    for j in range(count):
        metrics.append({
            "source_index": int(sources[j]), "model": model,
            "initial_disruption": float(severities[j]), "beta": float(betas[j]),
            "recovery_rate": float(recovery_rates[j]), "source_hold_steps": int(hold_steps),
            "max_steps": int(max_steps), "actual_steps": int(actual_steps[j]),
            "peak_output_weighted_disruption": float(peak[j]), "peak_step": int(peak_step[j]),
            "cumulative_output_weighted_disruption_share_steps": float(cum[j]),
            "final_output_weighted_disruption": float(final_share[j]),
            "max_affected_nodes_at_1pct": int(affected_max[j]),
            "affected_nodes_at_peak_1pct": int(affected_peak[j]),
            "final_affected_nodes_at_1pct": int(affected_final[j]),
            "final_mean_node_disruption": float(state[:, j].mean()),
            "converged": bool(converged_each[j]),
        })
    return metrics, traces


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_baseline_census(aggregate_nodes, Q, output, out_strength, pagerank, betweenness, results_dir, image_dir):
    """Run both models once for every country-sector source node."""
    started = time.perf_counter()
    weights = np.maximum(np.asarray(output, dtype=float), 0.0)
    weights = weights / weights.sum() if weights.sum() else np.full(len(weights), 1.0 / len(weights))
    n = len(aggregate_nodes)
    rows = []
    batch_size = 48
    for model in ("linear", "nonlinear"):
        print(f"  {MODEL_LABELS[model]}: {n:,} initial-source scenarios", flush=True)
        model_started = time.perf_counter()
        for start in range(0, n, batch_size):
            sources = np.arange(start, min(start + batch_size, n))
            batch_rows, _ = simulate_batch(
                Q, weights, sources,
                np.full(len(sources), BASELINE["initial_disruption"]),
                np.full(len(sources), BASELINE["beta"]),
                np.full(len(sources), BASELINE["recovery_rate"]),
                model, BASELINE["source_hold_steps"], BASELINE["max_steps"],
                BASELINE["convergence_tolerance"], False,
            )
            for r in batch_rows:
                i = r["source_index"]
                r.update({
                    "source_node": aggregate_nodes[i],
                    "source_out_strength_million_USD": float(out_strength[i]),
                    "source_weighted_pagerank": float(pagerank[i]),
                    "source_weighted_betweenness_approx": float(betweenness[i]),
                    "source_gross_output_million_USD": float(output[i]),
                    "model_description": MODEL_LABELS[model],
                })
            rows.extend(batch_rows)
            if (start + batch_size) % 192 == 0 or start + batch_size >= n:
                print(f"    processed {min(start+batch_size,n)}/{n}", flush=True)
        print(f"    elapsed {time.perf_counter()-model_started:.1f}s", flush=True)
    rows.sort(key=lambda r: (r["model"], r["source_index"]))
    write_csv(results_dir / "baseline_dynamics_all_sources.csv", rows)
    by = {(r["source_index"], r["model"]): r for r in rows}
    compare = []
    for i, node in enumerate(aggregate_nodes):
        lin, non = by[(i, "linear")], by[(i, "nonlinear")]
        compare.append({
            "source_index": i, "source_node": node, "initial_disruption": BASELINE["initial_disruption"],
            "beta": BASELINE["beta"], "recovery_rate": BASELINE["recovery_rate"],
            "source_hold_steps": BASELINE["source_hold_steps"],
            "linear_peak_disruption": lin["peak_output_weighted_disruption"],
            "nonlinear_peak_disruption": non["peak_output_weighted_disruption"],
            "difference_nonlinear_minus_linear_peak": non["peak_output_weighted_disruption"] - lin["peak_output_weighted_disruption"],
            "linear_cumulative_disruption": lin["cumulative_output_weighted_disruption_share_steps"],
            "nonlinear_cumulative_disruption": non["cumulative_output_weighted_disruption_share_steps"],
            "difference_nonlinear_minus_linear_cumulative": non["cumulative_output_weighted_disruption_share_steps"] - lin["cumulative_output_weighted_disruption_share_steps"],
            "linear_final_disruption": lin["final_output_weighted_disruption"],
            "nonlinear_final_disruption": non["final_output_weighted_disruption"],
            "linear_peak_step": lin["peak_step"], "nonlinear_peak_step": non["peak_step"],
            "linear_max_affected_nodes": lin["max_affected_nodes_at_1pct"],
            "nonlinear_max_affected_nodes": non["max_affected_nodes_at_1pct"],
        })
    write_csv(results_dir / "linear_vs_nonlinear_source_comparison.csv", compare)

    representatives, rep_note = select_representatives(aggregate_nodes, rows, out_strength, pagerank, betweenness, output)
    write_csv(results_dir / "representative_nodes.csv", representatives)
    sensitivity = run_sensitivity(representatives, Q, output, aggregate_nodes, results_dir)
    write_csv(results_dir / "representative_node_sensitivity.csv", sensitivity)

    rep_indices = [r["source_index"] for r in representatives]
    rep_weights = weights
    trajectories = []
    for model in ("linear", "nonlinear"):
        _, traces = simulate_batch(
            Q, rep_weights, np.asarray(rep_indices),
            np.full(len(rep_indices), BASELINE["initial_disruption"]),
            np.full(len(rep_indices), BASELINE["beta"]),
            np.full(len(rep_indices), BASELINE["recovery_rate"]),
            model, BASELINE["source_hold_steps"], BASELINE["max_steps"],
            BASELINE["convergence_tolerance"], True,
        )
        for tr in traces:
            tr["source_node"] = aggregate_nodes[tr["source_index"]]
        trajectories.extend(traces)
    write_csv(results_dir / "representative_trajectories.csv", trajectories)
    summary = summarize(rows, compare, out_strength, pagerank)
    dependency_rho = estimate_spectral_radius(Q)
    critical_beta = BASELINE["recovery_rate"] / dependency_rho if dependency_rho > 0 else None
    summary["linearized_disease_free_stability"] = {
        "dependency_matrix_spectral_radius": dependency_rho,
        "approximate_beta_critical": critical_beta,
        "baseline_beta": BASELINE["beta"],
        "baseline_beta_over_beta_critical": BASELINE["beta"] / critical_beta if critical_beta else None,
        "linearization_stable_at_baseline": bool(
            critical_beta is not None and BASELINE["beta"] < critical_beta
        ),
        "derivation": "Near zero disruption, both models linearize to J=(1-rho)I+beta Q; the disease-free state is locally stable if beta*rho(Q)<rho.",
        "caveat": "This is a local stability diagnostic for the un-clipped linearization, not an empirical crisis threshold. Clipping and nonlinear saturation change the global dynamics.",
    }
    summary.update({
        "baseline_parameters": BASELINE, "sensitivity_grid": SENSITIVITY,
        "source_node_count": n, "models": MODEL_LABELS,
        "simulation_wall_time_seconds": float(time.perf_counter() - started),
        "representative_nodes": representatives, "representative_selection_note": rep_note,
        "sensitivity_scenario_count": int(len(sensitivity)),
        "interpretation": "Comparison quantifies behaviour under matched assumptions, not predictive accuracy: annual ICIO contains no time-stamped disruption labels.",
    })
    (results_dir / "model_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(compare, trajectories, representatives, image_dir)
    return summary


def select_representatives(nodes, metrics, out_strength, pagerank, betweenness, output):
    """Select five distinct node roles after all-node baseline results are available."""
    n = len(nodes)
    eligible = [i for i in range(n) if out_strength[i] + max(output[i], 0) > 0] or list(range(n))
    taken = set()
    choices = []
    a = max(eligible, key=lambda i: out_strength[i])
    choices.append(("A", "Highest weighted out-strength", a)); taken.add(a)
    b = max((i for i in eligible if i not in taken), key=lambda i: pagerank[i])
    choices.append(("B", "Highest weighted PageRank as downstream-influence proxy", b)); taken.add(b)
    c = max((i for i in eligible if i not in taken), key=lambda i: betweenness[i])
    choices.append(("C", "Highest approximate weighted betweenness", c)); taken.add(c)
    nonlinear = {r["source_index"]: r for r in metrics if r["model"] == "nonlinear"}
    d = max((i for i in eligible if i not in taken),
            key=lambda i: nonlinear[i]["cumulative_output_weighted_disruption_share_steps"])
    choices.append(("D", "Highest nonlinear cumulative impact in source census", d)); taken.add(d)
    strengths = np.asarray([out_strength[i] for i in eligible])
    lo, hi = np.quantile(strengths, [0.20, 0.80]) if len(strengths) > 1 else (0, np.inf)
    pool = [i for i in eligible if i not in taken and lo <= out_strength[i] <= hi]
    if not pool:
        pool = [i for i in eligible if i not in taken]
    rng = np.random.default_rng(2022)
    e = int(rng.choice(pool))
    choices.append(("E", "Seeded random control from middle out-strength band", e)); taken.add(e)
    out = []
    for case, rule, i in choices:
        non = nonlinear[i]
        out.append({
            "case": case, "selection_rule": rule, "source_index": i, "source_node": nodes[i],
            "out_strength_million_USD": float(out_strength[i]), "weighted_pagerank": float(pagerank[i]),
            "weighted_betweenness_approx": float(betweenness[i]), "gross_output_million_USD": float(output[i]),
            "baseline_nonlinear_peak_disruption": non["peak_output_weighted_disruption"],
            "baseline_nonlinear_cumulative_disruption": non["cumulative_output_weighted_disruption_share_steps"],
        })
    return out, {"distinct_sources": len(taken), "random_control_quantile_band": [0.20, 0.80]}


def run_sensitivity(selection, Q, output, nodes, results_dir):
    weights = np.maximum(np.asarray(output, dtype=float), 0.0)
    weights = weights / weights.sum() if weights.sum() else np.full(len(weights), 1 / len(weights))
    targets = []
    for target in selection:
        for severity, beta, rho in itertools.product(
            SENSITIVITY["initial_disruption"], SENSITIVITY["beta"], SENSITIVITY["recovery_rate"]
        ):
            for model in ("linear", "nonlinear"):
                targets.append((target, severity, beta, rho, model))
    source_all = np.asarray([t[0]["source_index"] for t in targets], dtype=np.int64)
    severity_all = np.asarray([t[1] for t in targets])
    beta_all = np.asarray([t[2] for t in targets])
    rho_all = np.asarray([t[3] for t in targets])
    results = []
    for start in range(0, len(targets), 48):
        stop = min(start + 48, len(targets))
        for model in ("linear", "nonlinear"):
            cols = [i for i in range(start, stop) if targets[i][4] == model]
            if not cols:
                continue
            rows, _ = simulate_batch(
                Q, weights, source_all[cols], severity_all[cols], beta_all[cols], rho_all[cols],
                model, SENSITIVITY["source_hold_steps"], SENSITIVITY["max_steps"],
                BASELINE["convergence_tolerance"], False,
            )
            for j, row in enumerate(rows):
                target, severity, beta, rho, _ = targets[cols[j]]
                row.update({"case": target["case"], "selection_rule": target["selection_rule"],
                            "source_node": nodes[row["source_index"]], "model_description": MODEL_LABELS[model],
                            "severity_grid": severity, "beta_grid": beta, "recovery_rate_grid": rho})
                results.append(row)
    results.sort(key=lambda r: (r["case"], r["initial_disruption"], r["beta"], r["recovery_rate"], r["model"]))
    return results


def summarize(rows, compare, out_strength, pagerank):
    def stats(model):
        subset = [r for r in rows if r["model"] == model]
        peak = np.asarray([r["peak_output_weighted_disruption"] for r in subset])
        cum = np.asarray([r["cumulative_output_weighted_disruption_share_steps"] for r in subset])
        return {
            "source_scenario_count": len(subset), "mean_peak_disruption": float(peak.mean()),
            "median_peak_disruption": float(np.median(peak)), "p95_peak_disruption": float(np.quantile(peak, .95)),
            "mean_cumulative_disruption_share_steps": float(cum.mean()),
            "mean_final_disruption": float(np.mean([r["final_output_weighted_disruption"] for r in subset])),
            "mean_max_affected_nodes_at_1pct": float(np.mean([r["max_affected_nodes_at_1pct"] for r in subset])),
            "runs_converged": int(sum(r["converged"] for r in subset)),
        }
    lin = stats("linear"); non = stats("nonlinear")
    peak_diff = np.asarray([r["difference_nonlinear_minus_linear_peak"] for r in compare])
    cum_diff = np.asarray([r["difference_nonlinear_minus_linear_cumulative"] for r in compare])
    nonlinear_cum = np.asarray([r["nonlinear_cumulative_disruption"] for r in compare])
    rho1 = spearmanr(out_strength, nonlinear_cum).statistic
    rho2 = spearmanr(pagerank, nonlinear_cum).statistic
    return {
        "linear_model": lin, "nonlinear_model": non,
        "mean_paired_nonlinear_minus_linear_peak": float(peak_diff.mean()),
        "median_paired_nonlinear_minus_linear_peak": float(np.median(peak_diff)),
        "mean_paired_nonlinear_minus_linear_cumulative": float(cum_diff.mean()),
        "spearman_out_strength_vs_nonlinear_cumulative": float(rho1) if np.isfinite(rho1) else None,
        "spearman_pagerank_vs_nonlinear_cumulative": float(rho2) if np.isfinite(rho2) else None,
        "largest_nonlinear_peak_source": compare[int(np.argmax([r["nonlinear_peak_disruption"] for r in compare]))]["source_node"],
        "largest_nonlinear_cumulative_source": compare[int(np.argmax(nonlinear_cum))]["source_node"],
        "accuracy_note": "No observed outage time series are available, so this compares simulated behaviour and sensitivity, not predictive accuracy.",
    }


def make_plots(compare, trajectories, representatives, image_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["source_node"] for r in compare]
    lin = np.asarray([r["linear_peak_disruption"] for r in compare])
    non = np.asarray([r["nonlinear_peak_disruption"] for r in compare])
    order = np.argsort(non)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(order)); width = .38
    ax.bar(x-width/2, lin[order], width, label="Clipped linear")
    ax.bar(x+width/2, non[order], width, label="Nonlinear saturating")
    ax.set_xticks(x); ax.set_xticklabels([labels[i] for i in order], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Peak output-weighted disruption index")
    ax.set_title("Top 20 source shocks by nonlinear peak impact")
    ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(image_dir / "linear_vs_nonlinear_top20.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 6))
    ax.scatter(lin, non, s=14, alpha=.65)
    vmax = max(float(lin.max(initial=0)), float(non.max(initial=0)), 1e-6)
    ax.plot([0, vmax], [0, vmax], linestyle="--", linewidth=1)
    ax.set_xlabel("Linear peak disruption"); ax.set_ylabel("Nonlinear peak disruption")
    ax.set_title("Same sources, same parameter values")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(image_dir / "linear_vs_nonlinear_scatter.png", dpi=180); plt.close(fig)

    if trajectories:
        selected = {r["source_node"] for r in representatives}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
        for model, ax in zip(("linear", "nonlinear"), axes):
            for node in sorted(selected):
                series = sorted([r for r in trajectories if r["model"] == model and r["source_node"] == node],
                                key=lambda r: r["step"])
                if series:
                    ax.plot([r["step"] for r in series], [r["output_weighted_disruption"] for r in series],
                            marker="o", markersize=2, linewidth=1.1, label=node)
            ax.set_title(MODEL_LABELS[model]); ax.set_xlabel("Model step (not calendar time)")
            ax.set_ylabel("Output-weighted disruption index"); ax.grid(alpha=.25); ax.legend(fontsize=6)
        fig.suptitle("Representative-source trajectories under common baseline parameters")
        fig.tight_layout(); fig.savefig(image_dir / "representative_trajectories.png", dpi=180); plt.close(fig)
