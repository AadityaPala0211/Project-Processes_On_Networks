"""Impact summaries, reproducible CSV output, and dynamic-vs-structural rank statistics."""
import csv
import numpy as np
from scipy.stats import spearmanr

def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if fieldnames is None: fieldnames=list(rows[0].keys()) if rows else []
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)

def summarize_model(rows):
    result={}
    for model in sorted({r["model"] for r in rows}):
        sub=[r for r in rows if r["model"]==model]
        result[model]={"source_scenarios":len(sub)}
        for metric in ("peak_output_weighted_disruption","cumulative_disruption_share_steps","final_output_weighted_disruption"):
            x=np.asarray([float(r[metric]) for r in sub])
            result[model][metric+"_mean"]=float(x.mean()) if x.size else None
            result[model][metric+"_median"]=float(np.median(x)) if x.size else None
            result[model][metric+"_p95"]=float(np.quantile(x,.95)) if x.size else None
        result[model]["mean_max_affected_nodes"]=float(np.mean([r["max_affected_nodes"] for r in sub])) if sub else None
        result[model]["converged_count"]=int(sum(bool(r["converged"]) for r in sub))
    return result

def rank_correlations(census_rows, structural_rows, keys):
    structural={r["node"]:r for r in structural_rows}
    out=[]
    for model in sorted({r["model"] for r in census_rows}):
        by_node={r["source_node"]:r for r in census_rows if r["model"]==model}
        for dyn in ("peak_output_weighted_disruption","cumulative_disruption_share_steps","final_output_weighted_disruption"):
            for structural_key in keys:
                pairs=[]
                for node,r in structural.items():
                    if node not in by_node: continue
                    try: x=float(by_node[node][dyn]); y=float(r.get(structural_key,""))
                    except (ValueError,TypeError): continue
                    if np.isfinite(x) and np.isfinite(y): pairs.append((x,y))
                if len(pairs)<3: rho=""
                else:
                    x,y=np.asarray(pairs).T; value=spearmanr(x,y).statistic
                    rho=float(value) if np.isfinite(value) else ""
                out.append({"model":model,"dynamic_metric":dyn,"structural_metric":structural_key,
                            "n_nodes":len(pairs),"spearman_rho":rho})
    return out
