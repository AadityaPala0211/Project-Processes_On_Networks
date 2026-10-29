"""Batched sensitivity grid with all-source, top-K, and stratified source modes."""
from __future__ import annotations
import csv
import itertools
import numpy as np
from .config import MODEL_NAMES, Parameters, SensitivityGrid
from .metrics import write_csv
from .simulator import simulate_batch, prepare_model3_edges
from .source_census import eligible_sources, select_sources
from .structural_metrics import load_structural_metrics

def run_sensitivity(P,nodes,output,W,structural_dir,outdir,params=Parameters(),
                    source_mode="stratified",top_k=50,stratified_per_bin=5,
                    seed=2022,batch_size=16,edge_chunk_size=500_000,grid=SensitivityGrid()):
    if source_mode not in {"all","top-k","stratified"}: raise ValueError("invalid source mode")
    structural=load_structural_metrics(structural_dir,nodes,W)
    eligible=eligible_sources(W,output)
    if source_mode=="stratified" and stratified_per_bin>1:
        # Draw a reproducible, balanced subset within deciles of log out-strength.
        rng=np.random.default_rng(seed)
        score=np.log1p(np.asarray([float(r.get("out_strength_million_current_usd",0) or 0) for r in structural]))
        cuts=np.unique(np.quantile(score[eligible],np.linspace(0,1,11)))
        picks=[]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            pool=eligible[(score[eligible]>=lo)&(score[eligible]<=hi)]
            if pool.size:
                picks.extend(rng.choice(pool,size=min(stratified_per_bin,len(pool)),replace=False).astype(int).tolist())
        sources=np.asarray(sorted(set(picks)),dtype=int)
    else:
        sources=select_sources(eligible,source_mode,top_k,structural,seed)
    if not len(sources): raise ValueError("No source nodes selected for sensitivity")
    combos=list(itertools.product(grid.severities,grid.betas,grid.recovery_rates))
    total=len(sources)*len(combos)*len(MODEL_NAMES); done=0; rows=[]
    edge_arrays=prepare_model3_edges(P)
    for severity,beta,recovery in combos:
        for model in MODEL_NAMES:
            for start in range(0,len(sources),batch_size):
                batch=sources[start:start+batch_size]
                result_rows,_=simulate_batch(P,output,batch,model,severity=severity,beta=beta,
                    recovery_rate=recovery,source_hold_steps=params.source_hold_steps,
                    max_steps=params.max_steps,tolerance=params.convergence_tolerance,
                    affected_threshold=params.affected_threshold,batch_edge_arrays=edge_arrays,
                    edge_chunk_size=edge_chunk_size)
                for row in result_rows:
                    row["source_node"]=nodes[row["source_index"]]
                    rows.append(row); done+=1
                if done%1000==0 or done==total:
                    print(f"[sensitivity] {done:,}/{total:,} scenarios",flush=True)
    write_csv(outdir/"parameter_sensitivity.csv",rows)
    summary=[]
    for model in MODEL_NAMES:
        for severity,beta,recovery in combos:
            sub=[r for r in rows if r["model"]==model and r["initial_disruption"]==severity and r["beta"]==beta and r["recovery_rate"]==recovery]
            for metric in ("peak_output_weighted_disruption","cumulative_disruption_share_steps","final_output_weighted_disruption"):
                vals=np.asarray([float(r[metric]) for r in sub])
                if vals.size:
                    summary.append({"model":model,"severity":severity,"beta":beta,"recovery_rate":recovery,
                        "metric":metric,"n_sources":len(vals),"mean":float(vals.mean()),
                        "median":float(np.median(vals)),"sd":float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                        "p95":float(np.quantile(vals,.95))})
    write_csv(outdir/"parameter_sensitivity_summary.csv",summary)
    return {"source_mode":source_mode,"source_count":int(len(sources)),
            "stratified_per_bin":int(stratified_per_bin),"parameter_combinations":len(combos),
            "model_count":len(MODEL_NAMES),"scenario_count":total}
