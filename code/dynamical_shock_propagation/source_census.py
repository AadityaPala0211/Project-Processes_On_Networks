"""Baseline source-shock census for every eligible source on the full network."""
from __future__ import annotations
import json
import numpy as np
from .config import MODEL_NAMES, Parameters
from .metrics import write_csv, summarize_model, rank_correlations
from .simulator import simulate_batch, prepare_model3_edges
from .structural_metrics import load_structural_metrics

STRUCTURAL_KEYS=("out_strength_million_current_usd","in_strength_million_current_usd",
                 "out_degree","in_degree","weighted_pagerank",
                 "weighted_eigenvector_centrality","sampled_directed_betweenness","k_core")

def eligible_sources(W,output):
    out=np.asarray(W.sum(axis=1)).ravel()
    inc=np.asarray(W.sum(axis=0)).ravel()
    return np.flatnonzero((out+inc>0)|(np.asarray(output)>0))

def select_sources(eligible,mode,top_k,structural,seed):
    if mode=="all": return eligible
    scores=np.asarray([float(r.get("out_strength_million_current_usd",0) or 0) for r in structural])
    if mode=="top-k":
        return eligible[np.argsort(scores[eligible])[::-1][:min(top_k,len(eligible))]]
    if mode=="stratified":
        rng=np.random.default_rng(seed); chosen=[]
        cuts=np.unique(np.quantile(scores[eligible],np.linspace(0,1,11)))
        for low,high in zip(cuts[:-1],cuts[1:]):
            pool=eligible[(scores[eligible]>=low)&(scores[eligible]<=high)]
            if pool.size: chosen.append(int(rng.choice(pool)))
        if not chosen and eligible.size: chosen=[int(rng.choice(eligible))]
        return np.asarray(sorted(set(chosen)),dtype=int)
    raise ValueError("source mode must be all, top-k, or stratified")

def run_source_census(P,nodes,output,W,structural_dir,outdir,params=Parameters(),
                      source_mode="all",top_k=100,seed=2022,batch_size=16,edge_chunk_size=500_000):
    if batch_size<1 or edge_chunk_size<1: raise ValueError("batch sizes must be positive")
    structural=load_structural_metrics(structural_dir,nodes,W)
    eligible=eligible_sources(W,output)
    if source_mode=="top-k" and top_k<1: raise ValueError("top-k must be >= 1")
    sources=select_sources(eligible,source_mode,top_k,structural,seed)
    if not len(sources): raise ValueError("No eligible source nodes found")
    # Cache sparse edge indices for Model 3; all models use the same P and baseline parameters.
    edge_arrays=prepare_model3_edges(P)
    rows=[]
    for model in MODEL_NAMES:
        print(f"[source census] {model}: {len(sources):,} source nodes",flush=True)
        for start in range(0,len(sources),batch_size):
            group=sources[start:start+batch_size]
            batch_rows,_=simulate_batch(P,output,group,model,
                severity=params.initial_disruption,beta=params.beta,recovery_rate=params.recovery_rate,
                source_hold_steps=params.source_hold_steps,max_steps=params.max_steps,
                tolerance=params.convergence_tolerance,affected_threshold=params.affected_threshold,
                batch_edge_arrays=edge_arrays,edge_chunk_size=edge_chunk_size)
            for row in batch_rows:
                i=row["source_index"]; row["source_node"]=nodes[i]
                for key in STRUCTURAL_KEYS:
                    value=structural[i].get(key,"")
                    try: row["structural_"+key]=float(value) if value!="" else ""
                    except (ValueError,TypeError): row["structural_"+key]=""
            rows.extend(batch_rows)
            if start+len(group)==len(sources) or (start//batch_size)%10==0:
                print(f"  processed {min(start+len(group),len(sources))}/{len(sources)} source scenarios",flush=True)
    write_csv(outdir/"source_census.csv",rows)
    (outdir/"source_census_summary.json").write_text(json.dumps(summarize_model(rows),indent=2),encoding="utf-8")
    structural_for_corr=[{"node":nodes[i],**structural[i]} for i in range(len(nodes))]
    write_csv(outdir/"dynamic_vs_structural_rank_correlations.csv",
              rank_correlations(rows,structural_for_corr,STRUCTURAL_KEYS))
    for model in MODEL_NAMES:
        subset=[r for r in rows if r["model"]==model]
        for metric in ("peak_output_weighted_disruption","cumulative_disruption_share_steps","final_output_weighted_disruption"):
            top=sorted(subset,key=lambda r:float(r[metric]),reverse=True)[:50]
            write_csv(outdir/f"top50_{model}_{metric}.csv",top)
    metadata={"source_mode":source_mode,"eligible_source_count":int(len(eligible)),
              "simulated_source_count":int(len(sources)),"model_count":len(MODEL_NAMES),
              "batch_size":batch_size,"edge_chunk_size":edge_chunk_size,
              "parameters":params.__dict__}
    (outdir/"source_census_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return rows
