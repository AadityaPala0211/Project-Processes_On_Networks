"""Batched equal-budget interventions on the full detailed network.

Intervention sweeps support all eligible source nodes, top-K, or stratified source
samples. Every scenario always evolves on the full 4,050-node dependency matrix.
"""
from __future__ import annotations
import numpy as np
from .config import MODEL_NAMES, Parameters
from .metrics import write_csv
from .simulator import prepare_model3_edges
from .models import UPDATES
from .models.model3_mmca_sis import update_batch as mmca_update_batch
from .structural_metrics import load_structural_metrics
from .source_census import eligible_sources

STRATEGIES=("none","out_strength","pagerank","betweenness","k_core","random")
SCORE_FIELDS={"out_strength":"out_strength_million_current_usd","pagerank":"weighted_pagerank",
              "betweenness":"sampled_directed_betweenness","k_core":"k_core"}

def choose_intervention_sources(eligible,mode,top_k,per_bin,structural,seed):
    if mode=="all": return eligible
    scores=np.asarray([float(r.get("out_strength_million_current_usd",0) or 0) for r in structural])
    if mode=="top-k":
        return eligible[np.argsort(scores[eligible])[::-1][:min(max(top_k,1),len(eligible))]]
    if mode=="stratified":
        rng=np.random.default_rng(seed); chosen=[]
        cuts=np.unique(np.quantile(np.log1p(scores[eligible]),np.linspace(0,1,11)))
        log_scores=np.log1p(scores)
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            pool=eligible[(log_scores[eligible]>=lo)&(log_scores[eligible]<=hi)]
            if pool.size: chosen.extend(rng.choice(pool,size=min(max(per_bin,1),len(pool)),replace=False).astype(int).tolist())
        if not chosen and len(eligible): chosen=[int(rng.choice(eligible))]
        return np.asarray(sorted(set(chosen)),dtype=int)
    raise ValueError("intervention source mode must be all, top-k, or stratified")

def make_protection_sets(nodes,structural,sources,budget,reps,seed):
    scores={key:np.asarray([float(r.get(field,0) or 0) for r in structural])
            for key,field in SCORE_FIELDS.items()}
    rng=np.random.default_rng(seed)
    # One deterministic realization per structural strategy/source; only random
    # protection is replicated, since model trajectories are otherwise deterministic.
    out={}
    for source in sources:
        allowed=np.asarray([i for i in range(len(nodes)) if i!=int(source)],dtype=int)
        out[(int(source),"none",0)]=np.asarray([],dtype=int)
        for strategy,score in scores.items():
            order=allowed[np.argsort(score[allowed])[::-1]]
            out[(int(source),strategy,0)]=order[:min(budget,len(order))]
        for rep in range(reps):
            out[(int(source),"random",rep)]=rng.choice(allowed,size=min(budget,len(allowed)),replace=False)
    return out

def simulate_intervention_batch(P,output,sources,model,params,protected_masks,protection_factor,
                                edge_arrays,edge_chunk_size):
    n=P.shape[0]; b=len(sources)
    states=np.zeros((n,b),dtype=np.float64)
    states[np.asarray(sources,dtype=int),np.arange(b)]=params.initial_disruption
    weights=np.maximum(np.asarray(output,dtype=float),0.0)
    weights=weights/weights.sum() if weights.sum()>0 else np.full(n,1.0/n)
    impact=np.asarray(weights@states).ravel()
    peak=impact.copy(); peak_step=np.zeros(b,dtype=int); cumulative=impact.copy()
    affected=(states>=params.affected_threshold).sum(axis=0).astype(int)
    max_affected=affected.copy(); final_affected=affected.copy()
    converged=np.zeros(b,dtype=bool); actual_steps=np.full(b,params.max_steps,dtype=int)
    for step in range(1,params.max_steps+1):
        previous=states
        if model=="model3_mmca_sis":
            proposal=mmca_update_batch(previous,P,params.beta,params.recovery_rate,
                edge_arrays=edge_arrays,edge_chunk_size=edge_chunk_size)
        else:
            proposal=UPDATES[model](previous,P,params.beta,params.recovery_rate)
        baseline=(1.0-params.recovery_rate)*previous
        increment=np.maximum(proposal-baseline,0.0)
        increment[protected_masks] *= protection_factor
        updated=np.clip(baseline+increment,0.0,1.0)
        if step<=params.source_hold_steps:
            updated[np.asarray(sources,dtype=int),np.arange(b)]=params.initial_disruption
        updated[:,converged]=previous[:,converged]
        delta=np.max(np.abs(updated-previous),axis=0)
        newly=(~converged)&(step>params.source_hold_steps)&(delta<params.convergence_tolerance)
        actual_steps[newly]=step; converged|=newly
        states=updated
        impact=np.asarray(weights@states).ravel()
        greater=impact>peak; peak[greater]=impact[greater]; peak_step[greater]=step
        affected=(states>=params.affected_threshold).sum(axis=0).astype(int)
        max_affected=np.maximum(max_affected,affected); final_affected=affected
        cumulative+=impact
    final=np.asarray(weights@states).ravel()
    rows=[]
    for j,source in enumerate(sources):
        rows.append({"source_index":int(source),"model":model,
            "peak_output_weighted_disruption":float(peak[j]),"peak_step":int(peak_step[j]),
            "cumulative_disruption_share_steps":float(cumulative[j]),
            "final_output_weighted_disruption":float(final[j]),
            "max_affected_nodes":int(max_affected[j]),"final_affected_nodes":int(final_affected[j]),
            "converged":bool(converged[j]),"actual_steps":int(actual_steps[j])})
    return rows

def run_interventions(P,nodes,output,W,structural_dir,outdir,params=Parameters(),
                      source_mode="stratified",top_k=10,stratified_per_bin=1,
                      budget_fraction=0.01,protection_factor=0.5,reps=10,seed=2022,
                      batch_size=16,edge_chunk_size=500_000):
    if not 0<=budget_fraction<=1: raise ValueError("budget_fraction must be in [0,1]")
    if not 0<=protection_factor<=1: raise ValueError("protection_factor must be in [0,1]")
    if reps<1 or batch_size<1: raise ValueError("reps and batch_size must be positive")
    structural=load_structural_metrics(structural_dir,nodes,W)
    eligible=eligible_sources(W,output)
    sources=choose_intervention_sources(eligible,source_mode,top_k,stratified_per_bin,structural,seed)
    if not len(sources): raise ValueError("No eligible source nodes selected for intervention")
    budget=max(1,int(round(budget_fraction*len(nodes))))
    sets=make_protection_sets(nodes,structural,sources,budget,reps,seed)
    edge_arrays=prepare_model3_edges(P); rows=[]
    # Group by strategy and replicate so each source's intervention can be run as
    # a column in the same sparse-matrix × dense-state batch.
    jobs=[(strategy,rep) for strategy in STRATEGIES
          for rep in (range(reps) if strategy=="random" else (0,))]
    for model in MODEL_NAMES:
        for strategy,rep in jobs:
            for start in range(0,len(sources),batch_size):
                source_batch=sources[start:start+batch_size]
                masks=np.zeros((P.shape[0],len(source_batch)),dtype=bool)
                for col,source in enumerate(source_batch):
                    masks[sets[(int(source),strategy,rep)],col]=True
                batch_rows=simulate_intervention_batch(P,output,source_batch,model,params,masks,
                    protection_factor,edge_arrays,edge_chunk_size)
                for row in batch_rows:
                    row.update({"source_node":nodes[row["source_index"]],"model":model,
                        "strategy":strategy,"replicate":int(rep),"budget_nodes":int(len(sets[(row["source_index"],strategy,rep)])),
                        "budget_fraction":budget_fraction,"protection_factor":protection_factor})
                    rows.append(row)
        print(f"[interventions] {model} complete for {len(sources)} source nodes",flush=True)
    write_csv(outdir/"intervention_runs.csv",rows)
    baseline={(r["model"],r["source_index"]):float(r["cumulative_disruption_share_steps"])
              for r in rows if r["strategy"]=="none"}
    summaries=[]
    for model in MODEL_NAMES:
        for strategy in STRATEGIES:
            sub=[r for r in rows if r["model"]==model and r["strategy"]==strategy]
            deltas=[float(r["cumulative_disruption_share_steps"])-baseline[(model,r["source_index"])] for r in sub]
            if deltas:
                summaries.append({"model":model,"strategy":strategy,"scenario_count":len(sub),
                    "source_count":len({r["source_index"] for r in sub}),
                    "mean_cumulative_change_vs_none":float(np.mean(deltas)),
                    "median_cumulative_change_vs_none":float(np.median(deltas)),
                    "mean_peak_disruption":float(np.mean([float(r["peak_output_weighted_disruption"]) for r in sub]))})
    write_csv(outdir/"intervention_summary.csv",summaries)
    return {"source_mode":source_mode,"source_count":int(len(sources)),"top_k":top_k,
        "stratified_per_bin":stratified_per_bin,"budget_nodes":budget,"budget_fraction":budget_fraction,
        "protection_factor":protection_factor,"random_control_replicates":reps,
        "model_batch_size":batch_size,
        "protection_definition":"Protected nodes receive protection_factor times their modeled nonnegative new-disruption increment; recovery is unchanged.",
        "note":"The full network is used for every scenario. source_mode selects shock locations only."}
