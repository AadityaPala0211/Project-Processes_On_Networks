"""Equal-budget protection interventions across the full detailed network."""
from __future__ import annotations
import numpy as np
from .config import MODEL_NAMES, Parameters
from .metrics import write_csv
from .simulator import prepare_model3_edges
from .models import UPDATES
from .models.model3_mmca_sis import update_batch as mmca_update_batch
from .structural_metrics import load_structural_metrics
from .source_census import eligible_sources

def protected_simulation(P,output,source,model,params,protected_indices,protection_factor=0.5,edge_arrays=None):
    if not 0<=protection_factor<=1: raise ValueError("protection_factor must be in [0,1]")
    n=P.shape[0]; d=np.zeros(n); d[source]=params.initial_disruption
    weights=np.maximum(np.asarray(output,dtype=float),0.0)
    weights=weights/weights.sum() if weights.sum()>0 else np.full(n,1.0/n)
    protected=np.zeros(n,dtype=bool); protected[np.asarray(protected_indices,dtype=int)]=True
    peak=cumulative=float(weights@d); peak_step=0
    max_affected=int(np.count_nonzero(d>=params.affected_threshold)); converged=False; actual_steps=params.max_steps
    for step in range(1,params.max_steps+1):
        old=d
        if model=="model3_mmca_sis":
            proposal=mmca_update_batch(old[:,None],P,params.beta,params.recovery_rate,
                                       edge_arrays=edge_arrays)[:,0]
        else:
            proposal=UPDATES[model](old,P,params.beta,params.recovery_rate)
        # Protection attenuates new modeled disruption increments only, not recovery.
        baseline=(1.0-params.recovery_rate)*old
        increment=np.maximum(proposal-baseline,0.0)
        increment[protected]*=protection_factor
        new=np.clip(baseline+increment,0.0,1.0)
        if step<=params.source_hold_steps: new[source]=params.initial_disruption
        delta=float(np.max(np.abs(new-old))); d=new
        impact=float(weights@d); cumulative+=impact
        if impact>peak: peak,peak_step=impact,step
        max_affected=max(max_affected,int(np.count_nonzero(d>=params.affected_threshold)))
        if step>params.source_hold_steps and delta<params.convergence_tolerance:
            converged=True; actual_steps=step
            # For equal-horizon cumulative metrics, hold the stationary impact for
            # the remaining rounds rather than silently shortening this scenario.
            cumulative += impact*(params.max_steps-step)
            break
    return {"peak_output_weighted_disruption":peak,"peak_step":peak_step,
            "cumulative_disruption_share_steps":cumulative,
            "final_output_weighted_disruption":float(weights@d),
            "max_affected_nodes":max_affected,"converged":converged,"actual_steps":actual_steps}

def run_interventions(P,nodes,output,W,structural_dir,outdir,params=Parameters(),
                      budget_fraction=0.01,protection_factor=0.5,reps=20,seed=2022):
    structural=load_structural_metrics(structural_dir,nodes,W)
    eligible=eligible_sources(W,output)
    out_strength=np.asarray([float(r.get("out_strength_million_current_usd",0) or 0) for r in structural])
    candidates=eligible if len(eligible) else np.arange(len(nodes))
    source=int(candidates[np.argmax(out_strength[candidates])])
    budget=max(1,int(round(budget_fraction*len(nodes))))
    allowed=np.asarray([i for i in range(len(nodes)) if i!=source],dtype=int)
    score_keys={"out_strength":"out_strength_million_current_usd","pagerank":"weighted_pagerank",
                "betweenness":"sampled_directed_betweenness","k_core":"k_core"}
    strategies={"none":np.asarray([],dtype=int)}
    for label,key in score_keys.items():
        scores=np.asarray([float(r.get(key,0) or 0) for r in structural])
        order=np.argsort(scores[allowed])[::-1]
        strategies[label]=allowed[order[:min(budget,len(allowed))]]
    rows=[]; rng=np.random.default_rng(seed); edge_arrays=prepare_model3_edges(P)
    for model in MODEL_NAMES:
        for rep in range(reps):
            local=dict(strategies)
            local["random"]=rng.choice(allowed,size=min(budget,len(allowed)),replace=False)
            for strategy,protected in local.items():
                result=protected_simulation(P,output,source,model,params,protected,protection_factor,edge_arrays)
                rows.append({"model":model,"replicate":rep,"strategy":strategy,"source_node":nodes[source],
                    "budget_nodes":int(len(protected)),"budget_fraction":budget_fraction,
                    "protection_factor":protection_factor,**result})
        print(f"[interventions] {model} completed ({reps} replicates/strategy)",flush=True)
    write_csv(outdir/"intervention_runs.csv",rows)
    summaries=[]
    for model in MODEL_NAMES:
        sub=[r for r in rows if r["model"]==model]
        base={int(r["replicate"]):float(r["cumulative_disruption_share_steps"]) for r in sub if r["strategy"]=="none"}
        for strategy in sorted({r["strategy"] for r in sub}):
            comparisons=[]
            for r in sub:
                if r["strategy"]==strategy:
                    comparisons.append(float(r["cumulative_disruption_share_steps"])-base[int(r["replicate"])])
            if comparisons:
                summaries.append({"model":model,"strategy":strategy,"replicates":len(comparisons),
                    "mean_cumulative_change_vs_none":float(np.mean(comparisons)),
                    "median_cumulative_change_vs_none":float(np.median(comparisons)),
                    "mean_peak_disruption":float(np.mean([float(r["peak_output_weighted_disruption"]) for r in sub if r["strategy"]==strategy]))})
    write_csv(outdir/"intervention_summary.csv",summaries)
    return {"source_node":nodes[source],"budget_nodes":budget,"budget_fraction":budget_fraction,
            "protection_factor":protection_factor,"replicates":reps,
            "definition":"protected nodes receive protection_factor times each model's nonnegative new-disruption increment; recovery is unchanged."}
