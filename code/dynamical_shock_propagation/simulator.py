"""Batched simulations for 4,050-node source censuses."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .models import UPDATES
from .models.model3_mmca_sis import update_batch as mmca_update_batch

def prepare_model3_edges(P):
    coo=P.tocoo(copy=False)
    return (coo.row.astype(np.int32,copy=False),coo.col.astype(np.int32,copy=False),coo.data.astype(np.float64,copy=False))

def simulate_batch(P, output_weights, sources, model, severity=0.20, beta=0.30,
                   recovery_rate=0.10, source_hold_steps=3, max_steps=50,
                   tolerance=1e-7, affected_threshold=0.01, batch_edge_arrays=None,
                   edge_chunk_size=500_000, keep_trajectories=False):
    """Run independent source shocks in parallel columns of a dense state batch.

    The network remains the full sparse 4,050x4,050 matrix; only source scenarios
    are batched. Converged columns are frozen but retained through max_steps so
    cumulative impact has a common horizon across sources.
    """
    if not 0 <= severity <= 1: raise ValueError("severity must be in [0,1]")
    if beta < 0: raise ValueError("beta must be nonnegative")
    if not 0 < recovery_rate < 1: raise ValueError("recovery_rate must be in (0,1)")
    if source_hold_steps < 0 or max_steps < 1: raise ValueError("invalid hold/max steps")
    sources=np.asarray(sources,dtype=np.int64)
    n=P.shape[0]; b=sources.size
    if b == 0: return [], []
    if np.any((sources<0)|(sources>=n)): raise IndexError("Source index outside network")
    d=np.zeros((n,b),dtype=np.float64)
    d[sources,np.arange(b)]=severity
    weights=np.maximum(np.asarray(output_weights,dtype=float),0.0)
    weights=weights/weights.sum() if weights.sum()>0 else np.full(n,1.0/n)
    current=np.asarray(weights @ d).ravel()
    peak=current.copy(); peak_step=np.zeros(b,dtype=int); cumulative=current.copy()
    affected=(d>=affected_threshold).sum(axis=0).astype(int)
    max_affected=affected.copy(); final_affected=affected.copy()
    converged=np.zeros(b,dtype=bool); actual_steps=np.full(b,max_steps,dtype=int)
    trajectories=[]
    if keep_trajectories:
        trajectories.append({"step":0,"impact":current.copy(),"affected":affected.copy(),"states":d.copy()})
    if model=="model3_mmca_sis" and batch_edge_arrays is None:
        batch_edge_arrays=prepare_model3_edges(P)
    for step in range(1,max_steps+1):
        previous=d
        if model=="model3_mmca_sis":
            next_d=mmca_update_batch(previous,P,beta,recovery_rate,
                                     edge_arrays=batch_edge_arrays,edge_chunk_size=edge_chunk_size)
        else:
            next_d=UPDATES[model](previous,P,beta,recovery_rate)
        if step<=source_hold_steps:
            next_d[sources,np.arange(b)]=severity
        # Keep a converged scenario at its stationary state for the shared horizon.
        next_d[:,converged]=previous[:,converged]
        delta=np.max(np.abs(next_d-previous),axis=0)
        newly=(~converged)&(step>source_hold_steps)&(delta<tolerance)
        actual_steps[newly]=step
        converged |= newly
        d=np.clip(next_d,0.0,1.0)
        current=np.asarray(weights @ d).ravel()
        bigger=current>peak
        peak[bigger]=current[bigger]; peak_step[bigger]=step
        affected=(d>=affected_threshold).sum(axis=0).astype(int)
        max_affected=np.maximum(max_affected,affected); final_affected=affected
        cumulative += current
        if keep_trajectories:
            trajectories.append({"step":step,"impact":current.copy(),"affected":affected.copy(),"states":d.copy()})
    rows=[]
    final=np.asarray(weights @ d).ravel()
    for j,source in enumerate(sources):
        rows.append({
            "source_index":int(source),"model":model,"initial_disruption":float(severity),
            "beta":float(beta),"recovery_rate":float(recovery_rate),"source_hold_steps":int(source_hold_steps),
            "max_steps":int(max_steps),"actual_steps":int(actual_steps[j]),"converged":bool(converged[j]),
            "peak_output_weighted_disruption":float(peak[j]),"peak_step":int(peak_step[j]),
            "cumulative_disruption_share_steps":float(cumulative[j]),
            "final_output_weighted_disruption":float(final[j]),
            "max_affected_nodes":int(max_affected[j]),"final_affected_nodes":int(final_affected[j]),
            "final_mean_node_disruption":float(d[:,j].mean()),
        })
    return rows,trajectories

def simulate_source(P, output_weights, source, model, **kwargs):
    rows,trajectory=simulate_batch(P,output_weights,[source],model,keep_trajectories=kwargs.pop("keep_trajectories",False),**kwargs)
    path=[]
    for item in trajectory:
        path.append({"step":item["step"],"output_weighted_disruption":float(item["impact"][0]),
                     "affected_nodes":int(item["affected"][0]),"states":item["states"][:,0].copy()})
    return rows[0],path
