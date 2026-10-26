#!/usr/bin/env python3
"""Reproducible analysis pipeline for OECD regular ICIO 2022_SML.csv.

This script intentionally rebuilds the analysis from the raw table. It does NOT use
any pre-existing screenshot or manually typed chart values.

Usage:
  python full_supply_chain_analysis.py /path/to/2022_SML.csv --out results_2022

Outputs include structural metrics, top-node tables, plots, weighted-SIS experiments,
threshold-cascade sensitivity, intervention comparisons, and a Markdown results log.

Interpretation:
  * Rows of the intermediate block are supplying country-industries i.
  * Columns are purchasing country-industries j; Z[i,j] is the input flow i -> j.
  * Values are nominal/current USD millions, according to the ICIO README.
  * Dynamic results are counterfactual simulations, not observed losses or forecasts.
"""
from __future__ import annotations
import argparse, csv, json, math, re, sys, time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NODE_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9_]+$")
SECTOR_GROUPS = [
    "Agriculture/forestry/fishing", "Mining/quarrying", "Food/textiles/wood/paper",
    "Chemicals/petroleum/minerals", "Metals/machinery/electronics/transport",
    "Utilities", "Construction", "Trade", "Transport/accommodation",
    "Information/communications", "Finance/real estate", "Professional/admin services",
    "Public/social/other services"
]

def industry_group(code: str) -> str:
    c = code.strip().upper()
    if c in {"A01", "A02", "A03"}: return SECTOR_GROUPS[0]
    if c.startswith("B"): return SECTOR_GROUPS[1]
    if c in {"C10T12", "C13T15", "C16", "C17_18"}: return SECTOR_GROUPS[2]
    if c in {"C19", "C20", "C21", "C22", "C23"}: return SECTOR_GROUPS[3]
    if c.startswith("C"): return SECTOR_GROUPS[4]
    if c in {"D", "E"} or c.startswith("D") or c.startswith("E"): return SECTOR_GROUPS[5]
    if c == "F" or c.startswith("F"): return SECTOR_GROUPS[6]
    if c in {"G45", "G46", "G47"} or c.startswith("G"): return SECTOR_GROUPS[7]
    if c.startswith("H") or c == "I": return SECTOR_GROUPS[8]
    if c.startswith("J"): return SECTOR_GROUPS[9]
    if c in {"K", "L"}: return SECTOR_GROUPS[10]
    if c in {"M", "N"}: return SECTOR_GROUPS[11]
    if c in {"O", "P", "Q", "R", "S", "T", "U"}: return SECTOR_GROUPS[12]
    raise ValueError(f"Industry code {code!r} is not mapped to an aggregate group. Update industry_group().")

def detect_nodes(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        hpos = {}
        for i, lab in enumerate(header):
            s = lab.strip()
            if s and s not in hpos: hpos[s] = i
        nodes=[]
        for row in reader:
            if not row: break
            label=row[0].strip()
            if NODE_RE.fullmatch(label) and label in hpos:
                nodes.append(label)
            else:
                break
    if not nodes: raise ValueError("No country-industry rows found; expected labels such as USA_C20.")
    if len(set(nodes)) != len(nodes): raise ValueError("Duplicate country-industry row labels detected.")
    # Build numeric positions after removing the first CSV row-label column.
    cols=np.asarray([hpos[n]-1 for n in nodes], dtype=np.int64)
    if np.any(cols < 0): raise ValueError("Unexpected CSV header: row labels overlap numeric columns.")
    out_col = hpos.get("OUT", None)
    out_col = (out_col-1) if out_col is not None else None
    return header,nodes,cols,out_col

def load_icio(path: Path):
    header,nodes,colpos,outpos=detect_nodes(path)
    n=len(nodes)
    Z=np.zeros((n,n), dtype=np.float64)
    output=np.full(n, np.nan, dtype=np.float64)
    node_rows=[]
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader=csv.reader(f); next(reader)
        for i in range(n):
            try: row=next(reader)
            except StopIteration: raise ValueError(f"CSV ended after {i} node rows; expected {n}.")
            label=row[0].strip()
            if label != nodes[i]: raise ValueError(f"Row alignment error at {i}: expected {nodes[i]}, found {label}.")
            vals=np.asarray(row[1:], dtype=np.float64)
            if colpos.max() >= len(vals): raise ValueError(f"Row {label} is shorter than expected.")
            Z[i,:]=vals[colpos]
            if outpos is not None and outpos < len(vals): output[i]=vals[outpos]
            node_rows.append(label)
    # If a distinct OUT column is absent, recover output from the special OUT row.
    if np.any(~np.isfinite(output)):
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader=csv.reader(f); next(reader)
            for row in reader:
                if row and row[0].strip()=="OUT":
                    vals=np.asarray(row[1:], dtype=np.float64)
                    output=vals[colpos]
                    break
    if np.any(~np.isfinite(Z)): raise ValueError("Intermediate-flow block contains NaN/Inf.")
    return header,nodes,Z,output

def gini(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if x.size==0 or np.all(x==0): return 0.0
    x=np.sort(np.maximum(x,0)); n=x.size
    return float((2*np.sum((np.arange(1,n+1))*x)/(n*np.sum(x)))-(n+1)/n)

def weighted_pagerank(W, alpha=.85, max_iter=300, tol=1e-11):
    n=W.shape[0]
    out=np.asarray(W.sum(axis=1)).ravel()
    inv=np.zeros(n); ok=out>0; inv[ok]=1/out[ok]
    T=sparse.diags(inv) @ W
    p=np.full(n,1/n)
    dangling=~ok
    for it in range(max_iter):
        pnew=(1-alpha)/n + alpha*(T.T@p + p[dangling].sum()/n)
        pnew=np.asarray(pnew).ravel(); pnew/=pnew.sum()
        if np.linalg.norm(pnew-p,1)<tol: return pnew,it+1
        p=pnew
    return p,max_iter

INDUSTRY_LABELS = {
    "A01":"Agriculture and hunting", "A02":"Forestry and logging", "A03":"Fishing and aquaculture",
    "B06":"Crude petroleum and natural gas extraction", "C10T12":"Food, beverages and tobacco manufacturing",
    "C19":"Coke and refined petroleum manufacturing", "C20":"Chemicals manufacturing",
    "C23":"Other non-metallic mineral products", "C24A":"Basic iron and steel manufacturing",
    "C25":"Fabricated metal products", "C26":"Computer, electronic and optical products",
    "C28":"Machinery and equipment manufacturing", "D":"Electricity, gas, steam and air conditioning",
    "F":"Construction", "G":"Wholesale and retail trade; motor vehicle repair",
    "H49":"Land transport and pipelines", "I":"Accommodation and food services",
    "J62_63":"Computer programming and information services", "K":"Financial and insurance activities",
    "L":"Real estate activities", "M":"Professional, scientific and technical activities",
    "N":"Administrative and support services", "O":"Public administration and defence",
    "Q":"Human health and social work activities",
}
COUNTRY_LABELS = {"USA":"United States","CHN":"China","ROW":"Rest of World","IND":"India"}

def top_rows(nodes, x, topn=20, scale=1.0):
    ids=np.argsort(np.asarray(x))[::-1][:min(topn,len(nodes))]
    out=[]
    for k,i in enumerate(ids):
        node=nodes[int(i)]; country,industry=node.split("_",1)
        out.append({"rank":k+1,"node":node,"country_name":COUNTRY_LABELS.get(country,country),
                    "industry_name":INDUSTRY_LABELS.get(industry,industry),"value":float(x[i]/scale)})
    return out

def corr(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<3 or np.std(x[good])==0 or np.std(y[good])==0: return {"rho":None,"pvalue":None}
    r=spearmanr(x[good],y[good])
    return {"rho":float(r.statistic),"pvalue":float(r.pvalue)}

def spectral_radius(W):
    if W.shape[0] < 3 or W.nnz == 0: return 0.0
    try:
        val=eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=15000, tol=1e-8)[0]
        return float(abs(val))
    except Exception:
        # Fallback power iteration; estimate spectral radius by norm ratios.
        n=W.shape[0]; x=np.ones(n)/np.sqrt(n); prev=0.0
        for _ in range(5000):
            y=W@x; norm=np.linalg.norm(y)
            if norm==0: return 0.0
            x=y/norm
            if abs(norm-prev)<1e-10*max(1,norm): return float(norm)
            prev=norm
        return float(prev)

def run_sis(W, out, source, beta, gamma, steps, reps, rng, protected=None, protection_factor=1.0):
    """Stochastic SIS-like process. W[i,j] is transmission influence i -> j.

    The hazard for a susceptible j is 1-exp(-beta * sum_i W[i,j] I_i).
    Infected nodes recover with probability gamma. Optional protection reduces
    transmission from selected supplier nodes by scaling their outgoing rows.
    """
    Wrun=W
    if protected is not None and protection_factor != 1.0:
        factors=np.ones(W.shape[0]); factors[np.asarray(protected,dtype=int)]=protection_factor
        Wrun=sparse.diags(factors) @ W
    WT=Wrun.T.tocsr()
    output=np.maximum(np.asarray(out,dtype=float),0)
    if output.sum() <= 0: output=np.ones(W.shape[0])
    output=output/output.sum()
    records=[]
    for rep in range(reps):
        infected=np.zeros(W.shape[0],dtype=bool); infected[int(source)]=True
        series=[]
        node_series=[]
        for t in range(steps):
            series.append(float(output@infected.astype(float)))
            node_series.append(float(infected.mean()))
            pressure=np.asarray(WT@infected.astype(float)).ravel()
            pnew=-np.expm1(-beta*np.maximum(pressure,0))
            fresh=(~infected) & (rng.random(W.shape[0]) < np.clip(pnew,0,1))
            recovered=infected & (rng.random(W.shape[0]) < gamma)
            nxt=infected.copy(); nxt[recovered]=False; nxt[fresh]=True
            infected=nxt
            if not infected.any():
                # Fill remaining trajectory with zero; extinction is absorbing only
                # until a new seed, and there is no external reseeding here.
                series.extend([0.0]*(steps-t-1)); node_series.extend([0.0]*(steps-t-1)); break
        a=np.asarray(series,dtype=float); an=np.asarray(node_series,dtype=float)
        nz=np.where(a>0)[0]
        records.append({"peak_output_weighted_disruption":float(a.max(initial=0)),
                        "cumulative_output_weighted_disruption":float(a.sum()),
                        "mean_output_weighted_disruption":float(a.mean()),
                        "peak_node_fraction":float(an.max(initial=0)),
                        "final_output_weighted_disruption":float(a[-1]),
                        "extinction_step":int(nz[-1]+1) if len(nz) and a[-1]==0 else (steps if len(nz) else 0),
                        "replicate":rep})
    return records

def threshold_cascade(D, source, theta, max_steps=100):
    """Deterministic threshold cascade; D[i,j] is supplier i's share of buyer j's inputs."""
    n=D.shape[0]; failed=np.zeros(n,dtype=bool); failed[int(source)]=True
    for t in range(max_steps):
        exposure=np.asarray(D.T@failed.astype(float)).ravel()
        new=(exposure>=theta) & (~failed)
        if not new.any(): break
        failed |= new
    return {"cascade_nodes":int(failed.sum()),"cascade_fraction":float(failed.mean()),"steps":t+1 if max_steps else 0,"failed":failed}

def make_aggregate(Z, nodes, output):
    countries=[]; activities=[]
    for n in nodes:
        c,act=n.split("_",1); countries.append(c); activities.append(act)
    unique_countries=list(dict.fromkeys(countries)); cidx={c:i for i,c in enumerate(unique_countries)}
    groups=[industry_group(a) for a in activities]
    gidx={g:i for i,g in enumerate(SECTOR_GROUPS)}
    labels=[f"{c}_{g.replace('/','-').replace(' ','_')}" for c in unique_countries for g in SECTOR_GROUPS]
    mapidx=np.asarray([cidx[c]*len(SECTOR_GROUPS)+gidx[g] for c,g in zip(countries,groups)],dtype=int)
    S=sparse.csr_matrix((np.ones(len(nodes)),(np.arange(len(nodes)),mapidx)),shape=(len(nodes),len(labels)))
    Zsp=sparse.csr_matrix(np.maximum(Z,0))
    Za=(S.T @ Zsp @ S).tocsr(); Za.eliminate_zeros()
    Xa=np.asarray(S.T@np.maximum(np.nan_to_num(output,nan=0.0),0)).ravel()
    active=np.asarray(Za.sum(axis=0)).ravel() > 0
    return unique_countries, labels, mapidx, Za, Xa, S

def save_bar(nodes, values, title, xlabel, filename, scale=1.0, topn=20):
    ids=np.argsort(values)[::-1][:min(topn,len(nodes))][::-1]
    fig,ax=plt.subplots(figsize=(10,6))
    ax.barh(np.asarray(nodes)[ids],np.asarray(values)[ids]/scale)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(axis="x",alpha=.25); fig.tight_layout()
    fig.savefig(filename,dpi=180); plt.close(fig)

def main():
    p=argparse.ArgumentParser(description="Full structural and dynamical analysis of OECD ICIO 2022_SML.csv")
    p.add_argument("csv", type=Path, help="Path to OECD 2022_SML.csv")
    p.add_argument("--out", type=Path, default=Path("full_analysis_results"))
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--reps", type=int, default=10, help="Monte Carlo repetitions per SIS scenario")
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--random-seeds", type=int, default=10)
    p.add_argument("--topn", type=int, default=20)
    args=p.parse_args()
    outdir=args.out.resolve(); outdir.mkdir(parents=True,exist_ok=True)
    start=time.time(); rng=np.random.default_rng(args.seed)
    print("Reading raw ICIO table...")
    header,nodes,Z,X=load_icio(args.csv.resolve())
    n=len(nodes)
    neg=int(np.count_nonzero(Z<0)); zpos=np.maximum(Z,0)
    self_flows=np.diag(zpos).copy(); Zgraph=zpos.copy(); np.fill_diagonal(Zgraph,0)
    W=sparse.csr_matrix(Zgraph); W.eliminate_zeros()
    A=(W>0).astype(np.int8).tocsr(); A.eliminate_zeros()
    m=int(A.nnz); max_m=n*(n-1); density=m/max_m if max_m else 0
    outdeg=np.asarray(A.sum(axis=1)).ravel().astype(int); indeg=np.asarray(A.sum(axis=0)).ravel().astype(int)
    # Robustness: positive-but-tiny ICIO flows can make binary degree look saturated.
    # Thresholds are in million current USD; 1.0 means a one-million-dollar flow.
    edge_threshold_rows=[]
    for th in [0.0,0.01,0.1,1.0,10.0,100.0]:
        mask=(Zgraph>0) if th==0 else (Zgraph>=th)
        At=sparse.csr_matrix(mask.astype(np.uint8)); At.setdiag(0); At.eliminate_zeros()
        mt=int(At.nnz); dt=mt/max_m if max_m else 0
        odt=np.asarray(At.sum(axis=1)).ravel(); idt=np.asarray(At.sum(axis=0)).ravel()
        wc,wl=connected_components(At,directed=True,connection="weak")
        wsize=np.bincount(wl)
        edge_threshold_rows.append({"min_flow_million_USD":th,"edge_count":mt,"density":float(dt),
            "mean_out_degree":float(odt.mean()),"isolates":int(np.sum((odt==0)&(idt==0)),),
            "weak_component_count":int(wc),"largest_weak_component_nodes":int(wsize.max())})
    outstr=np.asarray(W.sum(axis=1)).ravel(); instr=np.asarray(W.sum(axis=0)).ravel()
    weak_n,weak_lab=connected_components(A,directed=True,connection="weak")
    strong_n,strong_lab=connected_components(A,directed=True,connection="strong")
    weak_sizes=np.bincount(weak_lab); strong_sizes=np.bincount(strong_lab)
    reciprocity=float(A.multiply(A.T).nnz/m) if m else 0.0
    print("Computing weighted PageRank...")
    pr,pr_iter=weighted_pagerank(W)
    spears={"out_degree_vs_out_strength":corr(outdeg,outstr),"in_degree_vs_in_strength":corr(indeg,instr),
            "out_strength_vs_page_rank":corr(outstr,pr),"out_strength_vs_gross_output":corr(outstr,np.nan_to_num(X,nan=0))}
    structural={"source_file":str(args.csv.resolve()),"year":2022,"node_count":n,
        "flow_matrix_shape":[n,n],"flow_values_negative_count":neg,
        "flow_positive_total_million_current_usd":float(zpos.sum()),
        "self_flow_total_million_current_usd":float(self_flows.sum()),
        "self_flow_share":float(self_flows.sum()/zpos.sum()) if zpos.sum() else 0.0,
        "directed_edges_excluding_self_loops":m,"possible_directed_edges_excluding_self_loops":max_m,
        "directed_density":float(density),"reciprocity_fraction_of_edge_directions":reciprocity,
        "weak_component_count":int(weak_n),"largest_weak_component_nodes":int(weak_sizes.max()),
        "weak_second_largest_component_nodes":int(np.sort(weak_sizes)[-2]) if len(weak_sizes)>1 else 0,
        "strong_component_count":int(strong_n),"largest_strong_component_nodes":int(strong_sizes.max()),
        "isolated_nodes":int(np.sum((outdeg==0)&(indeg==0)),),
        "source_only_nodes":int(np.sum((indeg==0)&(outdeg>0)),),
        "sink_only_nodes":int(np.sum((outdeg==0)&(indeg>0)),),
        "total_gross_output_million_current_usd":float(np.maximum(X,0).sum()),
        "off_diagonal_flow_total_million_current_usd":float(outstr.sum()),
        "out_degree":{"mean":float(outdeg.mean()),"median":float(np.median(outdeg)),"p90":float(np.quantile(outdeg,.9)),"max":int(outdeg.max())},
        "in_degree":{"mean":float(indeg.mean()),"median":float(np.median(indeg)),"p90":float(np.quantile(indeg,.9)),"max":int(indeg.max())},
        "out_strength_gini":gini(outstr),"in_strength_gini":gini(instr),"pagerank_iterations":pr_iter,
        "spearman_correlations":spears,
        "top_out_strength":top_rows(nodes,outstr,args.topn,1000),
        "top_in_strength":top_rows(nodes,instr,args.topn,1000),
        "top_out_degree":top_rows(nodes,outdeg,args.topn),"top_in_degree":top_rows(nodes,indeg,args.topn),
        "top_weighted_pagerank":top_rows(nodes,pr,args.topn),
        "top_gross_output":top_rows(nodes,np.maximum(X,0),args.topn,1000),
        "edge_threshold_sensitivity":edge_threshold_rows,
        "units":"Flow and strength values in million current USD. Excludes self-loops for graph degree/strength metrics.",
        "interpretation":"Structural results describe the annual 2022 ICIO table; they are not observed disruption outcomes."}
    (outdir/"structural_summary.json").write_text(json.dumps(structural,indent=2),encoding="utf-8")
    with (outdir/"node_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["node","country","industry_code","out_degree","in_degree","out_strength_million_USD","in_strength_million_USD","weighted_pagerank","gross_output_million_USD"])
        for i,node in enumerate(nodes):
            country,industry=node.split("_",1); w.writerow([node,country,industry,int(outdeg[i]),int(indeg[i]),outstr[i],instr[i],pr[i],X[i]])
    # Structural tables at country and sector level, built from the actual matrix.
    country_out={}; sector_out={}; country_in={}; sector_in={}
    for i,node in enumerate(nodes):
        country,industry=node.split("_",1); group=industry_group(industry)
        country_out[country]=country_out.get(country,0.0)+outstr[i]
        country_in[country]=country_in.get(country,0.0)+instr[i]
        sector_out[group]=sector_out.get(group,0.0)+outstr[i]
        sector_in[group]=sector_in.get(group,0.0)+instr[i]
    for name, data in [("country_flow_summary.csv",[(k,country_out[k],country_in[k]) for k in sorted(country_out)]),
                       ("sector_flow_summary.csv",[(k,sector_out.get(k,0.0),sector_in.get(k,0.0)) for k in SECTOR_GROUPS])]:
        with (outdir/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["country_or_sector_group","out_strength_million_USD","in_strength_million_USD"]); w.writerows(data)
    with (outdir/"edge_threshold_sensitivity.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(edge_threshold_rows[0].keys())); w.writeheader(); w.writerows(edge_threshold_rows)
    save_bar(nodes,outstr,"Top country-industries by out-strength — 2022","Out-strength (billion current USD)",outdir/"top_out_strength.png",1000,args.topn)
    save_bar(nodes,instr,"Top country-industries by in-strength — 2022","In-strength (billion current USD)",outdir/"top_in_strength.png",1000,args.topn)
    save_bar(nodes,pr,"Top country-industries by weighted PageRank — 2022","Weighted PageRank",outdir/"top_weighted_pagerank.png",1,args.topn)
    # Aggregate flow by the 13 broad industries, summing over all origin/destination countries.
    node_groups=[industry_group(node.split("_",1)[1]) for node in nodes]
    gidx={g:i for i,g in enumerate(SECTOR_GROUPS)}
    G=sparse.csr_matrix((np.ones(n),(np.arange(n),[gidx[g] for g in node_groups])),shape=(n,len(SECTOR_GROUPS)))
    Zs_full=sparse.csr_matrix(np.maximum(Z,0))
    sector_flow=(G.T@Zs_full@G).toarray()
    fig,ax=plt.subplots(figsize=(8.5,6.5))
    arr=np.log10(sector_flow/1000.0+1e-3)  # log10 of billions of current USD, with offset for zeros.
    im=ax.imshow(arr,aspect="auto",cmap="viridis")
    short=[g.replace("-"," ") for g in SECTOR_GROUPS]
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short,rotation=75,ha="right",fontsize=7)
    ax.set_yticks(range(len(short))); ax.set_yticklabels(short,fontsize=7)
    ax.set_xlabel("Purchasing sector group"); ax.set_ylabel("Supplying sector group")
    ax.set_title("Aggregated intermediate flows by sector (log scale)")
    cb=fig.colorbar(im,ax=ax); cb.set_label("log10(flow in billion current USD + 0.001)")
    fig.tight_layout(); fig.savefig(outdir/"sector_flow_heatmap.png",dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].hist(outdeg,bins=40); axs[0].set_title("Out-degree distribution"); axs[0].set_xlabel("Distinct buyers supplied"); axs[0].set_ylabel("Node count")
    axs[1].hist(indeg,bins=40); axs[1].set_title("In-degree distribution"); axs[1].set_xlabel("Distinct suppliers used")
    fig.tight_layout(); fig.savefig(outdir/"degree_distributions.png",dpi=180); plt.close(fig)
    # Aggregate flows to country x broad-sector nodes for feasible repeated dynamics.
    print("Aggregating to country x 13 broad sectors for dynamical experiments...")
    countries,agg_nodes,mapidx,Za,Xa,S=make_aggregate(Z,nodes,X)
    na=len(agg_nodes)
    agg_flow_total=float(Za.sum())
    raw_flow_total=float(np.maximum(Z,0).sum())
    agg_output_total=float(np.maximum(Xa,0).sum())
    raw_output_total=float(np.maximum(X,0).sum())
    aggregation_validation={
        "raw_intermediate_flow_total_million_USD":raw_flow_total,
        "aggregated_intermediate_flow_total_million_USD":agg_flow_total,
        "intermediate_flow_relative_error":abs(agg_flow_total-raw_flow_total)/max(abs(raw_flow_total),1e-12),
        "raw_gross_output_total_million_USD":raw_output_total,
        "aggregated_gross_output_total_million_USD":agg_output_total,
        "gross_output_relative_error":abs(agg_output_total-raw_output_total)/max(abs(raw_output_total),1e-12),
    }
    if aggregation_validation["intermediate_flow_relative_error"]>1e-8 or aggregation_validation["gross_output_relative_error"]>1e-8:
        raise ValueError(f"Aggregation conservation check failed: {aggregation_validation}")
    Za_pos=Za.maximum(0); Za_pos.setdiag(0); Za_pos.eliminate_zeros()
    Aagg=(Za_pos>0).astype(float).tocsr(); Aagg.eliminate_zeros()
    outstr_a=np.asarray(Za_pos.sum(axis=1)).ravel(); instr_a=np.asarray(Za_pos.sum(axis=0)).ravel()
    pr_a,_=weighted_pagerank(Za_pos)
    # Input coefficients: Z_ij divided by total output of purchaser j. Used as
    # transmission weights in a weighted SIS benchmark. Self links do not spread shocks.
    Xden=np.maximum(Xa,1e-12)
    Tech=(Za_pos @ sparse.diags(1.0/Xden)).tocsr(); Tech.eliminate_zeros()
    # A fair unweighted baseline: each buyer's incoming neighbours have equal influence.
    indeg_a=np.asarray(Aagg.sum(axis=0)).ravel()
    invind=np.zeros(na); invind[indeg_a>0]=1.0/indeg_a[indeg_a>0]
    U=(Aagg @ sparse.diags(invind)).tocsr()
    rho_t=spectral_radius(Tech.T.tocsr()); rho_u=spectral_radius(U.T.tocsr())
    gamma=0.25
    factors=[0.7,1.0,1.4]
    beta_t=[(gamma/rho_t)*f if rho_t>0 else 0.0 for f in factors]
    beta_u=[(gamma/rho_u)*f if rho_u>0 else 0.0 for f in factors]
    # Source nodes: high out-strength, high PageRank, and a reproducible random sample.
    top_os=int(np.argmax(outstr_a)); top_pr=int(np.argmax(pr_a))
    active_a=np.where((outstr_a+instr_a>0)&(np.maximum(Xa,0)>0))[0]
    if len(active_a)==0: active_a=np.arange(na)
    rand_count=min(args.random_seeds,len(active_a)); random_sources=rng.choice(active_a,size=rand_count,replace=False).tolist()
    source_scenarios=[("top_out_strength",top_os),("top_page_rank",top_pr)]+[("random",int(s)) for s in random_sources]
    sim_rows=[]
    for model,Wmodel,beta_list,rho in [("unweighted_SIS",U,beta_u,rho_u),("weighted_SIS",Tech,beta_t,rho_t)]:
        print(f"Running {model}: rho(W)={rho:.6g}; approximate beta_c={gamma/rho if rho>0 else float('inf'):.6g}")
        for label,source in source_scenarios:
            for mult,beta in zip(factors,beta_list):
                runs=run_sis(Wmodel,Xa,source,beta,gamma,args.steps,args.reps,rng)
                for r in runs:
                    sim_rows.append({"model":model,"source_type":label,"source_node":agg_nodes[source],"beta_over_beta_critical":mult,"beta":beta,"gamma":gamma,**r})
    with (outdir/"sis_simulation_runs.csv").open("w",newline="",encoding="utf-8") as f:
        cols=list(sim_rows[0].keys()) if sim_rows else []
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(sim_rows)
    # Targeted source-impact analysis. The candidate set deliberately combines the top
    # nodes by several structural measures with random active nodes; results are an
    # exploratory, stratified comparison rather than a census of every possible seed.
    outdeg_a=np.asarray(Aagg.sum(axis=1)).ravel()
    in_deg_a=np.asarray(Aagg.sum(axis=0)).ravel()
    candidate_set=set()
    for metric in [outstr_a, instr_a, pr_a, np.maximum(Xa,0), outdeg_a]:
        candidate_set.update(np.argsort(metric)[::-1][:min(20,na)].tolist())
    random_candidate_count=min(20,len(active_a))
    candidate_set.update(rng.choice(active_a,size=random_candidate_count,replace=False).tolist())
    impact_candidates=sorted(candidate_set)
    beta_impact=beta_t[-1] if beta_t else 0.0
    impact_rows=[]
    print(f"Running weighted-SIS source-impact comparison for {len(impact_candidates)} candidate seeds...")
    for source in impact_candidates:
        runs=run_sis(Tech,Xa,source,beta_impact,gamma,args.steps,max(6,args.reps//2),rng)
        impact_rows.append({"source_node":agg_nodes[source],"source_index":source,
            "out_strength_billion_USD":float(outstr_a[source]/1000),
            "in_strength_billion_USD":float(instr_a[source]/1000),
            "out_degree":float(outdeg_a[source]),"in_degree":float(in_deg_a[source]),
            "weighted_pagerank":float(pr_a[source]),"gross_output_billion_USD":float(max(Xa[source],0)/1000),
            "peak_output_weighted_infected_share_mean":float(np.mean([r["peak_output_weighted_disruption"] for r in runs])),
            "cumulative_output_weighted_infected_share_mean":float(np.mean([r["cumulative_output_weighted_disruption"] for r in runs])),
            "final_output_weighted_infected_share_mean":float(np.mean([r["final_output_weighted_disruption"] for r in runs])),
            "replicates":len(runs)})
    with (outdir/"source_impact_ranking.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(impact_rows[0].keys())); w.writeheader(); w.writerows(impact_rows)
    impact_corr={"out_strength_vs_peak_impact":corr([r["out_strength_billion_USD"] for r in impact_rows],[r["peak_output_weighted_infected_share_mean"] for r in impact_rows]),
        "out_strength_vs_cumulative_impact":corr([r["out_strength_billion_USD"] for r in impact_rows],[r["cumulative_output_weighted_infected_share_mean"] for r in impact_rows]),
        "pagerank_vs_peak_impact":corr([r["weighted_pagerank"] for r in impact_rows],[r["peak_output_weighted_infected_share_mean"] for r in impact_rows]),
        "pagerank_vs_cumulative_impact":corr([r["weighted_pagerank"] for r in impact_rows],[r["cumulative_output_weighted_infected_share_mean"] for r in impact_rows]),
        "gross_output_vs_peak_impact":corr([r["gross_output_billion_USD"] for r in impact_rows],[r["peak_output_weighted_infected_share_mean"] for r in impact_rows]),
        "out_degree_vs_peak_impact":corr([r["out_degree"] for r in impact_rows],[r["peak_output_weighted_infected_share_mean"] for r in impact_rows]),
        "candidate_set_size":len(impact_rows),"candidate_set_note":"Stratified union of top-20 nodes by five structural/output metrics and 20 random active nodes; correlations are exploratory and not population-wide causal estimates."}
    (outdir/"source_impact_correlations.json").write_text(json.dumps(impact_corr,indent=2),encoding="utf-8")
    # Visual comparison: each point is a tested seed node; y is its mean simulated peak impact.
    fig,ax=plt.subplots(figsize=(8.0,5.2))
    xv=np.asarray([r["out_strength_billion_USD"] for r in impact_rows])
    yv=np.asarray([r["peak_output_weighted_infected_share_mean"] for r in impact_rows])
    ax.scatter(xv,yv,s=26,alpha=.7)
    ax.set_xscale("symlog",linthresh=1); ax.set_xlabel("Source node out-strength (billion current USD, symlog scale)")
    ax.set_ylabel("Mean peak output-weighted infected share")
    ax.set_title("Structural supplier size vs simulated SIS impact")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(outdir/"centrality_vs_impact.png",dpi=180); plt.close(fig)

    # Threshold cascade: dependency shares of intermediate purchases. The denominator
    # includes ALL intermediate purchases, including flows internal to each aggregate.
    # No self-loop propagates a failure between distinct aggregate nodes.
    # Important aggregation correction: normalize external supplier flows by ALL
    # intermediate inputs of each aggregated buyer, including flows that become
    # within-node (diagonal) flows after sector aggregation. Otherwise dependency
    # shares are artificially inflated because the denominator omits internal flows.
    all_input_totals=np.asarray(Za.maximum(0).sum(axis=0)).ravel()
    inv=np.zeros(na); inv[all_input_totals>0]=1.0/all_input_totals[all_input_totals>0]
    D=(Za_pos @ sparse.diags(inv)).tocsr(); D.eliminate_zeros()
    theta_grid=[0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50]
    cascade_rows=[]
    cascade_sources=[("top_out_strength",top_os),("top_page_rank",top_pr)]+[("random",int(s)) for s in random_sources]
    for label,src in cascade_sources:
        for theta in theta_grid:
            r=threshold_cascade(D,src,theta,args.steps)
            cascade_rows.append({"source_type":label,"source_node":agg_nodes[src],"theta":theta,
                                 "cascade_nodes":r["cascade_nodes"],"cascade_fraction":r["cascade_fraction"],"steps":r["steps"],
                                 "output_weighted_cascade_fraction":float(np.sum(np.maximum(Xa,0)*r["failed"])/max(np.maximum(Xa,0).sum(),1e-12))})
    with (outdir/"threshold_cascade_runs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(cascade_rows[0].keys())); w.writeheader(); w.writerows(cascade_rows)
    # Intervention experiment: source is fixed at the highest out-strength node.
    # Same protection budget, source excluded from all strategies, same random numbers
    # per replicate across strategies (paired Monte Carlo comparison).
    beta0=(gamma/rho_t)*1.4 if rho_t>0 else 0.0
    eligible=np.asarray([i for i in range(na) if i != top_os and (outstr_a[i]+instr_a[i])>0],dtype=int)
    budget=max(1,int(.10*len(eligible)))
    eligible_set=set(eligible.tolist())
    os_order=np.asarray([i for i in np.argsort(outstr_a)[::-1] if int(i) in eligible_set][:budget],dtype=int)
    pr_order=np.asarray([i for i in np.argsort(pr_a)[::-1] if int(i) in eligible_set][:budget],dtype=int)
    intervention_strategies=[("none",None),("top_out_strength",os_order),("top_PageRank",pr_order),("random_protection",None)]
    int_rows=[]
    intervention_reps=max(args.reps,12)
    for rep in range(intervention_reps):
        rng_protect=np.random.default_rng(args.seed+30000+rep)
        random_protect=rng_protect.choice(eligible,size=min(budget,len(eligible)),replace=False)
        for name,protect in intervention_strategies:
            if name=="random_protection": protect=random_protect
            # Paired simulation RNG reduces variance between strategies.
            sim_rng=np.random.default_rng(args.seed+40000+rep)
            runs=run_sis(Tech,Xa,top_os,beta0,gamma,args.steps,1,sim_rng,protected=protect,protection_factor=0.5)
            rr=runs[0]
            int_rows.append({"strategy":name,"replicate":rep,"protection_budget_nodes":0 if name=="none" else int(len(protect)),
                             "source_node":agg_nodes[top_os],"beta":beta0,"gamma":gamma,**rr,"replicate":rep})
    with (outdir/"intervention_runs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(int_rows[0].keys())); w.writeheader(); w.writerows(int_rows)
    # Summaries for plots and report.
    import statistics
    def grouped_mean(rows,keys,metrics):
        buckets={}
        for row in rows: buckets.setdefault(tuple(row[k] for k in keys),[]).append(row)
        out=[]
        for group,items in buckets.items():
            x={k:v for k,v in zip(keys,group)}; x["n_runs"]=len(items)
            for m in metrics: x[m+"_mean"]=float(np.mean([r[m] for r in items])); x[m+"_sd"]=float(np.std([r[m] for r in items],ddof=1)) if len(items)>1 else 0.0
            out.append(x)
        return out
    sis_summary=grouped_mean(sim_rows,["model","source_type","beta_over_beta_critical"],
        ["peak_output_weighted_disruption","cumulative_output_weighted_disruption","final_output_weighted_disruption","peak_node_fraction"])
    with (outdir/"sis_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(sis_summary[0].keys())); w.writeheader(); w.writerows(sis_summary)
    int_summary=grouped_mean(int_rows,["strategy"],["peak_output_weighted_disruption","cumulative_output_weighted_disruption","final_output_weighted_disruption"])
    with (outdir/"intervention_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(int_summary[0].keys())); w.writeheader(); w.writerows(int_summary)
    fig,ax=plt.subplots(figsize=(9,5))
    for model in ["unweighted_SIS","weighted_SIS"]:
        subset=[r for r in sis_summary if r["model"]==model and r["source_type"]=="top_out_strength"]
        subset=sorted(subset,key=lambda x:float(x["beta_over_beta_critical"]))
        ax.errorbar([r["beta_over_beta_critical"] for r in subset],[r["peak_output_weighted_disruption_mean"] for r in subset],
                    yerr=[r["peak_output_weighted_disruption_sd"] for r in subset],marker="o",capsize=3,label=model)
    ax.set_xlabel(r"Transmission level / approximate critical value ($\beta/\beta_c$)")
    ax.set_ylabel("Peak output-weighted disrupted share")
    ax.set_title("SIS model comparison for a high-out-strength seed"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(outdir/"sis_model_comparison.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for label in ["top_out_strength","top_page_rank","random"]:
        rows=[r for r in cascade_rows if r["source_type"]==label]
        means=[]
        for th in theta_grid:
            vals=[r["output_weighted_cascade_fraction"] for r in rows if r["theta"]==th]
            means.append(float(np.mean(vals)) if vals else 0.0)
        ax.plot(theta_grid,means,marker="o",label=label)
    ax.set_xlabel("Failure threshold θ"); ax.set_ylabel("Output-weighted cascade fraction")
    ax.set_title("Threshold cascade sensitivity"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(outdir/"threshold_cascade_sensitivity.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    names=[r["strategy"] for r in int_summary]; vals=[r["peak_output_weighted_disruption_mean"] for r in int_summary]
    ax.bar(names,vals); ax.set_ylabel("Mean peak output-weighted disruption")
    ax.set_title("Equal-budget protection strategy comparison"); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.25); fig.tight_layout()
    fig.savefig(outdir/"intervention_comparison.png",dpi=180); plt.close(fig)
    results={"structural":structural,
        "aggregation":{"countries":len(countries),"sectors":len(SECTOR_GROUPS),"aggregate_nodes":na,"sector_groups":SECTOR_GROUPS,
                       "validation":aggregation_validation},
        "dynamics":{"gamma":gamma,"steps":args.steps,"repetitions":args.reps,"weighted_technical_coefficient_spectral_radius":rho_t,
                     "unweighted_spectral_radius":rho_u,"weighted_beta_critical_approx":gamma/rho_t if rho_t else None,
                     "unweighted_beta_critical_approx":gamma/rho_u if rho_u else None,"beta_multipliers":factors,
                     "model_note":"SIS and threshold cascade outputs are counterfactual simulations; parameters are illustrative and not estimated from observed outages."},
        "sis_summary":sis_summary,"intervention_summary":int_summary,"source_impact_correlations":impact_corr}
    (outdir/"full_analysis_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    # Short report, to be reviewed and expanded into the paper after results are inspected.
    lines=["# OECD ICIO 2022: Reproducible Structural and Dynamical Analysis", "",
           "## Data and construction",f"The raw ICIO file produced {n:,} country-industry nodes and a {n} x {n} intermediate-flow matrix. Flows are in current USD millions. Positive off-diagonal flows define supplier-to-buyer edges.","",
           "## Structural results",f"- Positive directed edges excluding self-flows: {m:,}.",f"- Possible directed edges: {max_m:,}.",f"- Directed density: {density:.6%}.",f"- Weakly connected components: {weak_n:,}; largest component: {int(weak_sizes.max()):,} nodes.",f"- Strongly connected components: {strong_n:,}; largest component: {int(strong_sizes.max()):,} nodes.",f"- Flow matrix total (positive, includes within-node flows): {zpos.sum():,.3f} million current USD.",f"- Self-flow share: {structural['self_flow_share']:.4%}.","",
           "### Top 10 by out-strength (billion current USD)"]
    for r in structural["top_out_strength"][:10]: lines.append(f"{r['rank']}. {r['node']}: {r['value']:,.2f}")
    lines += ["","### Top 10 by weighted PageRank"]
    for r in structural["top_weighted_pagerank"][:10]: lines.append(f"{r['rank']}. {r['node']}: {r['value']:.8f}")
    lines += ["","### Edge-threshold sensitivity", "See edge_threshold_sensitivity.csv. The headline binary graph includes every positive off-diagonal flow; additional thresholds (in USD millions) show how density and connectivity change when very small flows are omitted.","",
      "## Model results summary","The SIS, threshold-cascade, source-impact and intervention results are in their corresponding CSV files and figures. Interpret them as outcomes of the specified counterfactual models, not as observed economic losses.","",
      "## Limitations","The annual ICIO table has no time-stamped disruption episodes. SIS transmission and recovery settings and cascade thresholds are assumptions. The aggregation to country x 13 broad sectors improves computational feasibility and turns within-group flows into diagonal entries. These flows are retained in buyer input denominators but excluded as transmission links between distinct aggregate nodes. Source-impact correlations use a deliberately stratified subset and are exploratory, not estimates over every source node. Model rankings should be checked for parameter and aggregation sensitivity.",""]
    (outdir/"analysis_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Completed. Outputs: {outdir}")
    print(f"Elapsed: {time.time()-start:.1f}s")
    print(f"Nodes={n:,}; off-diagonal edges={m:,}; density={density:.6%}; aggregated model nodes={na:,}")
    print(f"Weighted SIS spectral radius={rho_t:.6g}; approximate beta_c={gamma/rho_t if rho_t else float('inf'):.6g}")

if __name__=="__main__": main()
