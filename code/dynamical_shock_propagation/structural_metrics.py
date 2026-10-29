"""Reuse existing structural metrics and calculate thresholded undirected k-core."""
from __future__ import annotations
import csv
import numpy as np
from scipy import sparse

def read_rows(path):
    if not path.is_file(): return []
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def core_numbers_undirected(A: sparse.csr_matrix):
    """Batagelj-Zaversnik core decomposition on an undirected CSR graph."""
    A=A.tocsr(copy=True)
    A.setdiag(0); A.eliminate_zeros()
    A.data[:]=1
    A=A.maximum(A.T).tocsr()
    A.sort_indices()
    n=A.shape[0]
    degree=np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    if n==0: return degree
    max_degree=int(degree.max(initial=0))
    bins=np.zeros(max_degree+1,dtype=np.int64)
    for d in degree: bins[d]+=1
    start=0
    for d in range(max_degree+1):
        count=bins[d]; bins[d]=start; start+=count
    vert=np.empty(n,dtype=np.int64); pos=np.empty(n,dtype=np.int64)
    for v in range(n):
        d=degree[v]; pos[v]=bins[d]; vert[pos[v]]=v; bins[d]+=1
    for d in range(max_degree,0,-1): bins[d]=bins[d-1]
    bins[0]=0
    for i in range(n):
        v=vert[i]
        for u in A.indices[A.indptr[v]:A.indptr[v+1]]:
            if degree[u]>degree[v]:
                du=degree[u]; pu=pos[u]; pw=bins[du]; w=vert[pw]
                if u!=w:
                    vert[pu]=w; pos[w]=pu; vert[pw]=u; pos[u]=pw
                bins[du]+=1; degree[u]-=1
    return degree

def load_structural_metrics(matrix_dir,nodes,W,kcore_threshold_million_usd=10.0):
    """Load saved full-network metrics and supplement k-core at a documented cutoff.

    K-core is calculated on the undirected projection of edges whose flow is at
    least the cutoff (default 10 million current USD), because raw positive-flow
    topology is extremely dense and tiny flows can make binary core numbers hard
    to interpret.
    """
    matrix_dir=__import__("pathlib").Path(matrix_dir)
    rows=read_rows(matrix_dir/"centrality_metrics_full.csv")
    if not rows: rows=read_rows(matrix_dir/"node_metrics_full.csv")
    if rows and len(rows)==len(nodes) and all(rows[i].get("node")==nodes[i] for i in range(len(nodes))):
        merged=[dict(r) for r in rows]
    else:
        out=np.asarray(W.sum(axis=1)).ravel(); inc=np.asarray(W.sum(axis=0)).ravel()
        A=W.copy(); A.data[:]=1
        od=np.asarray(A.sum(axis=1)).ravel(); ind=np.asarray(A.sum(axis=0)).ravel()
        merged=[{"node":nodes[i],"out_strength_million_current_usd":float(out[i]),
                 "in_strength_million_current_usd":float(inc[i]),"out_degree":int(od[i]),"in_degree":int(ind[i])}
                for i in range(len(nodes))]
    thresholded=W.copy()
    thresholded.data=(thresholded.data>=kcore_threshold_million_usd).astype(np.uint8)
    thresholded.eliminate_zeros()
    core=core_numbers_undirected(thresholded)
    for i,r in enumerate(merged):
        r["k_core"]=int(core[i])
        r["k_core_threshold_million_usd"]=float(kcore_threshold_million_usd)
    return merged
