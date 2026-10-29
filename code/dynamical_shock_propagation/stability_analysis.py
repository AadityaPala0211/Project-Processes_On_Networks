"""Spectral threshold and local zero-disruption stability diagnostics."""
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

def spectral_radius(A,tol=1e-9,max_iter=20000):
    A=A.tocsr().astype(float)
    if A.nnz==0: return 0.0
    if A.shape==(1,1): return abs(float(A[0,0]))
    try:
        values=eigs(A,k=1,which="LM",return_eigenvectors=False,tol=tol,maxiter=max_iter)
        return float(np.max(np.abs(values)))
    except Exception:
        # Power iteration fallback; use a nonnegative start for nonnegative matrices.
        x=np.ones(A.shape[0],dtype=float); x/=np.linalg.norm(x); old=0.0
        for _ in range(max_iter):
            y=np.asarray(A@x).ravel(); norm=float(np.linalg.norm(y))
            if norm==0: return 0.0
            x=y/norm
            if abs(norm-old)<=tol*max(1.0,norm): return norm
            old=norm
        return old

def analyze(P,beta,recovery_rate):
    r=spectral_radius(P.T.tocsr())
    beta_c=recovery_rate/r if r>0 else None
    # P is nonnegative, so its Perron mode gives the dominant eigenvalue of J0.
    rho_j=(1.0-recovery_rate)+beta*r
    return {"node_count":int(P.shape[0]),"dependency_matrix_spectral_radius":r,
            "beta":float(beta),"recovery_rate":float(recovery_rate),
            "critical_beta":beta_c,"beta_over_critical":beta/beta_c if beta_c else None,
            "linearized_jacobian_spectral_radius":rho_j,"zero_state_locally_stable":bool(rho_j<1.0),
            "jacobian":"J0=(1-rho)I+beta*P.T","threshold_condition":"beta*r(P.T) < rho",
            "all_four_models_share_jacobian":True,
            "caveat":"Local stability near zero disruption only; not an observed economic crisis threshold."}

def write_result(path,result):
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
