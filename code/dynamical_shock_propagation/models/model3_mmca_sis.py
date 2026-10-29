"""Model 3: independent-edge MMCA / Reed-Frost-style SIS update."""
from __future__ import annotations
import numpy as np
from scipy import sparse

def update_batch(states, P, beta, recovery_rate, edge_arrays=None, edge_chunk_size=500_000):
    """Update columns of states using 1-prod_i(1-q_ij), without dense N×N arrays.

    states has shape (nodes, scenarios). Sparse edges are chunked; each supplier
    edge has q_ij=clip(beta*P_ij*d_i,0,1). Log-products safely handle very small
    probabilities, while q=1 maps to certain infection (log no-infection=-inf).
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[:, None]
    n, batch = states.shape
    if edge_arrays is None:
        coo = P.tocoo()
        edge_arrays = (coo.row, coo.col, coo.data)
    rows, cols, weights = edge_arrays
    log_no = np.zeros((n, batch), dtype=np.float64)
    scenario_offsets = np.arange(batch, dtype=np.int64)[None, :]
    for start in range(0, len(weights), edge_chunk_size):
        stop = min(start + edge_chunk_size, len(weights))
        rr, cc, ww = rows[start:stop], cols[start:stop], weights[start:stop]
        q = np.clip(beta * ww[:, None] * states[rr, :], 0.0, 1.0)
        with np.errstate(divide="ignore", invalid="raise"):
            terms = np.log1p(-q)
        flat_indices = (cc[:, None] * batch + scenario_offsets).ravel()
        sums = np.bincount(flat_indices, weights=terms.ravel(), minlength=n * batch)
        log_no += sums.reshape(n, batch)
    with np.errstate(under="ignore", invalid="raise"):
        no_infection = np.exp(log_no)
    infection = np.clip(1.0 - no_infection, 0.0, 1.0)
    result = (1.0 - recovery_rate) * states + (1.0 - states) * infection
    return np.clip(result, 0.0, 1.0)

def infection_probability(d, P, beta):
    """Single-state infection probability helper, useful for tests and analysis."""
    d = np.asarray(d, dtype=float).reshape(-1)
    coo = P.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    log_no = np.zeros(P.shape[1], dtype=float)
    q = np.clip(beta * vals * d[rows], 0.0, 1.0)
    with np.errstate(divide="ignore"):
        contributions = np.log1p(-q)
    np.add.at(log_no, cols, contributions)
    return np.clip(1.0 - np.exp(log_no), 0.0, 1.0)

def update(d, P, beta, recovery_rate):
    d = np.asarray(d, dtype=float).reshape(-1)
    result = update_batch(d[:, None], P, beta, recovery_rate)
    return result[:, 0]
