"""Load/reuse full-resolution matrices from the existing structural pipeline."""
from __future__ import annotations
import csv, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
from scipy import sparse

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[2]
BUILDER = ROOT / "code" / "full_data_network_structure_analysis" / "01_build_adjacency_matrix.py"

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_node_index(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if [int(r["matrix_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("node_index.csv indices must be contiguous and zero-based")
    nodes = [r["node"] for r in rows]
    if len(set(nodes)) != len(nodes):
        raise ValueError("Duplicate node labels found")
    return nodes

def ensure_artifacts(csv_path, matrix_dir):
    matrix_dir, csv_path = Path(matrix_dir), Path(csv_path)
    required = [matrix_dir/"adjacency_weighted_full.npz",
                matrix_dir/"intermediate_flows_positive_full.npz",
                matrix_dir/"node_index.csv", matrix_dir/"gross_output_million_usd.csv"]
    if all(p.is_file() for p in required):
        return
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing structural artifacts and ICIO CSV: {csv_path}")
    if not BUILDER.is_file():
        raise FileNotFoundError(f"Existing adjacency builder not found: {BUILDER}")
    matrix_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(BUILDER), str(csv_path), "--out", str(matrix_dir)], check=True)

def build_dependency_matrix(W_cross, all_purchase_totals):
    """P[i,j]=cross-supplier flow i->j / all intermediate purchases of buyer j."""
    totals = np.asarray(all_purchase_totals, dtype=np.float64).reshape(-1)
    if W_cross.shape[1] != totals.size:
        raise ValueError("Purchase totals do not align with matrix columns")
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    P = (W_cross.tocsr() @ sparse.diags(inverse)).tocsr()
    P.setdiag(0.0); P.eliminate_zeros()
    if P.nnz and (not np.isfinite(P.data).all() or np.any(P.data < 0)):
        raise ValueError("Dependency matrix must be finite and nonnegative")
    # Column sums should not exceed 1 except for insignificant floating error.
    column_sums = np.asarray(P.sum(axis=0)).ravel()
    if np.any(column_sums > 1.0 + 1e-7):
        bad = int(np.argmax(column_sums))
        raise ValueError(f"Dependency shares exceed one at buyer index {bad}: {column_sums[bad]}")
    return P

def load_network(matrix_dir, csv_path=None, expected_nodes=4050):
    matrix_dir = Path(matrix_dir).expanduser().resolve()
    csv_path = Path(csv_path or ROOT/"data"/"2022_SML.csv").expanduser().resolve()
    ensure_artifacts(csv_path, matrix_dir)
    W = sparse.load_npz(matrix_dir/"adjacency_weighted_full.npz").tocsr().astype(np.float64)
    raw = sparse.load_npz(matrix_dir/"intermediate_flows_positive_full.npz").tocsr().astype(np.float64)
    nodes = read_node_index(matrix_dir/"node_index.csv")
    output = np.asarray(np.loadtxt(matrix_dir/"gross_output_million_usd.csv", delimiter=",", skiprows=1), dtype=float).reshape(-1)
    if W.shape != (len(nodes), len(nodes)) or raw.shape != W.shape:
        raise ValueError(f"Matrix dimensions {W.shape}/{raw.shape} do not match {len(nodes)} nodes")
    if expected_nodes and len(nodes) != expected_nodes:
        raise ValueError(f"Expected {expected_nodes} detailed nodes; found {len(nodes)}")
    if output.size != len(nodes):
        raise ValueError("Gross-output vector length does not match the node index")
    if not np.isfinite(W.data).all() or np.any(W.data < 0) or not np.isfinite(raw.data).all() or np.any(raw.data < 0):
        raise ValueError("Flows must be finite and nonnegative")
    if not np.isfinite(output).all() or np.any(output < 0):
        raise ValueError("Gross output must be finite and nonnegative")
    if np.any(W.diagonal() != 0):
        raise ValueError("Structural adjacency must exclude self-links")
    raw_totals = np.asarray(raw.sum(axis=0)).ravel()  # includes self-use; required denominator
    P = build_dependency_matrix(W, raw_totals)
    meta_path = matrix_dir/"adjacency_matrix_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    orientation = meta.get("matrix_orientation", "")
    if orientation and "row supplier" not in orientation.lower():
        raise ValueError(f"Unexpected matrix orientation metadata: {orientation}")
    return {"W":W, "raw_flows":raw, "P":P, "nodes":nodes, "output":output,
            "matrix_dir":str(matrix_dir), "csv_path":str(csv_path),
            "csv_sha256":sha256_file(csv_path) if csv_path.is_file() else meta.get("source_csv_sha256"),
            "metadata":meta}
