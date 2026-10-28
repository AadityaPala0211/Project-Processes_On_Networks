#!/usr/bin/env python3
"""Build the full-resolution adjacency matrix from OECD ICIO 2022_SML.csv.

Direction convention used throughout this project:
    row node i -> column node j
means supplying country-industry i provides intermediate inputs to purchasing
country-industry j. Input-flow values are in million current USD.

The primary output is a sparse CSR matrix saved as .npz, together with a node
index mapping each matrix row/column to a stable country-industry label.
Self-flows are retained in the flow matrix output for accounting transparency,
but removed from the structural adjacency matrix by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse

NODE_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9_]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_layout(csv_path: Path):
    """Discover country-industry row labels and their matching column positions."""
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {csv_path}") from exc

        header_positions = {}
        for position, raw_name in enumerate(header):
            name = raw_name.strip()
            if name and name not in header_positions:
                header_positions[name] = position

        nodes = []
        for row in reader:
            if not row:
                break
            label = row[0].strip()
            if NODE_RE.fullmatch(label) and label in header_positions:
                nodes.append(label)
            else:
                break

    if not nodes:
        raise ValueError(
            "No country-industry rows found. Expected matching row/column labels "
            "such as USA_C20, as in the OECD 2022_SML ICIO table."
        )
    if len(nodes) != len(set(nodes)):
        raise ValueError("Duplicate country-industry row labels found.")

    # CSV row label occupies column 0, so numeric row values start at column 1.
    value_column_positions = np.asarray(
        [header_positions[node] - 1 for node in nodes], dtype=np.int64
    )
    if np.any(value_column_positions < 0):
        raise ValueError("Unexpected layout: network column maps to row-label column.")

    out_col = header_positions.get("OUT")
    if out_col is not None:
        out_col -= 1
    return header, nodes, value_column_positions, out_col


def read_intermediate_block(csv_path: Path, nodes: list[str],
                            value_column_positions: np.ndarray,
                            out_col: int | None):
    """Load the country-industry intermediate-flow block and gross output vector."""
    n = len(nodes)
    flows = np.zeros((n, n), dtype=np.float64)
    gross_output = np.full(n, np.nan, dtype=np.float64)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for i, expected_label in enumerate(nodes):
            try:
                row = next(reader)
            except StopIteration as exc:
                raise ValueError(
                    f"CSV ended after {i} node rows; expected {n} rows."
                ) from exc

            actual_label = row[0].strip()
            if actual_label != expected_label:
                raise ValueError(
                    f"Row alignment error at {i}: expected {expected_label!r}, "
                    f"found {actual_label!r}."
                )
            try:
                values = np.asarray(row[1:], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value found in country-industry row {actual_label}."
                ) from exc

            if value_column_positions.max(initial=-1) >= len(values):
                raise ValueError(f"Row {actual_label} is shorter than the header.")
            flows[i, :] = values[value_column_positions]
            if out_col is not None and out_col < len(values):
                gross_output[i] = values[out_col]

    # Some ICIO layouts encode output as an OUT row rather than OUT column.
    if np.any(~np.isfinite(gross_output)):
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            next(reader)
            for row in reader:
                if row and row[0].strip() == "OUT":
                    values = np.asarray(row[1:], dtype=np.float64)
                    if value_column_positions.max(initial=-1) >= len(values):
                        raise ValueError("OUT row is shorter than the header.")
                    gross_output = values[value_column_positions].copy()
                    break

    if not np.all(np.isfinite(flows)):
        raise ValueError("Intermediate flow matrix contains NaN or infinite values.")
    if np.any(~np.isfinite(gross_output)):
        raise ValueError("Could not find a finite gross-output value for every node.")
    return flows, gross_output


def write_node_index(nodes: list[str], destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["matrix_index", "node", "country_code", "industry_code"])
        for index, node in enumerate(nodes):
            country, industry = node.split("_", 1)
            writer.writerow([index, node, country, industry])


def main():
    parser = argparse.ArgumentParser(
        description="Create full 4,050-node OECD ICIO adjacency/flow matrices."
    )
    parser.add_argument("csv", type=Path, help="Path to the raw OECD 2022_SML.csv")
    parser.add_argument("--out", type=Path, default=Path("results_full_network"),
                        help="Output directory (default: results_full_network)")
    parser.add_argument(
        "--include-self-loops", action="store_true",
        help="Keep within-node flows in the structural adjacency matrix. "
             "By default, self-flows are removed from the network matrix."
    )
    parser.add_argument(
        "--keep-negative-flows", action="store_true",
        help="Do not clip negative matrix entries to zero. This is not normally "
             "appropriate for a flow adjacency matrix."
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading ICIO file: {csv_path}", flush=True)
    header, nodes, positions, out_col = discover_layout(csv_path)
    flows, gross_output = read_intermediate_block(csv_path, nodes, positions, out_col)

    # For flow-network semantics, negative values are not valid edge weights.
    matrix = flows.copy() if args.keep_negative_flows else np.maximum(flows, 0.0)
    if not args.include_self_loops:
        np.fill_diagonal(matrix, 0.0)

    weighted = sparse.csr_matrix(matrix)
    weighted.eliminate_zeros()
    adjacency = weighted.copy()
    adjacency.data = (adjacency.data > 0).astype(np.uint8)
    adjacency.eliminate_zeros()

    # Preserve the raw intermediate block too, with self-flows, for accounting.
    raw_positive = sparse.csr_matrix(np.maximum(flows, 0.0))
    raw_positive.eliminate_zeros()

    weighted_path = out_dir / "adjacency_weighted_full.npz"
    binary_path = out_dir / "adjacency_binary_full.npz"
    raw_flow_path = out_dir / "intermediate_flows_positive_full.npz"
    sparse.save_npz(weighted_path, weighted, compressed=True)
    sparse.save_npz(binary_path, adjacency, compressed=True)
    sparse.save_npz(raw_flow_path, raw_positive, compressed=True)
    write_node_index(nodes, out_dir / "node_index.csv")
    np.savetxt(out_dir / "gross_output_million_usd.csv",
               gross_output, delimiter=",", header="gross_output_million_current_usd",
               comments="")

    n = len(nodes)
    possible = n * (n - 1)
    edge_count = int(adjacency.nnz)
    weighted_total_ex_self = float(weighted.sum())
    raw_total = float(raw_positive.sum())
    self_total = float(np.trace(raw_positive.toarray())) if n <= 5000 else float(np.trace(flows.clip(min=0)))
    metadata = {
        "source_csv": str(csv_path),
        "source_csv_size_bytes": csv_path.stat().st_size,
        "source_csv_sha256": sha256_file(csv_path),
        "node_count": n,
        "matrix_shape": [n, n],
        "matrix_orientation": "row supplier i -> column buyer j",
        "edge_definition": "positive intermediate input flow",
        "flow_units": "million current USD",
        "self_loops_in_structural_matrices": bool(args.include_self_loops),
        "negative_flows_clipped_to_zero": not args.keep_negative_flows,
        "positive_intermediate_flow_total_including_self_flows_million_usd": raw_total,
        "positive_self_flow_total_million_usd": self_total,
        "positive_intermediate_flow_total_excluding_self_loops_million_usd": weighted_total_ex_self,
        "binary_edge_count": edge_count,
        "possible_directed_edges_excluding_self_loops": possible,
        "binary_density": float(edge_count / possible) if possible else 0.0,
        "weighted_matrix_file": weighted_path.name,
        "binary_matrix_file": binary_path.name,
        "raw_positive_flow_matrix_file": raw_flow_path.name,
        "node_index_file": "node_index.csv",
        "gross_output_file": "gross_output_million_usd.csv",
        "note": "Sparse NPZ files use zero-based row/column indices mapped by node_index.csv."
    }
    (out_dir / "adjacency_matrix_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Nodes: {n}", flush=True)
    print(f"Binary directed edges (excluding self-loops): {edge_count:,}", flush=True)
    print(f"Binary density: {metadata['binary_density']:.6f}", flush=True)
    print(f"Saved weighted adjacency: {weighted_path}", flush=True)
    print(f"Saved binary adjacency: {binary_path}", flush=True)
    print(f"Saved node index: {out_dir / 'node_index.csv'}", flush=True)


if __name__ == "__main__":
    main()
