"""ICIO CSV loading, validation, and conservative country-sector aggregation."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from scipy import sparse

NODE_RE = re.compile(r"^[A-Z0-9]{3}_[A-Z0-9_]+$")

SECTORS = [
    "Agriculture_forestry_fishing",
    "Mining_quarrying",
    "Food_textiles_wood_paper",
    "Chemicals_petroleum_minerals",
    "Other_manufacturing",
    "Utilities",
    "Construction",
    "Wholesale_retail_trade",
    "Transport_accommodation_food",
    "Information_communications",
    "Finance_real_estate_professional_admin",
    "Public_social_other_services",
]

def industry_group(code: str) -> str:
    """Map the 50 ICIO activities into 12 broad teaching groups."""
    c = code.strip().upper()
    if c in {"A01", "A02", "A03"}:
        return SECTORS[0]
    if c.startswith("B"):
        return SECTORS[1]
    if c in {"C10T12", "C13T15", "C16", "C17_18"}:
        return SECTORS[2]
    if c in {"C19", "C20", "C21", "C22", "C23"}:
        return SECTORS[3]
    if c.startswith("C"):
        return SECTORS[4]
    if c in {"D", "E"} or c.startswith(("D", "E")):
        return SECTORS[5]
    if c == "F" or c.startswith("F"):
        return SECTORS[6]
    if c.startswith("G"):
        return SECTORS[7]
    if c.startswith("H") or c == "I":
        return SECTORS[8]
    if c.startswith("J"):
        return SECTORS[9]
    if c in {"K", "L", "M", "N"}:
        return SECTORS[10]
    if c in {"O", "P", "Q", "R", "S", "T", "U"}:
        return SECTORS[11]
    raise ValueError(f"ICIO activity code {code!r} was not mapped to a broad sector.")

def load_icio(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Load only the square country-industry intermediate-flow block and OUT.

    The CSV is streamed rather than loaded as a large DataFrame. The node order
    comes from the header; each row is checked against that order to avoid
    silently misaligning suppliers and buyers.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"ICIO input not found: {path}\nExpected data/2022_SML.csv under the project root."
        )
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"ICIO file is empty: {path}") from exc

    header_positions: dict[str, int] = {}
    for idx, name in enumerate(header):
        key = name.strip()
        if key and key not in header_positions:
            header_positions[key] = idx

    # Use the row-label block to distinguish true country-industry nodes from
    # final-demand columns that can also look like CODE_CODE (e.g. AGO_HFCE).
    # The ICIO file lists all country-industry rows first, followed by final-demand
    # and accounting rows. Every selected node must also have a matching header label.
    nodes = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                break
            label = row[0].strip()
            if not NODE_RE.fullmatch(label) or label not in header_positions:
                break
            nodes.append(label)
    if not nodes:
        raise ValueError(
            "No country-industry rows were found in the CSV. "
            "Expected row labels such as USA_C20."
        )
    if len(set(nodes)) != len(nodes):
        raise ValueError("Duplicate country-industry row labels detected.")

    node_columns = np.asarray([header_positions[name] - 1 for name in nodes], dtype=np.int64)
    if np.any(node_columns < 0):
        raise ValueError("Unexpected ICIO header layout: node columns overlap the row-label column.")

    out_header_pos = header_positions.get("OUT")
    out_numeric_pos = out_header_pos - 1 if out_header_pos is not None else None
    n = len(nodes)
    Z = np.zeros((n, n), dtype=np.float64)
    output = np.full(n, np.nan, dtype=np.float64)
    negative_entry_count = 0
    parsed_rows = 0

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for i in range(n):
            try:
                row = next(reader)
            except StopIteration as exc:
                raise ValueError(f"CSV ended after {i} country-industry rows; expected {n}.") from exc
            label = row[0].strip() if row else ""
            if label != nodes[i]:
                raise ValueError(
                    f"Row-order validation failed at row {i}: expected {nodes[i]!r}, got {label!r}."
                )
            try:
                values = np.asarray(row[1:], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f"Non-numeric value found in row {label!r}.") from exc
            if node_columns.size and int(node_columns.max()) >= values.size:
                raise ValueError(f"Row {label!r} is shorter than the header-defined node columns.")
            selected = values[node_columns]
            if not np.all(np.isfinite(selected)):
                raise ValueError(f"NaN/Inf found in intermediate-flow row {label!r}.")
            negative_entry_count += int(np.count_nonzero(selected < 0))
            Z[i, :] = selected
            if out_numeric_pos is not None and out_numeric_pos < values.size:
                output[i] = values[out_numeric_pos]
            parsed_rows += 1

    output_fallback_used = False
    if np.any(~np.isfinite(output)):
        output_fallback_used = True
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row and row[0].strip() == "OUT":
                    try:
                        values = np.asarray(row[1:], dtype=np.float64)
                    except ValueError as exc:
                        raise ValueError("The OUT row contains non-numeric values.") from exc
                    if int(node_columns.max()) >= values.size:
                        raise ValueError("The OUT row is shorter than the node-column range.")
                    output = values[node_columns].astype(np.float64, copy=True)
                    break
    if np.any(~np.isfinite(output)):
        missing = int(np.count_nonzero(~np.isfinite(output)))
        raise ValueError(
            f"Could not determine gross output for {missing} nodes. "
            "Expected an OUT column or a special OUT row."
        )
    if np.any(output < 0):
        output = np.maximum(output, 0.0)

    metadata = {
        "source_file": str(path),
        "source_file_bytes": path.stat().st_size,
        "raw_country_industry_nodes": int(n),
        "flow_matrix_shape": [int(n), int(n)],
        "parsed_country_industry_rows": int(parsed_rows),
        "negative_flow_entries_clipped_to_zero": int(negative_entry_count),
        "gross_output_fallback_used_OUT_row": bool(output_fallback_used),
        "flow_units": "millions of current USD",
        "direction": "supplier row -> purchasing industry column",
    }
    # IO tables should not contain negative intermediate flows for this purpose.
    # Clipping occurs in place to make the subsequent graph and summaries valid.
    np.maximum(Z, 0.0, out=Z)
    return nodes, Z, output, metadata

def aggregate_country_sector(
    nodes: list[str], Z_full: sparse.csr_matrix, gross_output: np.ndarray
) -> tuple[list[str], np.ndarray, sparse.csr_matrix, np.ndarray, list[str], np.ndarray]:
    """Aggregate detailed nodes to each economy x 12 broad sectors.

    Full flows (including within-node flows) are summed first. The aggregated
    input totals therefore retain flows internal to a broad sector, while the
    dynamic propagation matrix later removes its diagonal to avoid self-spread.
    """
    countries_by_node: list[str] = []
    activity_codes: list[str] = []
    for label in nodes:
        country, code = label.split("_", 1)
        countries_by_node.append(country)
        activity_codes.append(code)
    countries = list(dict.fromkeys(countries_by_node))
    country_index = {country: i for i, country in enumerate(countries)}
    sector_index = {sector: i for i, sector in enumerate(SECTORS)}
    groups = [industry_group(code) for code in activity_codes]
    detailed_to_aggregate = np.asarray([
        country_index[c] * len(SECTORS) + sector_index[g]
        for c, g in zip(countries_by_node, groups)
    ], dtype=np.int64)
    aggregate_nodes = [
        f"{country}_{sector}"
        for country in countries
        for sector in SECTORS
    ]
    n_agg = len(aggregate_nodes)
    S = sparse.csr_matrix(
        (np.ones(len(nodes)), (np.arange(len(nodes)), detailed_to_aggregate)),
        shape=(len(nodes), n_agg),
    )
    Z_aggregate = (S.T @ Z_full @ S).tocsr()
    Z_aggregate.sum_duplicates()
    Z_aggregate.eliminate_zeros()
    X_aggregate = np.bincount(
        detailed_to_aggregate,
        weights=np.maximum(np.asarray(gross_output, dtype=float), 0.0),
        minlength=n_agg,
    ).astype(np.float64)
    # Sector list parallel to detailed nodes, used for full-network sector flow matrix.
    return aggregate_nodes, detailed_to_aggregate, Z_aggregate, X_aggregate, groups, S
