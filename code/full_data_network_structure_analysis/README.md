# Full-data network structure analysis

This folder contains a separate, reproducible structural-analysis pipeline for the **full OECD ICIO 2022 country–industry network**, designed to complement the project's existing aggregate/dynamical analysis scripts.

## Network definition and matrix orientation

- **Node:** one country/region × one detailed industry, for example `USA_C20`.
- **Directed edge:** `i → j) means country-industry `i` supplies an intermediate input used by country-industry `j`.
- **Weighted adjacency value:** intermediate flow in **million current USD**.
- **Binary adjacency value:** 1 when the flow is positive at the chosen threshold; otherwise 0.
- **Self-flows:** the default structural adjacency matrices remove diagonal entries. The separate `intermediate_flows_positive_full.npz` retains positive diagonal/self-flows for accounting checks.
- **Node index:** matrix row/column number is mapped to a country-industry label in `node_index.csv`.

The current source CSV is `data/2022_SML.csv`. The scripts read the CSV; they do not modify it. Output matrices are generated locally and should normally not be committed because they are derived and can be large.

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r code/full_data_network_structure_analysis/requirements.txt
```

Dependencies: NumPy, SciPy, and Matplotlib. The pipeline uses SciPy sparse matrices/graph algorithms rather than constructing a large object-based graph, which is important for this relatively dense 4,050-node dataset.

## Run the entire pipeline

From the repository root:

```bash
python code/full_data_network_structure_analysis/run_full_structural_analysis.py \
  data/2022_SML.csv --out results_full_network
```

The pipeline executes the scripts in this order:

1. Create weighted and binary adjacency matrices and a node index.
2. Compute degree, strength, weighted PageRank, Gini coefficients and degree/strength distributions.
3. Measure weak/strong connected components and shortest paths from a reproducible sample of source nodes.
4. Compute weighted PageRank/eigenvector centrality and an estimate of directed betweenness from sampled sources.
5. Test sensitivity to different minimum-flow thresholds.
6. Summarize flows by country and broad sector and generate network/sector figures.

Useful options:

```bash
# Increase source sampling for shortest paths and betweenness (more computation).
python code/full_data_network_structure_analysis/run_full_structural_analysis.py \
  data/2022_SML.csv --out results_full_network --sample-sources 128 \
  --betweenness-sources 32 --topn 40 --seed 2022

# Re-run a single stage after the matrices exist.
python code/full_data_network_structure_analysis/03_connectivity_and_paths.py \
  --matrix-dir results_full_network --sample-sources 128 --seed 2022

# Skip sampled betweenness if runtime is a concern; PageRank/eigenvector and all
# other stages still run.
python code/full_data_network_structure_analysis/run_full_structural_analysis.py \
  data/2022_SML.csv --out results_full_network --skip-centrality
```

The `--skip-centrality` option skips the whole centrality stage, including PageRank/eigenvector output from that stage. Stage 2 already creates weighted PageRank and corresponding node rankings, so use it if you only want to avoid sampled betweenness computation.

## Scripts and outputs

### `01_build_adjacency_matrix.py`

Creates:

- `adjacency_weighted_full.npz`: sparse weighted adjacency, without self-loops by default.
- `adjacency_binary_full.npz`: sparse binary adjacency under the positive-flow edge definition.
- `intermediate_flows_positive_full.npz`: all positive flows including self-flows.
- `node_index.csv`: index ↔ node mapping.
- `gross_output_million_usd.csv`: gross output vector aligned to matrix indices.
- `adjacency_matrix_metadata.json`: input checksum, orientation, shape, units and matrix counts.

The matrices use zero-based indices. To load a matrix in Python:

```python
from scipy import sparse
W = sparse.load_npz("results_full_network/adjacency_weighted_full.npz")
A = sparse.load_npz("results_full_network/adjacency_binary_full.npz")
```

### `02_basic_network_metrics.py`

Creates `node_metrics_full.csv`, `structural_summary_full.json`, node rankings and figures for:
out-degree, in-degree, out-strength, in-strength, weighted PageRank, gross output, strength Gini, degree/strength distributions, and Spearman correlations.

Interpretation:
- **Out-degree:** number of distinct purchasing nodes supplied.
- **In-degree:** number of distinct supplier nodes used.
- **Out-strength:** sum of intermediate flows supplied to other nodes.
- **In-strength:** sum of intermediate inputs purchased from other nodes.

These are production-network flow measures. They are not GDP, exports, profits, or observed output losses.

### `03_connectivity_and_paths.py`

Computes exact weak/strong component counts and component sizes. For shortest paths, it calculates exact directed, unweighted distances from a reproducible sample of source nodes instead of materializing an all-pairs distance matrix. Output includes sampled mean/median/90th-percentile/max path lengths and reachability fractions.

Path lengths are measured in hops, not weighted by monetary flow. Sampling is important to keep runtime and memory reasonable.

### `04_centrality_analysis.py`

Computes weighted PageRank, weighted eigenvector centrality, degrees/strengths, and a **sampled estimate** of directed betweenness centrality. Betweenness uses unweighted shortest paths and samples source nodes; its output is explicitly marked as an estimate. Increase `--betweenness-sources` for more precise but more expensive estimates, or set it to 0 to skip the betweenness calculation.

### `05_edge_threshold_sensitivity.py`

Rebuilds binary topology under several minimum-flow cutoffs. The default thresholds are 0, 0.01, 0.1, 1, 10 and 100 million current USD. It reports density, edge count, isolate count, component counts and the fraction of flow retained. This is essential because treating every tiny positive accounting flow as a meaningful structural link can make binary density misleading.

### `06_flow_aggregation_and_visualization.py`

Produces country-level and broad-sector flow summaries, a supplier-sector → purchasing-sector heatmap, and a subnetwork of high-total-strength nodes with selected high-flow edges. The visualization is a readable selected subgraph; calculations and summaries use the full network.

### `run_full_structural_analysis.py`

Runs the stages in the required order.

## Reproducibility and interpretation notes

- The input CSV checksum is written into `adjacency_matrix_metadata.json` so runs can be compared against the exact input file.
- Random sampling is deterministic for a fixed seed.
- Structural values describe the annual 2022 input-output table.
- Shortest-path summaries and betweenness are sampled estimates; component counts and degree/strength/PageRank/eigenvector values are computed on the full network.
- Threshold results depend on the analyst's chosen minimum flow. Report the threshold alongside any binary degree or density result.
- Annual input-output flows are not time-stamped outage records. No script in this folder produces causal claims or forecasts of real economic disruption.
- Keep generated `.npz`, CSV metrics, and PNG files in the output folder. Commit the source scripts and documentation, not regenerated result bundles, unless your team explicitly wants to version those artifacts.

## Tests

Run the fast synthetic-data test (no 90 MB input file required):

```bash
python -m unittest discover -s tests -p "test_full_data_adjacency_matrix.py" -v
```
