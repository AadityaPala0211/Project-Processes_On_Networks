# Full ICIO Analysis — Single Entry Point

## Run the complete project

From the repository root, run only:

```bash
uv run master.py
```

No command-line options or manual paths are needed. The master automatically finds `data/2022_SML.csv`, creates the output folders, runs the full structural analysis, aggregates into 12 broad sectors per economy for dynamical simulations, compares clipped-linear and nonlinear models for every country-sector source, selects five representative nodes, runs the 27-combination parameter grid, and writes plots and a Markdown report.

### Output locations

- `results/linear_nonlinear/`: structural metrics, all-source dynamic results, linear/nonlinear source comparison, five representative nodes, sensitivity runs, summaries, run metadata, and `analysis_report.md`.
- `images/linear_nonlinear/`: structural and dynamical plots.
- `master.py`: the one file the user runs.
- `analysis/`: implementation modules called by the master; do not run these individually.

### Default dynamics

The baseline for every source node is initial disruption 0.20, propagation strength beta 0.30, recovery rate rho 0.10, source held for 3 rounds, maximum 50 model steps. All other nodes start at zero. The two models use the same dependency matrix, initial conditions, parameters, and measurement definitions:

`linear: d_j(t+1) = clip((1-rho)d_j(t) + beta e_j(t), 0, 1)`

`nonlinear: d_j(t+1) = clip((1-rho)d_j(t) + (1-d_j(t))(1-exp(-beta e_j(t))), 0, 1)`

where `e_j(t)` is the sum of disrupted supplier states weighted by each supplier's share of the buyer's intermediate inputs. The nonlinear model saturates as disruption approaches one.

Sensitivity on the five representative nodes tests initial disruption `{0.10, 0.20, 0.40}`, beta `{0.10, 0.30, 0.50}`, and recovery `{0.05, 0.10, 0.20}` for both models. The automatic source roles are highest out-strength, highest PageRank excluding prior picks, highest approximate weighted betweenness excluding prior picks, highest nonlinear cumulative baseline impact, and a seeded random control node.

### Interpretation

All time constants are shared assumptions because the annual ICIO table does not identify propagation or recovery rates. Output-weighted disruption is an index of gross output associated with disrupted nodes, not real production loss, GDP loss, or predictive accuracy. The full 4,050-node network is used for structural analysis; the 12-sector country-sector network (typically 972 nodes) is used for repeated dynamics to keep the analysis tractable. The automatic workflow tests the code and runs the full data pipeline whenever analysis source files change on this branch, then commits generated outputs.

# Global Production Network Project — Reproducible Analysis Bundle

## New primary model: signed shock propagation

**Revised final report (IEEE conference style, two-column, A4, double-sided):** [paper/supply_chain_network_project_report.tex](paper/supply_chain_network_project_report.tex). The LaTeX source has been cleaned up to remove duplicated sections and includes the signed shock model, target experiments, worked example, prior SIS/cascade/intervention results, limitations, and references. A workflow is configured to regenerate and commit the report PDF on pushes to this branch: [.github/workflows/build-ieee-report.yml](.github/workflows/build-ieee-report.yml). A concise [plain-language companion](paper/signed_shock_propagation_report.md) is also available.

The legacy SIS-style model represents a disruption spreading through the supplier network, but it does not explicitly model alternative suppliers gaining a substitution opportunity. The new `code/signed_shock_propagation.py` adds a bounded signed state: negative values represent adverse pressure; positive values represent a potential opportunity signal for alternative suppliers in the same detailed ICIO activity.

With (D_{ij}) equal to supplier (i)'s share of buyer (j)'s intermediate inputs and (C_{ki}) allocating a disrupted supplier's opportunity to potential substitutes, the update is

`x(t+1) = clip(0.55*x(t) + 0.35*D.T*x(t) + 0.25*C*max(-x(t), 0), -1, 1)`.

Substitute weights are based on similarities in observed buyer-flow patterns; out-strength is used only as a fallback. The coefficients are teaching assumptions, not estimated elasticities. Negative and positive indicators are dimensionless model indices, **not** actual output losses, GDP losses, or export gains. The model has no data on spare capacity, inventories, prices, or response speed.

Run the signed scenario set (highest-flow activity, lowest positive-flow activity, plus Russia B06 where present) using the exact raw CSV:

```bash
python code/signed_shock_propagation.py /path/to/2022_SML.csv --out results_signed --steps 12 --severity 1.0
python -m unittest discover -s tests -v
```

The script writes node/industry rankings, negative/positive scenario summaries, time series, per-node effects, metadata, two figures, and a 156-node instructional network (12 selected economies × 13 sector groups). The raw ICIO CSV was not bundled in this repository during this revision, so the new signed scenario numbers must be generated locally before reporting them as dataset-based results.

This bundle contains the signed-shock report and source, Python code, historical baseline tables, and reproducible scenario tooling for the OECD ICIO 2022 country–industry production network.

## Input data

The raw input file is **not bundled** because it is about 90 MB and was supplied separately. Use the official OECD regular ICIO 2022_SML table. The expected file is a CSV with country–industry rows/columns, an intermediate-input block, and an `OUT` vector/column.

The file analysed for this report was `2022_SML(1).csv`, 90,120,552 bytes, SHA-256:

`fbab8fd2727bda9fd8c6cfee8227816db82d33bef6aa4b90194ca5dcbc5ad14b`

If your copy has a different checksum, verify that it is the same ICIO edition and year before comparing outputs.

## Requirements

Python 3.10+; packages: `numpy`, `scipy`, `matplotlib`.

Install them with:

```bash
pip install numpy scipy matplotlib
```

## Re-running the analysis from the raw CSV

1. Run the full structural analysis and initial diagnostic experiments:

```bash
python code/supply_chain_complete_analysis.py /path/to/2022_SML.csv \
  --out results_rebuilt --steps 40 --reps 12 --random-seeds 12 --seed 2022 --topn 20
```

2. Run the refined 100-replication SIS and paired intervention comparisons. This writes the more stable result tables used in the paper:

```bash
python code/run_refined_monte_carlo.py --csv /path/to/2022_SML.csv \
  --out results_rebuilt --steps 40 --reps 100 --seed 2022
```

3. Regenerate the figures based on the results tables:

```bash
python code/generate_refined_figures.py --results results_rebuilt --figures results_rebuilt
```

The base analysis creates the other structural figures (top out-/in-strength, PageRank, degree distribution, sector-flow heatmap and source-impact scatter). Copy all PNGs from `results_rebuilt` into `paper/figures/` to replace the report's figures. Compile the IEEE report from the `paper` directory with `pdflatex` twice:

```bash
cd paper
pdflatex supply_chain_network_project_report.tex
pdflatex supply_chain_network_project_report.tex
```

## Network definitions

- Node: one economy/region × one OECD industry.
- Directed edge `i -> j`: intermediate flow supplied by industry `i` and used by industry `j`.
- Edge weight: monetary value in current USD millions.
- Full structural network: 4,050 nodes (81 regions including Rest of World × 50 activities).
- Dynamical network: 1,053 nodes (the same 81 regions × 13 broad groups).
- Aggregation preserves total intermediate flows and gross output; internal within-aggregate flows are kept in buyers' input denominator but excluded from cross-node spreading edges.

## Interpretation and limitations

Structural statistics derive from the actual 2022 ICIO matrix. SIS, threshold-cascade and intervention results are **counterfactual simulations**, not observed historical outages or forecasts of GDP loss. The output-weighted state measure represents the share of summed gross output belonging to nodes currently marked disrupted, not the fraction of output actually lost. Transmission, recovery, thresholds, and the intervention's 50% edge-weight reduction are assumed parameters. The source-impact correlation analysis uses a selected/stratified set of 65 aggregate nodes and six runs per candidate, so it is exploratory rather than population-wide evidence.

The paper's key refinements are (a) edge-threshold sensitivity, since raw binary density counts tiny monetary flows; (b) a weighted versus unweighted SIS comparison; (c) threshold-cascade sensitivity; and (d) equal-budget protection comparisons using 100 paired Monte Carlo runs.

Fill the group/member/institution placeholders in the paper before submission.
