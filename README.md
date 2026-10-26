# Global Production Network Project — Reproducible Analysis Bundle

This bundle contains the revised six-page IEEE-style report, its LaTeX source and figures, Python code, and computed output tables for the OECD ICIO 2022 country–industry production network.

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
