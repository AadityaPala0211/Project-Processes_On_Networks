# Full structural analysis of the OECD ICIO production network

**Course:** SC452 — Processes on Networks (Autumn 2026–27)  
**Scope:** Full 4,050-node country/region × detailed-industry network, informed by the `full-data-network-structure-analysis` and `dynamical-shock-propogation` branches.  
**Input year:** OECD ICIO reference year 2022.  
**Important:** This report interprets the repository's checked-in structural outputs. It does not claim to have rerun the 90 MB raw data locally. Dynamic simulation results are kept separate from measured annual flow structure.

## 1. Executive summary

The project represents the global production system as a directed weighted network. Each node is an economy/region × detailed industry. A directed edge `i → j` means supplier industry `i` provides intermediate inputs to buyer industry `j`; its weight is the monetary intermediate flow in million current USD. The full detailed network has 4,050 nodes (81 regions including Rest of World × 50 activities), with self-flows excluded from the structural adjacency matrix.

The checked-in full-network baseline contains 10,166,696 positive off-diagonal directed edges out of 16,398,450 possible, giving raw positive-flow density 0.61998 (62.00%). The network is highly connected at this definition: its largest weak component has 3,928 nodes (96.99%) and its largest strongly connected component has 3,927 nodes (96.96%). There are 122 isolates. Flow strengths are concentrated: out-strength Gini is 0.8625 and in-strength Gini is 0.8525.

The 62% raw density should not be interpreted as 62% economically meaningful dependence. When the minimum edge flow is raised to US$100 million, directed density falls to 0.3449%, while 89.22% of positive off-diagonal monetary flow is retained. Thus many links are tiny in monetary size; threshold sensitivity and weighted measures are essential to a defensible structural interpretation.

## 2. Data, node and edge definitions

- **Data source in the project:** OECD Inter-Country Input-Output (ICIO) regular table, reference year 2022 (`2022_SML.csv`; about 90 MB in the repository branch).
- **Node:** economy/region × detailed ICIO activity, e.g. `USA_C20`.
- **Directed edge:** supplier `i` → buyer `j` if the supplier sells a positive intermediate input flow to the buyer.
- **Weight:** observed annual intermediate flow, million current USD.
- **Self-links:** excluded from the structural adjacency matrix to represent inter-node dependencies, but retained in the raw positive-flow matrix and in the denominator used by the dynamics model.
- **Negative entries:** clipped to zero for the flow adjacency construction.

The weighted adjacency is `W[i,j] = Z[i,j]` for positive, off-diagonal intermediate flow and zero otherwise. The binary adjacency is `A[i,j] = 1(W[i,j] > 0)`. Keep the row-supplier/column-buyer orientation fixed: reversing it reverses the interpretation of in/out degree and strength.

The metadata records the input SHA-256 as `c27bf2ab425692b376965b90d12b73b34bd87047894cd0b1bcabdd8a8416b5cc`. The existing README on `main` lists a different checksum for an earlier dataset copy, so results should be compared only after confirming the exact file checksum and ICIO edition. The stored structural results in the structural-analysis branch and baseline notes should be treated as the referenced run, not a fresh execution during this commit.

## 3. Structural measures and project results

### 3.1 Size and density

| Measure | Stored result | Interpretation |
|---|---:|---|
| Nodes | 4,050 | 81 regions × 50 detailed industries |
| Possible directed non-self edges | 16,398,450 | `N(N−1)` |
| Positive directed non-self edges | 10,166,696 | At least one positive monetary flow |
| Directed density | 0.619979 (62.00%) | Positive-edge fraction under zero cutoff |
| Total positive off-diagonal flow | US$81.128 trillion | Sum of intermediate flows between distinct nodes |
| Total positive intermediate flow incl. self-use | US$103.410 trillion | Accounting total, not structural edge weight sum |
| Positive self-flow | US$22.282 trillion | Within-node use, not a link between distinct nodes |
| Gross output sum | US$199.691 trillion | Sum of node output vector; not GDP/value added |

Density is the fraction of possible directed links that exist under the chosen edge definition. It is not a fraction of production that is dependent, and it is especially sensitive to whether any tiny positive accounting flow counts as a link.

### 3.2 Degree

- **Out-degree** of supplier `i`: number of distinct buyer nodes receiving positive intermediate flows from `i`.
- **In-degree** of buyer `j`: number of distinct suppliers from which `j` purchases positive intermediate inputs.
- Mean out-degree and mean in-degree are both 2,510.30, as expected for a directed graph because every directed edge contributes once to total out-degree and once to total in-degree.
- Median out-degree is 2,921.5; median in-degree is 2,705.5. The reported 90th percentiles are 3,782.0 (out) and 3,422.1 (in). Maximum out-degree is 3,913 and maximum in-degree 3,858.

The network is dense under the raw-positive rule, but node degrees still vary considerably. Degree counts partners without considering the size of the transaction.

### 3.3 Strength and flow concentration

- **Out-strength** `s_out(i) = Σ_j W[i,j]`: total intermediate inputs supplied by a node to other nodes.
- **In-strength** `s_in(j) = Σ_i W[i,j]`: total intermediate inputs purchased by a node from other nodes.
- Out-strength Gini = 0.8625; in-strength Gini = 0.8525. Values close to 1 indicate a highly unequal distribution of monetary flow across nodes.
- Spearman rank correlation: out-degree vs out-strength = 0.7636; in-degree vs in-strength = 0.9451. Having more partners is associated with higher total flow, but degree and strength are not interchangeable.
- Out-strength vs weighted PageRank = 0.7986; out-strength vs gross output = 0.9091; in-strength vs gross output = 0.9669.

The largest saved out-strength nodes include USA_M (professional, scientific and technical activities; about US$2.196T), USA_G (wholesale/retail trade; about US$1.971T), CHN_G (wholesale/retail trade; about US$1.803T), USA_N (administrative/support activities; about US$1.442T), and USA_K (finance/insurance; about US$1.367T). CHN_F (construction) has the highest reported in-strength, about US$3.263T. These are intermediate-flow measures, not exports, profit, GDP, or value added.

### 3.4 Connectivity and components

A weakly connected component ignores edge direction when checking whether nodes are connected. A strongly connected component requires directed reachability in both directions between every pair of nodes in that component.

- Weak components: 123; largest has 3,928 nodes (96.99%).
- Strong components: 124; largest has 3,927 nodes (96.96%).
- Isolated nodes: 122 under the positive off-diagonal edge definition.

The giant weak and strong components show that nearly all detailed production activities belong to one mutually interlinked directed core. This is a structural statement about the annual positive-flow network, not evidence that a shock reaches every node with equal strength or speed.

### 3.5 Shortest paths and reachability

The code computes exact directed, unweighted shortest paths from 64 reproducibly sampled source nodes (seed 2022), rather than all-pairs paths. Among the 247,401 reachable ordered pairs in that sample:

- Mean shortest path: 1.285 hops.
- Median: 1 hop.
- 90th percentile: 2 hops.
- Maximum: 3 hops.

These numbers describe the sampled source/target pairs and positive-edge topology. They are not an all-pairs exact mean, and path lengths count links, not monetary flow, time, or economic substitution cost.

### 3.6 Centrality

Centrality tries to identify nodes that are structurally prominent in ways not captured by degree alone.

- **Weighted PageRank:** importance propagated through weighted incoming links, with damping and dangling-node handling. Computed for the full network.
- **Weighted eigenvector centrality:** nodes score highly when connected to other high-scoring nodes, using the monetary-flow matrix; computed for the full network.
- **Directed betweenness:** counts how often a node lies on shortest directed paths. The repository estimates it with unweighted Brandes from 12 sampled source nodes (seed 2022), scaled to a network-level estimate. It is not monetary-flow weighted and should be labelled an estimate.

Do not write that betweenness is an exact full-network value. Do not compare its scale directly with weighted PageRank/eigenvector scores.

### 3.7 Clustering

Local clustering asks how interconnected a node's neighbours are. Since the main production network is directed, the project constructs an undirected projection `A_u = max(A, Aᵀ)` for this measure only. For a reproducible sample of 100 of 4,050 nodes (seed 2022), mean local clustering is 0.8550, median 0.8971, standard deviation 0.2045, minimum 0, maximum 0.9882.

This is a sampled summary, not a full-network census. Symmetrization is only for clustering; all other structural metrics retain supplier → buyer direction. A high clustering coefficient means neighbours tend to be connected to one another in the undirected projection; it does not by itself mean the supply chain is resilient.

### 3.8 Threshold sensitivity (key methodological result)

The pipeline repeats binary topology calculations after keeping only flows at or above a minimum cutoff. Threshold units are **million current USD**.

| Minimum flow | Edges | Density | Isolates | Weak components | Largest weak component | Positive flow retained |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10,166,696 | 62.00% | 122 | 123 | 3,928 (96.99%) | 100.00% |
| 0.01 | 4,531,407 | 27.63% | 135 | 136 | 3,915 (96.67%) | 99.9866% |
| 0.1 | 2,320,570 | 14.15% | 154 | 155 | 3,896 (96.20%) | 99.8858% |
| 1 | 896,358 | 5.466% | 207 | 208 | 3,843 (94.89%) | 99.2745% |
| 10 | 256,248 | 1.563% | 390 | 391 | 3,660 (90.37%) | 96.7227% |
| 100 | 56,563 | 0.3449% | 1,021 | 1,027 | 3,019 (74.54%) | 89.2161% |

At US$1 million cutoff, density has dropped to 5.47% while more than 99.27% of the off-diagonal flow value remains. At US$100 million, only 0.345% of possible edges remain, but nearly 89.22% of flow is retained. This demonstrates that link count is dominated by many low-value relationships; a flow-weighted interpretation is more stable. The cutoff remains an analyst choice, not a natural law. Report it whenever presenting thresholded binary metrics.

### 3.9 Country/sector aggregation and visualisations

The pipeline sums flows by economy and by broad sector, and produces a supplier-sector → buyer-sector flow heatmap. Aggregation helps interpret a network that is too large to draw node-by-node. The displayed high-strength subnetwork is a selected visual subset; it must not be confused with the full 4,050-node calculations. Matrix aggregation should preserve flow totals, subject to the specific handling of internal/self-aggregate flows described by the code.

## 4. How structural metrics connect to shock propagation

The dynamic branch reuses the same detailed 4,050-node orientation. It builds a dependency matrix:

`P[i,j] = Z_cross[i,j] / total_intermediate_purchases[j]`

The numerator excludes self-flows; the denominator includes the buyer's total intermediate purchases including self-use. Thus `P[i,j]` estimates the fraction of buyer `j`'s intermediate purchases supplied by `i`, under the data's annual accounting values.

For nonnegative state `d_i(t) ∈ [0,1]`, buyer exposure is `(Pᵀ d)_j`. Four updates are implemented: clipped linear, bilinear/mass-action SIS, independent-edge MMCA/Reed-Frost-style SIS, and exponential-hazard SIS. All share zero-state Jacobian `J₀ = (1−ρ)I + βPᵀ` and local threshold `β_c = ρ / spectral_radius(Pᵀ)` (when spectral radius is positive). This is a local mathematical stability diagnostic, not an observed economic crisis threshold.

A fixed-state one-step ordering is tested: Model 1 ≥ Model 2 ≥ Model 3 ≥ Model 4. This does not prove that different multi-step trajectories stay ordered after their states diverge. The shock simulations use assumptions for transmission, recovery, initial severity, source holding, and intervention; their output-weighted disruption is a model index, not actual output loss, GDP loss, or a forecast.

The other branch also includes a signed shock model: negative states represent adverse pressure and positive states represent potential opportunity for alternative suppliers, with candidate substitutes selected from similar buyer-flow patterns within the same detailed activity. Spare capacity, inventories, prices and elasticities are unavailable; positive state is therefore not a guaranteed production gain. The signed model is a teaching stress-test, distinct from the four nonnegative SIS-style models.

## 5. Mapping to SC452 course content

The attached SC452 syllabus (Appendix I, pages 2–3) lists Module 1 Networks (13 lectures), Module 2 Dynamics (10), Module 3 Dynamical Processes on Networks (10), and Module 4 Problem design (7). Course outcomes CO1–CO5 emphasise graph measures, model comparison, fixed points/stability/bifurcations, network spreading/synchronisation/resilience/temporal or multiplex networks, and research communication.

| SC452 topic | How this project addresses it | Coverage status / caution |
|---|---|---|
| Graph representation; directed/weighted network | 4,050 economy-industry nodes; supplier → buyer weighted edges | Directly covered |
| Degree and degree distribution | In/out-degree and distributions | Directly covered |
| Strength / weighted structure | In/out-strength, flow Gini, weighted adjacency | Directly covered |
| Centrality | PageRank, eigenvector, sampled directed betweenness | Covered; betweenness is approximate |
| Connectivity and paths | Weak/strong components, sampled shortest paths/reachability | Covered; path summary sampled |
| Clustering | Local clustering in undirected projection | Partially covered; only 100 sampled nodes in stored run |
| Network models: Erdős–Rényi, scale-free, small-world | No checked-in calibrated ER/BA/WS model comparison identified in the two requested analysis packages | Not demonstrated; add a model benchmark only if required by assessment |
| Degree distributions / general random graphs | Observed ICIO degree distributions; not necessarily fitted to a random-graph family | Empirical structure covered; model fitting not demonstrated |
| Percolation | Thresholded edge deletion and component changes | Related robustness/sensitivity experiment, but not a formal probabilistic percolation threshold study |
| Autonomous systems, fixed points, linear stability, bifurcations | Dynamic branch has a zero-state Jacobian and spectral threshold | Local stability covered; fixed-point diagrams and bifurcation continuation are not established by structural metrics alone |
| Epidemic spreading / SIS | Four nonnegative SIS-style disruption processes; sensitivity and intervention experiments | Covered as an analogy for supply-chain disruption, not epidemiological evidence |
| Heterogeneity | Flow/degree distributions and node-wise source-shock census | Covered structurally and dynamically |
| Interventions | Equal-budget node protection strategies in older aggregate simulations; full-resolution dynamic intervention tooling | Covered as counterfactual modelling; parameter-dependent |
| Temporal/multiplex networks | Annual ICIO matrix and a static dependency network | Not covered as genuine temporal/multiplex data |
| Synchronization in coupled oscillators | No oscillator phase/state model in these branches | Not covered |
| Resilience | Components/threshold sensitivity and dynamic recovery/protection simulations | Partially covered; not a complete formal resilience framework |
| Research design/reporting | Reproducible scripts, metadata, tests, figures and limitations | Directly supports CO5 |

A key point for the viva: the project is fundamentally a **static, annual, directed weighted production network** used as the substrate for counterfactual dynamics. It should not be described as a temporal network, multiplex network, epidemic dataset, oscillator system or measured disruption history.

## 6. Limitations and interpretation discipline

1. Annual monetary input-output flows do not provide event timestamps, disruption duration, inventory buffers, spare capacity, prices, elasticities, supplier switching speed, or observed outage histories.
2. A positive flow edge is an accounting relationship, not proof of critical dependence. Small flows create high raw binary density.
3. Thresholds are sensitivity parameters. They alter topology substantially; conclusions must report the cutoff and units.
4. Shortest paths and betweenness use unweighted hops. The 64-source path statistics and 12-source betweenness are sampled estimates.
5. Clustering statistics are based on 100 sampled nodes and an undirected projection.
6. Flow strengths and gross output must not be labelled GDP, exports, profits, welfare or losses.
7. SIS/threshold-cascade/intervention simulations are conditional on assumed model parameters. They are not forecasts or causal policy-impact estimates.
8. Signed positive opportunity does not demonstrate spare capacity or realised gains.
9. OECD's Rest of World aggregation may hide country-specific supply relationships; do not attribute ROW effects to an individual country without separate evidence.
10. A mismatch in raw input checksum means numerical outputs may not be directly comparable.

## 7. Recommended presentation sequence

1. Explain the data and node-edge definition; draw one supplier → buyer edge example.
2. Show size, density and why self-flows are treated separately.
3. Explain degree vs strength with one concrete node ranking.
4. Show Gini and correlations to discuss heterogeneity.
5. Show weak/strong giant components and sampled path lengths.
6. Explain PageRank/eigenvector/betweenness and their different meanings.
7. Show clustering and explicitly mention sample/projection.
8. Use the threshold sensitivity figure/table as the critical robustness caveat.
9. Connect the dependency-share matrix to SIS and signed shock dynamics.
10. Finish with course coverage, assumptions and limitations.

## 8. Reproducible commands

From repository root (Python 3.10+):

```bash
python -m pip install -r code/full_data_network_structure_analysis/requirements.txt
python code/full_data_network_structure_analysis/run_full_structural_analysis.py data/2022_SML.csv --out results_full_network --seed 2022
python -m pip install -r code/dynamical_shock_propagation/requirements.txt
python -m unittest discover -s code/dynamical_shock_propagation/tests -v
python code/dynamical_shock_propagation/run_all.py --csv data/2022_SML.csv --matrix-dir results_full_network --out results_dynamics --source-mode all
```

For higher precision structural sampling, increase `--sample-sources` and `--betweenness-sources`; use `--all-clustering-nodes` to calculate clustering for every node. These choices can increase runtime significantly. The full dynamic census and its sensitivity/intervention sweeps are also computationally heavier; a top-k smoke run can validate setup but is not a full-source analysis.

## 9. References

- OECD, *Inter-Country Input-Output Tables* (ICIO), 2025 edition, regular table, reference year 2022. https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html
- Newman, M. E. J., *Networks: An Introduction*.
- Barrat, A., Barthélemy, M., and Vespignani, A. (2008), *Dynamical Processes on Complex Networks*, Cambridge University Press.
- Boccaletti, S. et al. (2006), “Complex networks: Structure and dynamics,” *Physics Reports*, 424(4–5), 175–308.
- Pastor-Satorras, R. et al. (2015), “Epidemic processes in complex networks,” *Reviews of Modern Physics*, 87(3), 925.
- Liu, X. et al. (2022), “Network Resilience,” *Physics Reports*, 971, 1–108.

---

**Interpretation rule:** report structural statistics as properties of the stored 2022 annual ICIO network; report dynamic values as outputs of a specified counterfactual model and parameter set. Never conflate the two.
