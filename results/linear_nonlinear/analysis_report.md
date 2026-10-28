# OECD ICIO 2022: Structural and Linear-versus-Nonlinear Dynamics

## Executive summary
- Full detailed network: **4,050 nodes**, **10,166,696 positive off-diagonal edges**, density **62.00%**.
- Dynamical network: **972 country-sector nodes** across 12 sectors.
- Flow conservation relative error: 0.000e+00; gross-output conservation relative error: 0.000e+00.
- Each model ran 972 source scenarios with the same baseline parameters.
- Mean peak output-weighted disruption: linear **3.134%**, nonlinear **2.074%**.
- Mean cumulative disruption: linear **0.3410 share-steps**, nonlinear **0.2526 share-steps**.

The numerical dynamics are counterfactual outcomes, not forecasts. Annual ICIO has no observed, time-stamped disruption labels, so the model comparison is about simulated behaviour rather than predictive accuracy.

## 1. Structural analysis

| Analysis | What it measures | Why it is useful |
|---|---|---|
| Out-strength | Total intermediate sales supplied by a node | Finds large suppliers |
| In-strength | Total intermediate inputs purchased | Identifies nodes with large input dependence in monetary terms |
| In/out degree | Number of distinct supplier/buyer links | Summarizes topology; sensitive to tiny positive flows |
| Weighted PageRank | Recursive influence on the directed weighted network | Candidate proxy for downstream influence |
| Flow Gini | Concentration of supplier/buyer flow totals | Shows whether flows are evenly distributed |
| Component and edge-cutoff analysis | Connectivity as small flows are excluded | Tests whether raw connectivity is driven by tiny transactions |
| Sector-flow matrix | Global flows between 12 broad sectors | Gives a more readable production-system overview |

Out-strength Gini is 0.863; in-strength Gini is 0.852. Raw density is highly sensitive to the edge threshold, so do not treat every tiny positive transaction as equally important.

### Top 10 detailed suppliers by out-strength

| Rank | Node | Out-strength (billion current USD) |
|---:|---|---:|
| 1 | USA_M | 2,196.15 |
| 2 | USA_G | 1,971.28 |
| 3 | CHN_G | 1,803.38 |
| 4 | USA_N | 1,442.33 |
| 5 | USA_K | 1,366.58 |
| 6 | USA_L | 1,248.49 |
| 7 | CHN_A01 | 1,167.97 |
| 8 | CHN_C20 | 1,023.04 |
| 9 | CHN_C23 | 1,017.11 |
| 10 | CHN_C24A | 1,000.47 |

### Top 10 by in-strength

| Rank | Node | In-strength (billion current USD) |
|---:|---|---:|
| 1 | CHN_F | 3,263.46 |
| 2 | USA_G | 2,058.02 |
| 3 | USA_O | 1,311.09 |
| 4 | CHN_C10T12 | 1,253.38 |
| 5 | USA_L | 1,201.31 |
| 6 | USA_Q | 964.95 |
| 7 | CHN_C26 | 942.34 |
| 8 | USA_F | 937.67 |
| 9 | CHN_C28 | 864.00 |
| 10 | CHN_G | 847.54 |

## 2. Dynamical network

For tractable repeated dynamics, the 50 detailed activities are grouped into 12 broad activity groups for each economy. This produces about 972 country-sector nodes. Flow and gross-output totals are checked after aggregation.

$$P_{ij}=\frac{Z^{cross}_{ij}}{\sum_k Z^{all}_{kj}}, \qquad e_j(t)=\sum_i P_{ij}d_i(t).$$

P gives each buyer's intermediate input share from each supplier. Its denominator includes all intermediate purchases, including flows internal to an aggregated sector; cross-node propagation removes the diagonal.

### Linear approximation
$$d_j(t+1)=\operatorname{clip}_{[0,1]}\left[(1-\rho)d_j(t)+\beta e_j(t)\right].$$

The update is linear before clipping. Clipping bounds severity between zero and one, but means the implemented baseline is a clipped-linear model.

### Local stability diagnostic

Spectral radius of the dependency influence matrix Q is **0.68286**; approximate linearized critical beta is **0.14644**.
The chosen beta is **0.30** (beta/beta-critical = **2.05**), so the near-zero disease-free state is **not locally stable** under the un-clipped linearization.
This is a mathematical local-stability check, not an empirically estimated economic tipping point. The clipping/nonlinear terms bound states but do not guarantee that every source trajectory settles during the 50-step horizon.

### Nonlinear SIS-inspired model
$$d_j(t+1)=\operatorname{clip}_{[0,1]}\left[(1-\rho)d_j(t)+(1-d_j(t))(1-e^{-\beta e_j(t)})\right].$$

The nonlinear response saturates: when a node is already highly disrupted, the remaining opportunity for it to become more disrupted is smaller.

| Parameter | Shared baseline value | Meaning |
|---|---:|---|
| Initial source disruption | 0.20 | Same 20% severity for each source |
| Propagation strength beta | 0.30 | Shared influence of disrupted supplier exposure |
| Recovery rate rho | 0.10 | Share of existing disruption that recovers per step |
| Source hold | 3 steps | Externally maintain the source shock for three rounds |
| Maximum length | 50 steps | Stop earlier if state change meets the convergence tolerance |

These values are not estimated from ICIO. Shared parameters ensure that baseline differences mainly reflect source-node location, input dependencies, and gross-output weights.

The baseline experiment shocks every country-sector node once, with all other nodes initially undisturbed, for each of the two models.

## 3. Linear versus nonlinear results

| Metric | Linear | Nonlinear |
|---|---:|---:|
| Mean peak disruption | 3.1343% | 2.0745% |
| Median peak disruption | 0.6933% | 0.6173% |
| 95th percentile peak disruption | 12.8700% | 8.7756% |
| Mean cumulative disruption (share-steps) | 0.3410 | 0.2526 |
| Mean final disruption | 3.1343% | 2.0745% |
| Mean maximum nodes at >=1% disruption | 470.87 | 455.80 |

Mean paired peak difference (nonlinear minus linear): -1.0598 percentage points. Mean paired cumulative difference: -0.0884 share-steps.

Runs converged before the 50-step cap: linear **0/972**, nonlinear **0/972**. A non-converged trajectory's final value is a horizon value, not a steady-state estimate.

Interpret these numbers as differences between mathematical models under matched assumptions. Without a separate dataset of observed outages and recovery trajectories, neither model can be called more accurate in the predictive sense.

## 4. Five representative sources

| Case | Selection rule | Data-selected node |
|---|---|---|
| A | Highest weighted out-strength | USA_Finance_real_estate_professional_admin |
| B | Highest weighted PageRank as downstream-influence proxy | CHN_Other_manufacturing |
| C | Highest approximate weighted betweenness | DEU_Other_manufacturing |
| D | Highest nonlinear cumulative impact in source census | USA_Wholesale_retail_trade |
| E | Seeded random control from middle out-strength band | PAK_Chemicals_petroleum_minerals |

A: highest out-strength. B: highest weighted PageRank, excluding A. C: highest approximate weighted betweenness, excluding A and B. D: highest cumulative nonlinear impact in the all-source census, excluding A-C. E: seeded random control from the middle 60% of out-strength, avoiding earlier choices.

Betweenness is approximated on the aggregate graph after filtering edges below US$10 million; this limits the influence of very small flows. The random control uses a fixed seed.

Each of the five sources is tested over 27 combinations: initial severity 0.10/0.20/0.40, beta 0.10/0.30/0.50, and recovery 0.05/0.10/0.20. Both models are run for all 27 combinations (270 runs total).

### Sensitivity results (averaged over representative sources)

| Model | Parameter | Value | Mean peak disruption | Mean cumulative share-steps |
|---|---|---:|---:|---:|
| linear | initial_disruption | 0.10 | 40.319% | 7.899 |
| linear | initial_disruption | 0.20 | 44.369% | 9.678 |
| linear | initial_disruption | 0.40 | 47.554% | 11.654 |
| linear | beta | 0.10 | 1.032% | 0.274 |
| linear | beta | 0.30 | 41.960% | 7.212 |
| linear | beta | 0.50 | 89.250% | 21.744 |
| linear | recovery_rate | 0.05 | 59.679% | 14.134 |
| linear | recovery_rate | 0.10 | 46.856% | 10.326 |
| linear | recovery_rate | 0.20 | 25.707% | 4.770 |
| nonlinear | initial_disruption | 0.10 | 23.736% | 4.842 |
| nonlinear | initial_disruption | 0.20 | 25.823% | 5.794 |
| nonlinear | initial_disruption | 0.40 | 27.712% | 6.854 |
| nonlinear | beta | 0.10 | 0.950% | 0.256 |
| nonlinear | beta | 0.30 | 22.480% | 4.070 |
| nonlinear | beta | 0.50 | 53.841% | 13.163 |
| nonlinear | recovery_rate | 0.05 | 42.232% | 9.632 |
| nonlinear | recovery_rate | 0.10 | 26.159% | 5.971 |
| nonlinear | recovery_rate | 0.20 | 8.880% | 1.887 |

Across both models, greater beta and lower recovery generally lead to larger and longer cascades in this parameter grid. Nonlinear saturation reduces average impact relative to linear propagation at the same beta, while higher initial severity raises impact. These trends are conditional on the selected five nodes and the stipulated parameter values.

## 5. Metrics and caveats

- Peak disruption: maximum over time of gross-output-weighted node severity.
- Cumulative disruption: sum of the weighted severity index over model rounds; represents magnitude and persistence.
- Affected nodes: count with severity at least 1%, a reporting convention.
- Final disruption: weighted index at the end of the simulation.

The output-weighted index is not actual production lost, GDP loss, or welfare loss. Gross output counts intermediate production at multiple stages. The model also does not include spare capacity, inventories, price changes, or explicit positive substitution gains. Those require data and equations beyond these two requested disruption models.

Input: data/2022_SML.csv
Input SHA-256: c27bf2ab425692b376965b90d12b73b34bd87047894cd0b1bcabdd8a8416b5cc
Run timestamp UTC: 2026-10-28T18:29:15.239540+00:00
Python 3.12.15; NumPy 2.5.4; SciPy 1.18.1.

Reproduce all stages by running only: uv run master.py
