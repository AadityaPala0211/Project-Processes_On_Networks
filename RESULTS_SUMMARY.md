# Results Summary — Signed Shock Propagation and Historical Baseline

## New signed model: status and target comparisons

The added model in `code/signed_shock_propagation.py` represents both (a) adverse pressure that travels from a disrupted supplier to its buyers and (b) positive opportunity signals allocated to plausible alternative suppliers. The signed state is dimensionless and bounded. It is not a measurement of actual output lost or gained.

The raw OECD ICIO CSV was not present in the repository at this revision. Therefore, the model code and synthetic-network tests are committed, but **the full ICIO signed-shock scenario tables have not been computed in this commit**. Run the script on the same local `2022_SML.csv` used for the baseline to generate the tables and figures. No signed-scenario result values are invented here.

Stored baseline structural rankings select these targets (flow-strength-based, not GDP-based):

| Scenario | Representative target | Baseline flow evidence |
|---|---|---|
| Highest-flow activity | `USA_G` — wholesale/retail trade | Activity G totals about US$9.291T outgoing intermediate flows; USA_G contributes about US$1.971T |
| Lowest positive-flow activity | `CHN_C301` — building of ships and boats | Activity C301 totals about US$83.3B; CHN_C301 contributes about US$22.9B |
| Oil illustration | `RUS_B06` — crude petroleum and natural gas extraction | Node out-strength about US$165.5B in the stored baseline |

Household own-use activity `T` is excluded from the low-industry target because it has zero positive supplier flow in the stored results. Venezuela may be grouped inside Rest of World in the country coverage; a modelled ROW opportunity must not be described as a Venezuela-specific gain without separate country-level evidence. B06 combines oil and natural gas extraction.

## What was tested

Four unit tests pass on a small synthetic network. They verify buyer input-share normalization, same-industry-only substitute edges, the appearance of both negative and positive states, and initial shock severity. These tests validate the code mechanics, not the realism of parameters or economic forecasts.

## Historical results from the previous model

## Structural network (actual input data)

- 4,050 country–industry nodes (81 regions including Rest of World × 50 activities).
- 10,166,696 positive off-diagonal flows of 16,398,450 possible directed links; raw positive-edge density = **62.00%**.
- The total intermediate flow including within-industry self-flows is **$103.410 trillion** in current USD; off-diagonal flows total **$81.128 trillion**. Self-flows are 21.55% of total positive intermediate flows and do not count as inter-node edges.
- 123 weakly connected components; largest component 3,928 nodes. Largest strongly connected component 3,927 nodes. 122 isolates under the positive-flow definition.
- Out-strength Gini = 0.863; in-strength Gini = 0.852, showing strongly concentrated monetary flows.
- Spearman correlations over all 4,050 nodes: out-degree vs out-strength 0.764; in-degree vs in-strength 0.945; out-strength vs weighted PageRank 0.799; out-strength vs gross output 0.909.

## Edge-threshold sensitivity

Directed density falls from 62.00% when any positive flow is an edge to 5.47% at a $1 million flow cutoff, 1.56% at $10 million, and 0.345% at $100 million. Therefore, binary degree depends heavily on the cutoff; flow-weighted metrics are central to interpretation.

## Selected high-strength nodes (full 4,050-node network)

Out-strength means intermediate flows supplied, not exports.

1. USA_M — US professional, scientific and technical activities: approx. $2.196T.
2. USA_G — US wholesale/retail trade: approx. $1.971T.
3. CHN_G — Chinese wholesale/retail trade: approx. $1.803T.
4. USA_N — US administrative/support activities: approx. $1.442T.
5. USA_K — US finance/insurance: approx. $1.367T.

Highest in-strength is CHN_F (Chinese construction), approx. $3.263T of intermediate inputs purchased. Strength is an input-flow measure, not gross value added or GDP.

## Dynamic network and SIS experiments (counterfactual)

The 50 industries were grouped into 13 broad sectors for the repeated simulations, keeping all 81 regions: 1,053 aggregate nodes. Total flows and gross output are conserved by aggregation. Each SIS scenario ran for 40 steps and 100 Monte Carlo replicates; recovery probability was 0.25 per step.

For the highest out-strength source at 1.4 times the approximate threshold:

- Unweighted SIS: mean peak output-weighted disrupted-node share **5.15%** (SD 4.93%); cumulative share-steps **0.425**.
- Weighted SIS: mean peak share **10.96%** (SD 8.24%); cumulative share-steps **1.894**.

Weighted SIS produces larger disruption in this scenario, but the large run-to-run variation and absent observed outage trajectories mean this does **not** establish greater real-world accuracy.

## Threshold-cascade sensitivity (counterfactual)

From the high out-strength seed, assumed tolerance thresholds 0.02, 0.05 and 0.10 make 1,052 of 1,053 aggregate nodes fail in the model, with 100% of summed gross output represented by failed nodes. At threshold 0.15, 78 nodes fail, representing 26.52% of summed gross output; at 0.30 or higher only the initial source fails, representing 2.56%. This demonstrates extreme dependence on the assumed threshold, not an empirical economic tipping point.

## Equal-budget protection (100 paired Monte Carlo runs)

Source: highest aggregate out-strength node; model transmission = 1.4 times approximate threshold; 105 nodes protected (about 10% of eligible active nodes); protection halves selected nodes' outgoing transmission weights.

| Strategy | Mean peak share (SD) | Mean cumulative share-steps | Relative reduction in cumulative share-steps vs none |
|---|---:|---:|---:|
| No protection | 11.64% (7.87%) | 2.016 | — |
| Random protection | 10.66% (6.97%) | 1.737 | 13.9% |
| High PageRank | 7.49% (4.87%) | 0.881 | 56.3% |
| High out-strength | 7.37% (4.89%) | 0.835 | 58.6% |

Paired high-out-strength protection vs no protection reduced the mean peak by 4.27 percentage points (paired 95% CI 3.18–5.37) and cumulative share-steps by 1.181 (CI 0.943–1.419). These are model-conditional simulation contrasts, not policy impact estimates.

## Central caveat

The ICIO table contains annual monetary flows, not time-stamped disruption events. Thus all dynamic rates and thresholds are assumed. The measure `H(t)` is the fraction of summed gross output assigned to nodes currently in the disrupted state; it is not actual lost output, total welfare loss, or GDP loss.
