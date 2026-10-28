# Signed Shock Propagation in a Global Production Network

*An undergraduate network-dynamics model with negative and positive effects — October 2026*

## The idea

A supplier outage harms buyers that depend on it, while competing suppliers may gain orders. The earlier SIS-like model captured spreading disruption but did not explicitly represent substitution. The revised model tracks a **signed state**: negative values mean adverse pressure; positive values mean an opportunity signal for a plausible alternative supplier.

## Data and baseline context

The project uses the regular OECD ICIO 2025 edition for reference year 2022. It contains 80 named economies plus Rest of World (ROW) and 50 detailed activities: 4,050 country-industry nodes. Checked-in baseline results report 10,166,696 positive off-diagonal flows (62.0% raw density), US$103.410 trillion in total positive intermediate flows including within-node flows, US$81.128 trillion in off-diagonal flows, and out-strength Gini 0.863. The original raw CSV is not bundled; the baseline checksum is recorded in \`README.md\`.

Targets are ranked by outgoing intermediate-flow strength, not GDP or welfare:

| Scenario | Representative target | Baseline selection evidence |
|---|---|---|
| Highest-flow activity | \`USA_G\` — wholesale/retail trade | Activity G totals US$9.291T of outgoing intermediate flows; USA_G contributes US$1.971T |
| Lowest positive activity | \`CHN_C301\` — building of ships and boats | Activity C301 totals US$83.3B; CHN_C301 contributes US$22.9B |
| Oil illustration | \`RUS_B06\` — crude petroleum and natural gas extraction | Node out-strength about US$165.5B in stored baseline |
| Sensitivity check | \`STP_C302T309\` | Weakest active node by recorded out-strength; its flow is extremely small, so it is not the main least-industry choice |

Household own-use activity \`T\` has zero supplier flows in the stored baseline and is excluded from the low-industry choice.

## The model

Let \`Z[i,j]\` be intermediate sales from supplier \`i\` to buyer \`j\`. Define

\`D[i,j] = Z_cross[i,j] / sum_k Z_all[k,j]\`.

The numerator excludes self-links, but the denominator includes the buyer's full intermediate-input total. Thus each buyer's column of \`D\` represents shares of purchased inputs from suppliers.

Let \`x[i,t]\` be a state in \`[-1, +1]\`. Negative values represent adverse pressure, positive values represent substitution opportunity, and zero means no modeled deviation. The target starts at \`-severity\`.

The substitute matrix \`C[k,i]\` allocates opportunity from disrupted supplier \`i\` to alternative supplier \`k\` in the *same detailed ICIO activity*. Its weights are based on cosine similarity between observed supplier-to-buyer flow patterns. If none overlap, outgoing flow strength is used as a fallback. It is a proxy for potential competition; it does not measure spare capacity.

\`x(t+1) = clip(lambda*x(t) + beta*D.T*x(t) + sigma*C*max(-x(t),0), -1, +1)\`

Defaults: \`lambda=0.55\`, \`beta=0.35\`, \`sigma=0.25\`, and 12 model steps.

- **Persistence:** \`lambda*x(t)\` lets pressure fade rather than disappear instantly.
- **Dependency propagation:** \`beta*D.T*x(t)\` passes the sign of a supplier's state to its buyers, weighted by their input reliance.
- **Substitution opportunity:** \`sigma*C*max(-x(t),0)\` gives alternative suppliers a positive opportunity signal when a source is disrupted.
- **Boundedness:** clipping holds the state in \`[-1,1]\`.

The dependency-only linear part has Jacobian \`J=lambda*I + beta*D.T\`. Since the row sums of \`D.T\` are at most one, \`lambda+beta=0.90<1\` is a sufficient contraction bound under these assumptions. The substitution term is piecewise nonlinear, so this is a bounded teaching stress test, not a calibrated market-equilibrium model.

For gross output \`X[i]\`, the script reports separate indices:

\`N(t) = sum_i X[i]*max(-x[i,t],0) / sum_i X[i]\`

\`P(t) = sum_i X[i]*max( x[i,t],0) / sum_i X[i]\`

\`N\` is an adverse-pressure index and \`P\` is an opportunity index. Neither is actual lost output, GDP loss, or extra exports. Model steps are not days or months.

## What the run does

The script uses all detailed country-industry nodes for dynamic scenarios. It selects the highest-flow global activity and lowest positive-flow activity (excluding \`T\`), chooses a representative country node within each, and adds \`RUS_B06\` when present. Each target is run at full and half initial severity. It writes rankings, the scenario summary, per-step series, per-node effects, model metadata, and two figures.

For a readable presentation example, the script also exports a subnetwork with 12 economies and 13 broad sectors (up to 156 country-sector nodes): \`worked_example_nodes.csv\` and \`worked_example_edges.csv\`. Those files help explain paths by hand; the signed simulation uses all 4,050 detailed nodes.

## Course connections

- **Network dynamics / linear approximation:** understand edge direction, the input-share matrix, the signed state, and Jacobian \`J\`.
- **SIS/SIR:** the old SIS analogy has binary disrupted/operational states and recovery; the new state is continuous and signed. SIS is useful when nodes can recover and fail again.
- **Bifurcation / spectral threshold:** vary \`beta\` and track the spectral radius of \`J\`. This revision does not claim to detect a saddle-node, transcritical, or pitchfork bifurcation.
- **Synchronization:** a follow-up could compare node trajectories or cross-node variance to test whether a shared shock makes sector states move together. This is not claimed as a result here.

## Interpretation and limits

Annual ICIO flows do not identify outage timing, inventories, spare capacity, elasticities, prices, or the speed of switching suppliers. A positive opportunity signal does not guarantee that a country can increase production. The oil activity \`B06\` combines crude petroleum and natural gas. Since some economies may be included in ROW rather than named separately, an opportunity for ROW cannot be attributed to Venezuela without a separate country-specific data source.

## Reproducibility and validation status

From the repository root, run on the exact raw data file:

\`\`\`bash
python code/signed_shock_propagation.py /path/to/2022_SML.csv --out results_signed --steps 12 --severity 1.0
python -m unittest discover -s tests -v
\`\`\`

Four unit tests pass on a synthetic network. They check the input-share weights, same-industry substitution links, that both negative and positive states can occur, and the initial shock severity. The raw ICIO CSV was unavailable in the connected repository when this revision was prepared, so the signed scenario numerical tables must be generated locally before being described as dataset-based findings. No signed-simulation values are fabricated in this report.

Data description: [OECD Inter-Country Input-Output Tables](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html), 2025 edition, regular table.
