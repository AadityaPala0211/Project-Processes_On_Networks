#!/usr/bin/env python3
"""Run the full-data structural-analysis pipeline in dependency order.

Example:
  python code/full_data_network_structure_analysis/run_full_structural_analysis.py \
      data/2022_SML.csv --out results_full_network

The source CSV is not modified. Generated matrix/result files are written only
under the selected output directory.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to OECD 2022_SML.csv")
    parser.add_argument("--out", type=Path, default=Path("results_full_network"))
    parser.add_argument("--topn", type=int, default=30)
    parser.add_argument("--sample-sources", type=int, default=64)
    parser.add_argument("--betweenness-sources", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--top-edges", type=int, default=120)
    parser.add_argument("--clustering-sample-nodes", type=int, default=100,
                        help="Sample size for local clustering; ignored with --all-clustering-nodes")
    parser.add_argument("--all-clustering-nodes", action="store_true",
                        help="Calculate exact local clustering for every node")
    parser.add_argument("--include-self-loops", action="store_true")
    parser.add_argument("--skip-centrality", action="store_true",
                        help="Skip the centrality stage (e.g. to run it later separately)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.out.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    stages = [
        ("01_build_adjacency_matrix.py", [str(csv_path), "--out", str(output_dir)] +
         (["--include-self-loops"] if args.include_self_loops else [])),
        ("02_basic_network_metrics.py", ["--matrix-dir", str(output_dir),
                                         "--out", str(output_dir), "--topn", str(args.topn)]),
        ("03_connectivity_and_paths.py", ["--matrix-dir", str(output_dir),
                                          "--out", str(output_dir),
                                          "--sample-sources", str(args.sample_sources),
                                          "--seed", str(args.seed)]),
        ("07_clustering_analysis.py", ["--matrix-dir", str(output_dir),
                                       "--out", str(output_dir),
                                       "--sample-nodes", str(args.clustering_sample_nodes),
                                       "--seed", str(args.seed)] +
         (["--all-nodes"] if args.all_clustering_nodes else [])),
    ]
    if not args.skip_centrality:
        stages.append(("04_centrality_analysis.py", ["--matrix-dir", str(output_dir),
                                                      "--out", str(output_dir),
                                                      "--betweenness-sources", str(args.betweenness_sources),
                                                      "--seed", str(args.seed),
                                                      "--topn", str(args.topn)]))
    stages.extend([
        ("05_edge_threshold_sensitivity.py", ["--matrix-dir", str(output_dir),
                                              "--out", str(output_dir)]),
        ("06_flow_aggregation_and_visualization.py", ["--matrix-dir", str(output_dir),
                                                       "--out", str(output_dir),
                                                       "--top-nodes", str(args.topn),
                                                       "--top-edges", str(args.top_edges)]),
    ])

    for number, (script_name, script_args) in enumerate(stages, start=1):
        command = [sys.executable, str(script_dir / script_name), *script_args]
        print(f"\n[{number}/{len(stages)}] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)

    print("\nPipeline completed.")
    print(f"Outputs: {output_dir}")
    print("Generated results are reproducible derivatives. Do not commit the large NPZ matrices unless needed.")


if __name__ == "__main__":
    main()
