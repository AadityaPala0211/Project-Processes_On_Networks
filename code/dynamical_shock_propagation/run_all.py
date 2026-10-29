"""Run full-resolution four-model shock analysis on all 4,050 detailed nodes."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

PACKAGE_DIR=Path(__file__).resolve().parent
CODE_DIR=PACKAGE_DIR.parent
ROOT=PACKAGE_DIR.parents[1]
if str(CODE_DIR) not in sys.path: sys.path.insert(0,str(CODE_DIR))

from dynamical_shock_propagation.config import DEFAULT_CSV,DEFAULT_MATRIX_DIR,DEFAULT_OUTPUT_DIR,Parameters,SensitivityGrid,parameters_dict,MODEL_NAMES
from dynamical_shock_propagation.data_loader import load_network
from dynamical_shock_propagation.metrics import write_csv
from dynamical_shock_propagation.models import UPDATES
from dynamical_shock_propagation.source_census import run_source_census
from dynamical_shock_propagation.sensitivity_analysis import run_sensitivity
from dynamical_shock_propagation.stability_analysis import analyze as analyze_stability,write_result
from dynamical_shock_propagation.intervention_analysis import run_interventions
from dynamical_shock_propagation.visualization import plot_source_impacts,plot_sensitivity,plot_interventions,plot_model_ordering

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",type=Path,default=DEFAULT_CSV)
    ap.add_argument("--matrix-dir",type=Path,default=DEFAULT_MATRIX_DIR)
    ap.add_argument("--out",type=Path,default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--source-mode",choices=("all","top-k","stratified"),default="all")
    ap.add_argument("--top-k",type=int,default=100)
    ap.add_argument("--sensitivity-source-mode",choices=("all","top-k","stratified"),default="stratified")
    ap.add_argument("--sensitivity-top-k",type=int,default=50)
    ap.add_argument("--stratified-per-bin",type=int,default=5)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--edge-chunk-size",type=int,default=500_000)
    ap.add_argument("--intervention-reps",type=int,default=20)
    ap.add_argument("--seed",type=int,default=2022)
    ap.add_argument("--skip-sensitivity",action="store_true")
    ap.add_argument("--skip-interventions",action="store_true")
    args=ap.parse_args()
    if args.batch_size<1 or args.edge_chunk_size<1: ap.error("batch/chunk sizes must be positive")
    out=args.out.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    figures=out/"figures"; figures.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); params=Parameters()
    network=load_network(args.matrix_dir,args.csv,expected_nodes=4050)
    W,P,nodes,output=network["W"],network["P"],network["nodes"],network["output"]
    metadata={"source_csv":network["csv_path"],"source_csv_sha256":network["csv_sha256"],
        "node_count":len(nodes),"matrix_shape":list(P.shape),"structural_edges":int(W.nnz),
        "dependency_matrix_nonzeros":int(P.nnz),
        "orientation":"row supplier i -> column buyer j",
        "dependency_definition":"cross-node supplier flow divided by buyer's total intermediate purchases including self-use",
        "parameters":parameters_dict(params),"sensitivity_grid":parameters_dict(SensitivityGrid()),
        "source_mode":args.source_mode,"sensitivity_source_mode":args.sensitivity_source_mode,
        "full_resolution":True,"signed_substitution_included":False,"leontief_included":False}
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Loaded {len(nodes):,} detailed nodes; {W.nnz:,} positive cross-node flows; P.nnz={P.nnz:,}",flush=True)
    stability=analyze_stability(P,params.beta,params.recovery_rate)
    write_result(out/"stability_diagnostics.json",stability)
    # Verify shared linearization on the actual network using a small random state.
    rng=np.random.default_rng(args.seed); d=rng.uniform(0,1,len(nodes)); order_rows=[]
    model_sequence=("model1_linear","model2_bilinear_sis","model3_mmca_sis","model4_exponential_sis")
    updates=[UPDATES[m](d,P,params.beta,params.recovery_rate) for m in model_sequence]
    for name,arr in zip(model_sequence,updates):
        order_rows.append({"model":name,"one_step_mean_update":float(np.mean(arr)),
                           "one_step_max_update":float(np.max(arr))})
    checks=[bool(np.all(updates[i]>=updates[i+1]-1e-10)) for i in range(3)]
    order_check={"same_state_pointwise_ordering":all(checks),"pairwise_checks":checks,
        "model_order":list(model_sequence),
        "note":"Pointwise inequality at a common state; this does not assert that separated multi-step trajectories remain ordered."}
    (out/"model_ordering_check.json").write_text(json.dumps(order_check,indent=2),encoding="utf-8")
    write_csv(out/"model_ordering_means.csv",order_rows); plot_model_ordering(order_rows,figures)
    if not order_check["same_state_pointwise_ordering"]:
        raise RuntimeError("The numerical pointwise model-ordering check failed.")
    print("[1/3] Full eligible-source baseline census",flush=True)
    census=run_source_census(P,nodes,output,W,args.matrix_dir,out,params,args.source_mode,args.top_k,
                             args.seed,args.batch_size,args.edge_chunk_size)
    plot_source_impacts(out/"source_census.csv",figures)
    sens_meta=None; intervention_meta=None
    if not args.skip_sensitivity:
        print("[2/3] Parameter sensitivity",flush=True)
        sens_meta=run_sensitivity(P,nodes,output,W,args.matrix_dir,out,params,args.sensitivity_source_mode,
                                  args.sensitivity_top_k,args.stratified_per_bin,args.seed,
                                  args.batch_size,args.edge_chunk_size)
        (out/"sensitivity_metadata.json").write_text(json.dumps(sens_meta,indent=2),encoding="utf-8")
        plot_sensitivity(out/"parameter_sensitivity.csv",figures)
    if not args.skip_interventions:
        print("[3/3] Equal-budget interventions",flush=True)
        intervention_meta=run_interventions(P,nodes,output,W,args.matrix_dir,out,params,
                                             reps=args.intervention_reps,seed=args.seed)
        (out/"intervention_metadata.json").write_text(json.dumps(intervention_meta,indent=2),encoding="utf-8")
        plot_interventions(out/"intervention_runs.csv",figures)
    summary={"metadata":metadata,"stability":stability,"model_ordering":order_check,
        "source_census_summary":json.loads((out/"source_census_summary.json").read_text(encoding="utf-8")),
        "sensitivity":sens_meta,"interventions":intervention_meta,
        "elapsed_seconds":time.perf_counter()-started}
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Complete. Results: {out}",flush=True)

if __name__=="__main__": main()
