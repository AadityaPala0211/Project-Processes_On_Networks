#!/usr/bin/env python3
"""Run refined 100-replicate SIS and equal-budget intervention experiments.

Requires supply_chain_complete_analysis.py in the same directory.
Input is the original OECD 2022_SML.csv. The data are not bundled.
"""
from __future__ import annotations
import argparse, csv, json, importlib.util, sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import t as student_t, ttest_rel

HERE = Path(__file__).resolve().parent

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, type=Path, help='Path to OECD ICIO 2022_SML.csv')
    ap.add_argument('--out', type=Path, default=Path('results_2022'), help='Directory produced by base analysis')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=2022)
    args=ap.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    sp=importlib.util.spec_from_file_location('scanalysis', HERE/'supply_chain_complete_analysis.py')
    if sp is None or sp.loader is None: raise RuntimeError('Could not load supply_chain_complete_analysis.py')
    sc=importlib.util.module_from_spec(sp); sys.modules[sp.name]=sc; sp.loader.exec_module(sc)

    print('Loading 2022 ICIO CSV...', flush=True)
    _,nodes,Z,X=sc.load_icio(args.csv)
    countries,agg_nodes,mapidx,Za,Xa,S=sc.make_aggregate(Z,nodes,X)
    Za=Za.maximum(0).tocsr(); Za.eliminate_zeros()
    Za_no_diag=Za.copy(); Za_no_diag.setdiag(0); Za_no_diag.eliminate_zeros()
    A=(Za_no_diag>0).astype(float).tocsr(); A.eliminate_zeros()
    outstr=np.asarray(Za_no_diag.sum(axis=1)).ravel()
    instr=np.asarray(Za_no_diag.sum(axis=0)).ravel()
    pr,_=sc.weighted_pagerank(Za_no_diag)
    Tech=(Za_no_diag @ sparse.diags(1/np.maximum(Xa,1e-12))).tocsr(); Tech.eliminate_zeros()
    in_deg=np.asarray(A.sum(axis=0)).ravel(); inv=np.zeros(len(agg_nodes)); inv[in_deg>0]=1/in_deg[in_deg>0]
    U=(A @ sparse.diags(inv)).tocsr()
    rho_t=sc.spectral_radius(Tech.T.tocsr()); rho_u=sc.spectral_radius(U.T.tocsr()); gamma=.25
    bc_t=gamma/rho_t; bc_u=gamma/rho_u
    src_os=int(np.argmax(outstr)); src_pr=int(np.argmax(pr))
    active=np.where((outstr+instr>0)&(np.maximum(Xa,0)>0))[0]
    print(f'Aggregated nodes={len(agg_nodes)}, weighted nnz={Tech.nnz}, rho(W)={rho_t:.6g}, beta_c={bc_t:.6g}', flush=True)

    # Main SIS comparisons; a shared initial condition and controlled seed schedule are used.
    sis_rows=[]
    scenarios=[('unweighted_SIS',U,bc_u,'top_out_strength',src_os,[.7,1.0,1.4]),
               ('weighted_SIS',Tech,bc_t,'top_out_strength',src_os,[.7,1.0,1.4]),
               ('weighted_SIS',Tech,bc_t,'top_page_rank',src_pr,[1.4])]
    for model,W,bc,srcname,source,ratios in scenarios:
        for ratio in ratios:
            beta=bc*ratio
            print('SIS',model,srcname,'beta/beta_c=',ratio,flush=True)
            for rep in range(args.reps):
                model_offset=0 if model=='unweighted_SIS' else 100000
                source_offset=0 if srcname=='top_out_strength' else 200000
                rng=np.random.default_rng(args.seed+500000+model_offset+source_offset+int(ratio*1000)*1000+rep)
                result=sc.run_sis(W,Xa,source,beta,gamma,args.steps,1,rng)[0]
                sis_rows.append({'model':model,'source_type':srcname,'source_node':agg_nodes[source],
                    'beta_over_beta_critical':ratio,'beta':beta,'gamma':gamma,**result,'replicate':rep})
    write_csv(out/'sis_core_100rep.csv',sis_rows)
    measures=['peak_output_weighted_disruption','cumulative_output_weighted_disruption','final_output_weighted_disruption','peak_node_fraction']
    buckets={}
    for r in sis_rows: buckets.setdefault((r['model'],r['source_type'],r['beta_over_beta_critical']),[]).append(r)
    sis_summary=[]
    for (model,source_type,ratio),rr in buckets.items():
        row={'model':model,'source_type':source_type,'beta_over_beta_critical':ratio,'n_runs':len(rr)}
        for m in measures:
            vals=np.array([float(q[m]) for q in rr]); row[m+'_mean']=float(vals.mean()); row[m+'_sd']=float(vals.std(ddof=1)); row[m+'_median']=float(np.median(vals))
        sis_summary.append(row)
    write_csv(out/'sis_core_100rep_summary.csv',sis_summary)

    # Equal-budget paired intervention comparison. The seed source is not eligible for protection.
    eligible=np.asarray([i for i in active if i!=src_os],dtype=int); eligible_set=set(eligible.tolist())
    budget=max(1,int(.10*len(eligible)))
    os_order=np.asarray([i for i in np.argsort(outstr)[::-1] if int(i) in eligible_set][:budget],dtype=int)
    pr_order=np.asarray([i for i in np.argsort(pr)[::-1] if int(i) in eligible_set][:budget],dtype=int)
    strategies=[('none',None),('top_out_strength',os_order),('top_PageRank',pr_order),('random_protection',None)]
    int_rows=[]; beta=bc_t*1.4
    for rep in range(args.reps):
        prng=np.random.default_rng(args.seed+700000+rep)
        random_set=prng.choice(eligible,size=min(budget,len(eligible)),replace=False)
        for name,protect in strategies:
            chosen=random_set if name=='random_protection' else protect
            srng=np.random.default_rng(args.seed+800000+rep)
            result=sc.run_sis(Tech,Xa,src_os,beta,gamma,args.steps,1,srng,protected=chosen,protection_factor=.5)[0]
            int_rows.append({'strategy':name,'replicate':rep,'protection_budget_nodes':0 if name=='none' else len(chosen),
                'source_node':agg_nodes[src_os],'beta':beta,'gamma':gamma,**result})
    write_csv(out/'intervention_runs_100rep.csv',int_rows)
    int_summary=[]
    for name,_ in strategies:
        rr=[q for q in int_rows if q['strategy']==name]
        row={'strategy':name,'n_runs':len(rr)}
        for m in ['peak_output_weighted_disruption','cumulative_output_weighted_disruption','final_output_weighted_disruption']:
            vals=np.array([float(q[m]) for q in rr]); row[m+'_mean']=float(vals.mean()); row[m+'_sd']=float(vals.std(ddof=1)); row[m+'_median']=float(np.median(vals))
        int_summary.append(row)
    write_csv(out/'intervention_summary_100rep.csv',int_summary)

    # Paired contrasts: baseline minus each strategy for matched replicate IDs.
    by={(q['strategy'],int(q['replicate'])):q for q in int_rows}
    paired=[]
    for strategy,_ in strategies[1:]:
        for metric in ['peak_output_weighted_disruption','cumulative_output_weighted_disruption','final_output_weighted_disruption']:
            diff=np.array([float(by[('none',r)][metric])-float(by[(strategy,r)][metric]) for r in range(args.reps)])
            mean=float(diff.mean()); sd=float(diff.std(ddof=1)); se=sd/np.sqrt(len(diff)); crit=float(student_t.ppf(.975,len(diff)-1))
            lo,hi=mean-crit*se,mean+crit*se
            test=ttest_rel([float(by[('none',r)][metric]) for r in range(args.reps)],
                           [float(by[(strategy,r)][metric]) for r in range(args.reps)])
            baseline=np.mean([float(by[('none',r)][metric]) for r in range(args.reps)])
            paired.append({'metric':metric,'strategy':strategy,'n_pairs':args.reps,'baseline_mean':float(baseline),
              'strategy_mean':float(np.mean([float(by[(strategy,r)][metric]) for r in range(args.reps)])),
              'mean_absolute_reduction':mean,'relative_reduction_vs_baseline':mean/baseline if baseline else None,
              'paired_difference_ci95_low':lo,'paired_difference_ci95_high':hi,'paired_t_pvalue':float(test.pvalue)})
    write_csv(out/'paired_intervention_tests.csv',paired)
    meta={'raw_csv':str(args.csv.resolve()),'raw_nodes':len(nodes),'aggregate_nodes':len(agg_nodes),'aggregate_regions':len(countries),
          'sector_groups':S,'steps':args.steps,'repetitions':args.reps,'seed':args.seed,'gamma':gamma,
          'weighted_spectral_radius':float(rho_t),'weighted_beta_c_approx':float(bc_t),
          'unweighted_spectral_radius':float(rho_u),'unweighted_beta_c_approx':float(bc_u),
          'intervention_budget_nodes':budget,'protected_outgoing_transmission_factor':.5,
          'interpretation':'Counterfactual stochastic simulations, not fitted predictions or observed economic losses.'}
    (out/'refined_model_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Wrote refined SIS, intervention, and paired-contrast outputs to',out,flush=True)
    for r in int_summary:
        print(r['strategy'], 'peak=',round(100*r['peak_output_weighted_disruption_mean'],3),'%',
              'cumulative=',round(r['cumulative_output_weighted_disruption_mean'],3),flush=True)

if __name__=='__main__': main()
