#!/usr/bin/env python3
"""Generate final figures from computed CSV/JSON outputs; never reads screenshots.

Example:
  python generate_refined_figures.py --results results_2022 --figures results_2022
The base analysis generates the remaining structural plots (degree, top strength,
PageRank, sector heatmap, and source-impact scatter).
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,required=True,help='Output folder from base and refined analysis scripts')
    ap.add_argument('--figures',type=Path,required=True,help='Destination for regenerated report figures')
    args=ap.parse_args(); R=args.results.resolve(); F=args.figures.resolve(); F.mkdir(parents=True,exist_ok=True)
    # Pipeline schematic
    fig,ax=plt.subplots(figsize=(10.4,2.5));ax.axis('off')
    blocks=[('OECD ICIO 2022','4,050 country-industry nodes'),('Flow matrix Z','Supplier i → buyer j'),('Structure','Strength, degree, PageRank'),('Dynamics','Weighted SIS / threshold'),('Evaluation','Cascade, sensitivity, recovery')]
    xs=np.linspace(.11,.89,len(blocks))
    for i,((title,sub),x) in enumerate(zip(blocks,xs)):
        ax.text(x,.58,title,ha='center',va='center',fontsize=9.5,fontweight='bold',bbox=dict(boxstyle='round,pad=.5',facecolor='#eff4f8',edgecolor='#34546d',linewidth=1.2),transform=ax.transAxes)
        ax.text(x,.31,sub,ha='center',va='center',fontsize=7.2,wrap=True,transform=ax.transAxes)
        if i<len(blocks)-1: ax.annotate('',xy=(xs[i+1]-.065,.58),xytext=(x+.065,.58),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=1.4,color='#34546d'))
    fig.tight_layout();fig.savefig(F/'analysis_pipeline.png',dpi=230,bbox_inches='tight');plt.close(fig)
    # Edge threshold sensitivity
    r=read_csv(R/'edge_threshold_sensitivity.csv'); threshold=np.array([float(x['min_flow_million_USD']) for x in r]);density=np.array([100*float(x['density']) for x in r]);xx=threshold.copy();xx[0]=.001
    fig,ax=plt.subplots(figsize=(7.1,3.7));ax.plot(xx,density,marker='o',lw=1.8)
    labs=['Any >0','$10k','$100k','$1m','$10m','$100m']
    for i,(x,y,lab) in enumerate(zip(xx,density,labs)):ax.annotate(f'{y:.1f}%\n{lab}',(x,y),xytext=(0,7 if i not in(0,5) else -27),textcoords='offset points',ha='center',fontsize=7)
    ax.set_xscale('log');ax.set_xlabel('Minimum retained flow (current USD; logarithmic position)');ax.set_ylabel('Directed density (%)');ax.set_title('Binary network density is sensitive to flow threshold');ax.grid(True,which='both',alpha=.25);fig.tight_layout();fig.savefig(F/'edge_threshold_sensitivity.png',dpi=230);plt.close(fig)
    # SIS comparison
    r=read_csv(R/'sis_core_100rep_summary.csv');fig,ax=plt.subplots(figsize=(7.1,3.8))
    for key,label in [('unweighted_SIS','Unweighted SIS'),('weighted_SIS','Dependency-weighted SIS')]:
        sub=sorted([q for q in r if q['source_type']=='top_out_strength' and q['model']==key],key=lambda q:float(q['beta_over_beta_critical']))
        ax.errorbar([float(q['beta_over_beta_critical']) for q in sub],[100*float(q['peak_output_weighted_disruption_mean']) for q in sub],yerr=[100*float(q['peak_output_weighted_disruption_sd']) for q in sub],marker='o',capsize=2.5,lw=1.5,label=label)
    ax.set_xlabel(r'Transmission rate relative to approximate threshold, $\beta/\beta_c$');ax.set_ylabel('Peak output-weighted disrupted-node share (%)');ax.set_title('SIS comparison: highest out-strength source');ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(F/'sis_model_comparison.png',dpi=230);plt.close(fig)
    # Interventions
    r=read_csv(R/'intervention_summary_100rep.csv');order=['none','random_protection','top_PageRank','top_out_strength'];labels=['No protection','Random','Top PageRank','Top out-strength'];look={q['strategy']:q for q in r}
    fig,ax=plt.subplots(figsize=(7.1,3.8));means=[100*float(look[k]['peak_output_weighted_disruption_mean']) for k in order];sds=[100*float(look[k]['peak_output_weighted_disruption_sd']) for k in order]
    ax.bar(labels,means,yerr=sds,capsize=3);ax.set_ylabel('Mean peak output-weighted disrupted share (%)');ax.set_title('Protection strategies: 100 paired Monte Carlo runs');ax.tick_params(axis='x',rotation=12);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(F/'intervention_comparison.png',dpi=230);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,3.8));means=[float(look[k]['cumulative_output_weighted_disruption_mean']) for k in order];sds=[float(look[k]['cumulative_output_weighted_disruption_sd']) for k in order]
    ax.bar(labels,means,yerr=sds,capsize=3);ax.set_ylabel('Cumulative disrupted share (share-steps)');ax.set_title('Cumulative disruption: 100 paired Monte Carlo runs');ax.tick_params(axis='x',rotation=12);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(F/'intervention_cumulative.png',dpi=230);plt.close(fig)
    # Threshold cascade
    r=read_csv(R/'threshold_cascade_runs.csv');fig,ax=plt.subplots(figsize=(7.1,3.8))
    for key,label in [('top_out_strength','Highest out-strength source'),('top_page_rank','Highest PageRank source'),('random','Mean across random seeds')]:
        sub=[q for q in r if q['source_type']==key];by={}
        for q in sub:by.setdefault(float(q['theta']),[]).append(float(q['output_weighted_cascade_fraction']))
        xx=sorted(by);yy=[100*np.mean(by[x]) for x in xx];ax.plot(xx,yy,marker='o',lw=1.5,label=label)
    ax.set_xlabel(r'Assumed failure threshold, $\theta$');ax.set_ylabel('Gross-output-weighted failed-node share (%)');ax.set_title('Threshold cascade is highly sensitive to tolerance');ax.set_ylim(-2,102);ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=7);fig.tight_layout();fig.savefig(F/'threshold_cascade_sensitivity.png',dpi=230);plt.close(fig)
    # Grouped sector-flow chart (total out-/in-strength across countries)
    r=read_csv(R/'sector_flow_summary.csv');names=[q['country_or_sector_group'] for q in r]
    out=np.array([float(q['out_strength_million_USD'])/1e6 for q in r]);inn=np.array([float(q['in_strength_million_USD'])/1e6 for q in r])
    order=np.argsort(out+inn);fig,ax=plt.subplots(figsize=(8.2,5.5));y=np.arange(len(names));ax.barh(y-.18,out[order],height=.34,label='Out-strength: flows supplied');ax.barh(y+.18,inn[order],height=.34,label='In-strength: inputs purchased')
    ax.set_yticks(y);ax.set_yticklabels(np.array(names)[order],fontsize=8);ax.set_xlabel('Total intermediate flow (trillion current USD)');ax.set_title('Supply and purchase flows by aggregated industry group');ax.grid(axis='x',alpha=.25);ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(F/'sector_group_flow.png',dpi=230);plt.close(fig)
    # Final machine-readable summary links observed data to the refined model outputs.
    structural=json.loads((R/'structural_summary.json').read_text(encoding='utf-8'))
    source_corr=json.loads((R/'source_impact_correlations.json').read_text(encoding='utf-8'))
    metadata=json.loads((R/'refined_model_metadata.json').read_text(encoding='utf-8')) if (R/'refined_model_metadata.json').exists() else {}
    summary={'structural':structural,'refined_model_metadata':metadata,'sis_summary':read_csv(R/'sis_core_100rep_summary.csv'),
             'intervention_summary':read_csv(R/'intervention_summary_100rep.csv'),'paired_intervention_tests':read_csv(R/'paired_intervention_tests.csv'),
             'source_impact_correlations_exploratory':source_corr,
             'threshold_cascade_runs':read_csv(R/'threshold_cascade_runs.csv'),
             'interpretation':'ICIO observations are annual monetary flows. SIS and threshold-cascade results are counterfactual scenarios under assumed parameters; H(t) represents gross-output-weighted disrupted-node share, not observed lost output or GDP.'}
    (F/'refined_results_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Wrote refined figures and summary to',F)

if __name__=='__main__':main()
