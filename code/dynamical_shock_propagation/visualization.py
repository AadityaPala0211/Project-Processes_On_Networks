"""Compact figures for source impacts, parameter sweeps, and interventions."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def plot_source_impacts(path,outdir):
    rows=read_csv(path)
    for metric in ("peak_output_weighted_disruption","cumulative_disruption_share_steps"):
        fig,ax=plt.subplots(figsize=(9,5))
        for model in sorted({r["model"] for r in rows}):
            vals=sorted([float(r[metric]) for r in rows if r["model"]==model],reverse=True)[:50]
            ax.plot(np.arange(1,len(vals)+1),vals,marker=".",linewidth=1.3,label=model)
        ax.set_xlabel("Source rank (top 50)"); ax.set_ylabel(metric.replace("_"," "))
        ax.set_title("Source-shock ranking: "+metric.replace("_"," "))
        ax.grid(alpha=.25); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(outdir/f"top_source_{metric}.png",dpi=180); plt.close(fig)

def plot_sensitivity(path,outdir):
    rows=read_csv(path)
    if not rows: return
    for model in sorted({r["model"] for r in rows}):
        sub=[r for r in rows if r["model"]==model]
        fig,ax=plt.subplots(figsize=(7.5,4.8))
        pairs=sorted({(float(r["beta"]),float(r["recovery_rate"])) for r in sub})
        for beta,rho in pairs:
            rr=sorted([r for r in sub if float(r["beta"])==beta and float(r["recovery_rate"])==rho],
                      key=lambda r:float(r["initial_disruption"]))
            grouped=[]
            for severity in sorted({float(r["initial_disruption"]) for r in rr}):
                vals=[float(r["peak_output_weighted_disruption"]) for r in rr if float(r["initial_disruption"])==severity]
                grouped.append((severity,float(np.mean(vals))))
            if grouped:
                ax.plot([x[0] for x in grouped],[x[1] for x in grouped],marker="o",label=f"beta={beta:g}, rho={rho:g}")
        ax.set_xlabel("Initial source severity"); ax.set_ylabel("Peak output-weighted index")
        ax.set_title("Parameter sensitivity — "+model); ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=2)
        fig.tight_layout(); fig.savefig(outdir/f"sensitivity_{model}.png",dpi=180); plt.close(fig)

def plot_interventions(path,outdir):
    rows=read_csv(path)
    if not rows: return
    for model in sorted({r["model"] for r in rows}):
        sub=[r for r in rows if r["model"]==model]
        strategies=sorted({r["strategy"] for r in sub})
        means=[float(np.mean([float(r["cumulative_disruption_share_steps"]) for r in sub if r["strategy"]==s])) for s in strategies]
        fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(strategies,means)
        ax.set_title("Equal-budget intervention — "+model); ax.set_ylabel("Cumulative disruption index")
        ax.tick_params(axis="x",rotation=25); ax.grid(axis="y",alpha=.25); fig.tight_layout()
        fig.savefig(outdir/f"interventions_{model}.png",dpi=180); plt.close(fig)

def plot_model_ordering(rows,outdir):
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar([r["model"] for r in rows],[float(r["one_step_mean_update"]) for r in rows])
    ax.set_ylabel("Mean updated state (same initial vector)")
    ax.set_title("Pointwise one-step model comparison")
    ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.25); fig.tight_layout()
    fig.savefig(outdir/"one_step_model_ordering.png",dpi=180); plt.close(fig)
