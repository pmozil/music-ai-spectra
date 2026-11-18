#!/usr/bin/env python3
"""Create class-comparison plots from compare_exact_randomized.py outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir",type=Path,default=Path("artifacts/svd_comparison"))
    ap.add_argument("--output-dir",type=Path,default=Path("artifacts/svd_comparison/plots"))
    a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input_dir/"features_exact_vs_randomized.csv")
    angles=pd.read_csv(a.input_dir/"principal_angles.csv")
    # Distribution plots: class distinction for both algorithms.
    for feature in ("effective_rank","principal_angle_mean_deg","subspace_stability","topk_energy_fraction"):
        fig,ax=plt.subplots(figsize=(8,5))
        for method, col in (("exact",f"exact_{feature}"),("randomized",f"randomized_{feature}")):
            for label,name in ((0,"Human"),(1,"AI-generated")):
                x=df.loc[df.label.eq(label),col].dropna()
                if len(x):
                    ax.hist(x,bins=35,density=True,histtype="step",linewidth=1.8,
                            label=f"{name} — {method}")
        ax.set(title=feature.replace("_"," ").title(),xlabel=feature.replace("_"," "),ylabel="Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.output_dir/f"{feature}_distributions.png",dpi=180)
        plt.close(fig)
    # Direct exact vs randomized agreement.
    for feature in ("effective_rank","principal_angle_mean_deg","subspace_stability"):
        fig,ax=plt.subplots(figsize=(6,6))
        for label,name in ((0,"Human"),(1,"AI-generated")):
            sub=df[df.label.eq(label)]
            ax.scatter(sub[f"exact_{feature}"],sub[f"randomized_{feature}"],
                       s=12,alpha=.45,label=name)
        ax.set(xlabel=f"Exact {feature}",ylabel=f"Randomized {feature}",
               title=f"Exact vs randomized: {feature.replace('_',' ')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.output_dir/f"{feature}_exact_vs_randomized.png",dpi=180)
        plt.close(fig)
    # Raw principal-angle distributions split by method and class.
    fig,ax=plt.subplots(figsize=(9,5))
    groups=[]
    labels=[]
    for method in ("exact","randomized"):
        for label,name in ((0,"Human"),(1,"AI")):
            groups.append(angles.loc[(angles.method==method)&(angles.label==label),"angle_deg"].dropna().values)
            labels.append(f"{name}\n{method}")
    ax.boxplot(groups,tick_labels=labels,showfliers=False)
    ax.set(ylabel="Principal angle (degrees)",title="Adjacent-window subspace angles")
    fig.tight_layout()
    fig.savefig(a.output_dir/"principal_angle_boxplots.png",dpi=180)
    plt.close(fig)
    print(f"Plots saved to {a.output_dir}")

if __name__=="__main__":
    main()
