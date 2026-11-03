"""独立绘图进程：避免Windows中Torch和SciPy加载不同OpenMP运行库。"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from common import read_config,resolve,metrics,TARGETS,BINS

def density(ax,x,color,label):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if len(x)>1 and x.std()>1e-10:
        lo,hi=np.quantile(x,[.005,.995]);grid=np.linspace(lo,hi,240)
        ax.plot(grid,gaussian_kde(x)(grid),color=color,label=label,lw=1.5)
    else:ax.axvline(float(np.mean(x)),color=color,label=label)

def make_figures(config,meta,seed,evaldir):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    colors=["#2563eb","#06b6d4","#64748b","#f59e0b","#dc2626"]
    real=np.load(resolve(config["data"]["processed"])/"test_metrics.npy",mmap_mode="r")
    for method in ["continuous","discrete"]:
        fig,axes=plt.subplots(2,3,figsize=(15,7));volfig,volaxes=plt.subplots(1,3,figsize=(15,3.6))
        for j,target in enumerate(TARGETS):
            for k,name in enumerate(BINS):
                file=evaldir/f"samples_{target}_{method}_{name}.npz"
                with np.load(file) as z:s=z["states"]
                vals=metrics(s[:,0,:])[:,j]
                if k in [0,2,4]:
                    prices=np.concatenate([np.ones((len(s),1)),np.exp(s[:,0,:].cumsum(axis=1))],axis=1)*100
                    for curve in prices[:15]:axes[0,j].plot(curve,color=colors[k],alpha=.17,lw=.8)
                    axes[0,j].plot([],[],color=colors[k],label=name)
                    vols=np.expm1(s[:,1,:].astype(np.float64))
                    volaxes[j].plot(np.median(vols,axis=0),color=colors[k],label=name)
                density(axes[1,j],vals,colors[k],name)
            density(axes[1,j],real[:,j],"#171717","Real test")
            axes[0,j].set(title=target.capitalize(),xlabel="Trading day",ylabel="Price index (initial=100)")
            axes[1,j].set(xlabel=f"{target} (log-return units)",ylabel="Density")
            volaxes[j].set(title=target.capitalize(),xlabel="Trading day",ylabel="Median generated daily volume")
            axes[0,j].legend(fontsize=8);axes[1,j].legend(fontsize=8);volaxes[j].legend(fontsize=8)
        fig.suptitle(f"Daily60: {method} control, seed {seed}; {config['evaluation']['test_samples']} samples/bin")
        fig.tight_layout();volfig.tight_layout()
        for ext in ["png","svg"]:
            fig.savefig(evaldir/f"figure3_{method}.{ext}",dpi=160)
            volfig.savefig(evaldir/f"volume_{method}.{ext}",dpi=160)
        plt.close(fig);plt.close(volfig)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--seed",type=int,default=0);a=p.parse_args()
    c=read_config(a.config);folder=resolve(c["output"])/f"seed_{a.seed}"/"evaluation"
    make_figures(c,None,a.seed,folder)
