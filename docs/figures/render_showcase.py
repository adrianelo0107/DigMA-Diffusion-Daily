"""Render curated, report-backed figures from aggregate metrics only.
No raw observations, generated sequences or model weights are embedded here.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFERENCE = {'sampling_ablation': {'clip6': {'LogV': [0.411585998261442, 0.9949092046123388, 0.6905136609102683], 'DayR': [0.0688975104722836, 0.0864861328552695, 0.1807325122908799], 'P99_ratio': [20.77030116303428, 41.59805708483351, 38.66940134414357]}, 'volume3_return6': {'LogV': [0.1077834626870057, 0.4395042664636037, 0.2491176962015481], 'DayR': [0.0582588620727225, 0.0629963727061524, 0.1627753775013487], 'P99_ratio': [1.73388003698763, 2.070817431173585, 2.047336444843671]}}, 'volatility_control': {'0': {'target': [0.0150395883598284, 0.0208308024617064, 0.0257176053678194, 0.0315686212277609, 0.0424319294117247], 'generated_mean': [0.0145949290192191, 0.0206672805537028, 0.0255487426295515, 0.0311614221312346, 0.0422132757191743], 'paths_each_level': [16196, 16196, 16196, 16196, 16196]}, '1': {'target': [0.0150395883598284, 0.0208308024617064, 0.0257176053678194, 0.0315686212277609, 0.0424319294117247], 'generated_mean': [0.0159124125636187, 0.0222805757212513, 0.0274591812092488, 0.0332998980770072, 0.044252328930962], 'paths_each_level': [16196, 16196, 16196, 16196, 16196]}, '2': {'target': [0.0150395883598284, 0.0208308024617064, 0.0257176053678194, 0.0315686212277609, 0.0424319294117247], 'generated_mean': [0.0142333108315413, 0.0200849116049019, 0.0252202576249648, 0.0309583989426794, 0.040896936286776], 'paths_each_level': [16196, 16196, 16196, 16196, 16196]}}, 'history_control': {'return': {'no_control': [0.2316564883799071, 0.3217278109076463, 0.1892430371173822], 'discrete': [0.0730679603403457, 0.0551923969213182, 0.0865752841319777], 'continuous': [0.0072687552076347, 0.0092620183770571, 0.0099991350505169]}, 'amplitude': {'no_control': [0.149355331343385, 0.2707402290300648, 0.1705253359953312], 'discrete': [0.0484696575368444, 0.0563775656935581, 0.0950519304224084], 'continuous': [0.0110090293448893, 0.0146548637336846, 0.0102177060745556]}, 'volatility': {'no_control': [0.0006749471109442, 0.000852610219151, 0.0010543709084821], 'discrete': [0.0001609774410364, 0.0001058405021846, 5.699298533912653e-05], 'continuous': [7.168700837511567e-06, 6.3862445036934296e-06, 5.856623166114473e-06]}}}


def main():
    dest=Path(__file__).resolve().parent
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.labelcolor":"#334155","text.color":"#0f172a","axes.titleweight":"bold","figure.facecolor":"white"})
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.4))
    titles={"LogV":"Log-volume KL","DayR":"Daily-return KL","P99_ratio":"Volume P99 / real P99"}
    for ax,key in zip(axes,["LogV","DayR","P99_ratio"]):
        before=np.array(REFERENCE["sampling_ablation"]["clip6"][key])
        after=np.array(REFERENCE["sampling_ablation"]["volume3_return6"][key])
        for i,(v,color,label) in enumerate([(before,"#94a3b8","Before\nr6 / v6"),(after,"#2563eb","Selected\nr6 / v3")]):
            ax.bar(i,v.mean(),color=color,width=.58,alpha=.85)
            ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),color="#334155",capsize=4,linewidth=1)
            ax.scatter(i+np.linspace(-.1,.1,3),v,color="#0f172a",s=16,zorder=5)
            ax.text(i,max(v.max(),v.mean()+v.std(ddof=1))+.035*max(before.max(),after.max()),f"{v.mean():.3f}",ha="center",weight="bold")
        ax.set(title=titles[key],xticks=[0,1],xticklabels=["Before\nr6 / v6","Selected\nr6 / v3"],ylim=(0,max(before.max(),before.mean()+before.std(ddof=1))*1.27))
        ax.grid(axis="y",alpha=.12);ax.set_axisbelow(True)
        if key=="P99_ratio":ax.axhline(1,color="#059669",ls="--",lw=1,label="Real-data ratio = 1");ax.legend(fontsize=8)
    fig.suptitle("Channel-wise denoising bounds: tighter volume, unchanged return support",fontsize=14,weight="bold",y=1.01)
    fig.text(.5,-.005,"Same weights and paired sampling noise | 3 seeds x 5,000 paths | dots: seeds; bars: mean +/- sample SD",ha="center",fontsize=9,color="#64748b")
    fig.tight_layout()
    fig.savefig(dest/"clipping_improvement.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.2,5.2))
    wanted=np.array(REFERENCE["volatility_control"]["0"]["target"])*100
    ax.plot(range(5),wanted,"k--D",lw=2,markersize=6,label="Requested volatility")
    for seed,color in zip(["0","1","2"],["#2563eb","#f59e0b","#8b5cf6"]):
        actual=np.array(REFERENCE["volatility_control"][seed]["generated_mean"])*100
        ax.plot(range(5),actual,"o-",lw=1.7,markersize=5,color=color,label=f"Generated mean / seed {seed}")
    ax.set(xticks=range(5),xticklabels=["Lower","Low","Medium","High","Higher"],ylabel="Daily log-return volatility (%)",xlabel="Training-quantile control level")
    ax.set_title("Five-level continuous volatility control",pad=14,fontsize=15)
    ax.grid(alpha=.13);ax.legend(frameon=False,fontsize=9)
    fig.text(.5,.015,"Selected target example | r6 / v3 | 3 generator seeds | 16,196 paths per level and seed",ha="center",fontsize=9,color="#64748b")
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(dest/"volatility_control.png",dpi=180,bbox_inches="tight");plt.close(fig)
    print("REPORT_BACKED_FIGURES_RENDERED")

if __name__=="__main__":main()
