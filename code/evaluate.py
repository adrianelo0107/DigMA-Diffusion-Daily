"""五档控制MSE、验证集CFG选择、Figure3风格价格/分布图及成交量诊断。"""
from __future__ import annotations
import argparse,json,os,time,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from common import read_config,resolve,atomic_json,metrics,bin_labels,TARGETS,METHODS,BINS,seed_all
from model import build_model,condition_tensor
from train import run_dir,amp_context

def sample(model,config,meta,method,target,target_value,bin_index,n,scale,seed,device):
    seed_all(seed);info=meta["targets"][target];out=[];batch=config["evaluation"]["sample_batch_size"]
    with torch.inference_mode():
        for start in range(0,n,batch):
            size=min(batch,n-start)
            values=np.full(size,bin_index if method=="discrete" else target_value)
            cond=condition_tensor(values,method,info,device)
            with amp_context(device,config):
                x=model.sample(cond,cond_scale=0. if method=="no_control" else scale,
                               rescaled_phi=0. if method=="no_control" else config["evaluation"]["rescaled_phi"])
            out.append(x.float().cpu().numpy())
    normalized=np.concatenate(out)
    mean=np.asarray(meta["channel_mean"]).reshape(1,2,1);std=np.asarray(meta["channel_std"]).reshape(1,2,1)
    # 与训练使用同一带epsilon的尺度进行严格逆变换。
    states=normalized*(std+1e-6)+mean
    if not np.isfinite(states).all():raise FloatingPointError("生成结果存在非有限值")
    return states.astype(np.float32)

def evaluate_seed(config,seed,device="cuda"):
    device=torch.device(device);torch.set_num_threads(config["training"]["cpu_threads"])
    output=resolve(config["output"]);folder=output/f"seed_{seed}"/"evaluation";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():return
    data=resolve(config["data"]["processed"]);meta=json.loads((data/"preprocessing.json").read_text())
    valmetrics=np.load(data/"validation_metrics.npy")
    e=config["evaluation"];rows=[];calibration=[];volrows=[]
    def load(method,target):
        checkpoint=run_dir(config,seed,method,target)/"last.pt"
        ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if ck["epoch"]+1!=config["training"]["epochs"]:raise ValueError("正式评估必须使用训练完成的断点")
        model=build_model(config,method).to(device);model.load_state_dict(ck["model"]);model.eval()
        return model
    baseline_model=load("no_control","return")
    baseline=sample(baseline_model,config,meta,"no_control","return",0,0,e["test_samples"],0,seed+70000,device)
    del baseline_model
    if device.type=="cuda":torch.cuda.empty_cache()
    np.savez_compressed(folder/"samples_no_control.npz",states=baseline)
    for j,target in enumerate(TARGETS):
        info=meta["targets"][target];vlabels=bin_labels(valmetrics[:,j],info)
        vmedians=[float(np.median(valmetrics[vlabels==k,j])) for k in range(5)]
        for method in METHODS:
            model=None if method=="no_control" else load(method,target)
            for k,name in enumerate(BINS):
                desired=info["medians"][k];best_scale=0.
                if method=="no_control":s=baseline
                else:
                    candidates=[]
                    for scale in e["guidance_scales"]:
                        # 仅用验证期各档中位数选CFG，同尺度共用随机噪声便于比较。
                        v=sample(model,config,meta,method,target,vmedians[k],k,e["calibration_samples"],scale,
                                 seed+10000+j*100+k,device)
                        loss=float(np.mean((metrics(v[:,0,:])[:,j]-vmedians[k])**2))
                        candidates.append((loss,scale))
                        calibration.append({"seed":seed,"target":target,"method":method,"bin":name,"scale":scale,
                                            "validation_target":vmedians[k],"validation_mse":loss})
                    best_scale=min(candidates)[1]
                    s=sample(model,config,meta,method,target,desired,k,e["test_samples"],best_scale,
                             seed+30000+j*100+k,device)
                    np.savez_compressed(folder/f"samples_{target}_{method}_{name}.npz",states=s,
                                        metrics=metrics(s[:,0,:]),target=desired,guidance_scale=best_scale)
                values=metrics(s[:,0,:])[:,j];mse=float(np.mean((values-desired)**2))
                rows.append({"seed":seed,"target":target,"method":method,"bin":name,"target_value":desired,
                             "guidance_scale":best_scale,"n_samples":len(s),"mse":mse,"normalized_mse":mse/info["std"]**2,
                             "generated_mean":float(values.mean()),"generated_std":float(values.std()),
                             "mean_bias":float(values.mean()-desired)})
                volumes=np.expm1(s[:,1,:].astype(np.float64))
                volrows.append({"seed":seed,"target":target,"method":method,"bin":name,
                    "volume_median":float(np.median(volumes)),"volume_p01":float(np.quantile(volumes,.01)),
                    "volume_p99":float(np.quantile(volumes,.99)),
                    "return_abs_p99":float(np.quantile(np.abs(s[:,0,:]),.99)),
                    "returns_above_training_filter_fraction":float(np.mean(np.abs(s[:,0,:])>config["data"]["max_abs_log_return"]))})
                print(json.dumps(rows[-1]),flush=True)
                pd.DataFrame(rows).to_csv(folder/"table1_long.csv",index=False)
                pd.DataFrame(calibration).to_csv(folder/"guidance_validation.csv",index=False)
            del model
            if device.type=="cuda":torch.cuda.empty_cache()
    pd.DataFrame(volrows).to_csv(folder/"volume_diagnostics.csv",index=False)
    pd.DataFrame(rows).pivot(index=["target","method"],columns="bin",values="mse").reindex(columns=BINS).to_csv(folder/"table1_mse.csv")
    if seed==e["figure_seed"]:
        atomic_json(folder/"evaluation_config.json",config)
        subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("plot.py")),"--config",str(folder/"evaluation_config.json"),"--seed",str(seed)],check=True)
    atomic_json(folder/"complete.json",{"state":"complete","seed":seed,"rows":len(rows),"samples_per_bin":e["test_samples"],
                                     "checkpoint":"last","sampler":"DDIM","sampling_steps":config["model"]["sampling_steps"]})

def aggregate(config):
    out=resolve(config["output"]);e=config["evaluation"]
    dfs=[pd.read_csv(out/f"seed_{s}"/"evaluation/table1_long.csv") for s in config["training"]["seeds"]]
    d=pd.concat(dfs,ignore_index=True);d.to_csv(out/"table1_all_seeds.csv",index=False)
    a=d.groupby(["target","method","bin"],sort=False).agg(mse_mean=("mse","mean"),mse_std=("mse","std"),
                  normalized_mse_mean=("normalized_mse","mean"),normalized_mse_std=("normalized_mse","std")).reset_index()
    a.to_csv(out/"table1_summary.csv",index=False)
    a["cell"]=[f"{m:.6g} ± {s:.3g}" for m,s in zip(a.mse_mean,a.mse_std.fillna(0))]
    table=a.pivot(index=["target","method"],columns="bin",values="cell").reindex(columns=BINS)
    table.to_csv(out/"table1_mean_std.csv")
    lines=["# 60日行情生成评估","","结果对应用户日频数据上的适配实验，指标单位与原论文表格不同。","","| Target | Method | "+" | ".join(BINS)+" |",
           "|---|---|"+"---|"*5]
    for (target,method),row in table.iterrows():lines.append("| "+target+" | "+method+" | "+" | ".join(row)+" |")
    lines+=["","MSE按三个目标各自的原始单位计算；normalized_mse为除以训练集目标方差的误差，详见CSV。",
            "Figure3与成交量图位于seed_0/evaluation。目标均为训练集五分位箱中位数；CFG仅按验证期目标选取。"]
    (out/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--seed",type=int,default=0);p.add_argument("--device",default="cuda");p.add_argument("--aggregate",action="store_true");a=p.parse_args()
    c=read_config(a.config)
    aggregate(c) if a.aggregate else evaluate_seed(c,a.seed,a.device)
