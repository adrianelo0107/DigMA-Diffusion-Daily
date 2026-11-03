"""从本地检查点生成任意数量的60日量价路径；默认收益6/对数成交量3裁剪。"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from common import read_config,resolve,atomic_json,sha256,TARGETS,BINS,metrics
from model import build_model
from evaluate import sample

def generate(checkpoint,metadata,config,output,n=1000,bin_name="Medium",target_value=None,
             guidance=4.,seed=0,device="cpu",return_clip=None,volume_clip=None):
    if n<=0:raise ValueError("n must be positive")
    checkpoint=resolve(checkpoint);metadata=resolve(metadata);output=resolve(output)
    if output.suffix!=".npz":raise ValueError("output must end with .npz")
    if output.exists() or output.with_suffix(".json").exists():raise FileExistsError(output)
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    contract=ck["contract"];method=contract["method"];target=contract["target"]
    if contract.get("data_sha256") and contract["data_sha256"]!=sha256(metadata):raise ValueError("metadata does not match checkpoint training data")
    c=copy.deepcopy(contract["config"])
    c["evaluation"]=copy.deepcopy(config["evaluation"])
    c["model"]["sampling_steps"]=config["model"]["sampling_steps"]
    bounds=config["model"]["x0_clip"]
    limits=[float(bounds),float(bounds)] if isinstance(bounds,(int,float)) else list(bounds)
    if return_clip is not None:limits[0]=return_clip
    if volume_clip is not None:limits[1]=volume_clip
    c["model"]["x0_clip"]=limits
    dev=torch.device(device);torch.set_num_threads(config["training"]["cpu_threads"])
    meta=json.loads(metadata.read_text(encoding="utf-8"))
    k=BINS.index(bin_name);desired=float(meta["targets"][target]["medians"][k]) if target_value is None else float(target_value)
    if target_value is not None and method!="continuous":raise ValueError("a numeric target_value requires a continuous checkpoint")
    model=build_model(c,method).to(dev);model.load_state_dict(ck["model"]);model.eval();del ck
    states=sample(model,c,meta,method,target,desired,k,n,0. if method=="no_control" else guidance,seed,dev)
    normalized=(states-np.asarray(meta["channel_mean"]).reshape(1,2,1))/(np.asarray(meta["channel_std"]).reshape(1,2,1)+1e-6)
    assert states.shape==(n,2,c["data"]["sequence_length"]) and np.isfinite(states).all()
    assert (np.abs(normalized).max(axis=(0,2))<=np.asarray(limits)+1e-5).all()
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,states=states,metrics=metrics(states[:,0,:]))
    record={"method":method,"target":target,"bin":bin_name,"target_value":desired,"paths":n,"sample_seed":seed,
            "sampler":"DDIM" if model.is_ddim_sampling else "DDPM","sampling_steps":model.sampling_timesteps,
            "x0_clip":limits,"guidance_scale":0. if method=="no_control" else guidance,
            "checkpoint_sha256":sha256(checkpoint),"metadata_sha256":sha256(metadata),"output_sha256":sha256(output),
            "channels":["daily_log_return","log1p_daily_volume"]}
    atomic_json(output.with_suffix(".json"),record)
    print(json.dumps(record,indent=2),flush=True)
    return states

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config");p.add_argument("--checkpoint",required=True);p.add_argument("--metadata")
    p.add_argument("--output",default="outputs/generated.npz");p.add_argument("--n",type=int,default=1000)
    p.add_argument("--bin",choices=BINS,default="Medium");p.add_argument("--target-value",type=float)
    p.add_argument("--guidance",type=float,default=4.);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cpu");p.add_argument("--return-clip",type=float);p.add_argument("--volume-clip",type=float)
    a=p.parse_args();c=read_config(a.config)
    metadata=a.metadata or str(Path(c["data"]["processed"])/"preprocessing.json")
    generate(a.checkpoint,metadata,c,a.output,a.n,a.bin,a.target_value,a.guidance,a.seed,a.device,a.return_clip,a.volume_clip)
