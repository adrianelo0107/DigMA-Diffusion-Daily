"""官方噪声预测损失的纯PyTorch训练器；断点含优化器和随机状态。"""
from __future__ import annotations
import argparse,contextlib,json,os,random,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader,Subset
from common import read_config,resolve,atomic_json,seed_all,sha256,TARGETS,bin_labels
from model import build_model

class DailyDataset(Dataset):
    def __init__(self,path,meta,method,target):
        self.x=np.load(path,mmap_mode="r")
        self.mean=np.asarray(meta["channel_mean"],np.float32).reshape(2,1)
        self.std=np.asarray(meta["channel_std"],np.float32).reshape(2,1)
        self.method=method
        values=np.load(path.with_name(path.stem+"_metrics.npy"),mmap_mode="r")
        if method=="no_control":self.cond=np.zeros((len(self.x),1),np.float32)
        elif method=="discrete":self.cond=bin_labels(values[:,TARGETS.index(target)],meta["targets"][target]).astype(np.int64).reshape(-1,1)
        else:
            info=meta["targets"][target]
            self.cond=((values[:,TARGETS.index(target)]-info["mean"])/(info["std"]+1e-6)).astype(np.float32).reshape(-1,1)
    def __len__(self):return len(self.x)
    def __getitem__(self,i):
        x=(np.array(self.x[i],copy=True)-self.mean)/(self.std+1e-6)
        return torch.from_numpy(x),torch.from_numpy(self.cond[i].copy())

def now():return datetime.now(timezone.utc).isoformat()
def run_dir(config,seed,method,target):
    return resolve(config["output"])/f"seed_{seed}"/("no_control" if method=="no_control" else f"{target}_{method}")
def save_torch(path,obj):
    tmp=path.with_suffix(".tmp");torch.save(obj,tmp);os.replace(tmp,path)
def amp_context(device,config):
    if device.type!="cuda":return contextlib.nullcontext()
    dtype=torch.bfloat16 if config["training"]["precision"]=="bf16" else torch.float16
    return torch.autocast("cuda",dtype=dtype)
def train_one(config,method,target,seed,device="cuda",resume=False):
    cfg=config["training"];torch.set_num_threads(cfg["cpu_threads"]);seed_all(seed)
    device=torch.device(device);folder=run_dir(config,seed,method,target);folder.mkdir(parents=True,exist_ok=True)
    statefile=folder/"status.json"
    if statefile.exists():
        old=json.loads(statefile.read_text())
        if old["state"]=="complete":return old
        if not resume:raise RuntimeError(f"已有未完成运行 {folder}，请核查后显式 --resume")
    data=resolve(config["data"]["processed"]);meta=json.loads((data/"preprocessing.json").read_text())
    trainset=DailyDataset(data/"train.npy",meta,method,target)
    valset=DailyDataset(data/"validation.npy",meta,method,target)
    if cfg.get("smoke_train_limit"):
        if "smoke" not in config["experiment"]:raise ValueError("样本截断仅允许用于明确命名的smoke实验")
        trainset=Subset(trainset,range(min(cfg["smoke_train_limit"],len(trainset))))
    ids=np.random.default_rng(1701).choice(len(valset),min(cfg["validation_samples"],len(valset)),replace=False)
    trainloader=DataLoader(trainset,batch_size=cfg["batch_size"],shuffle=True,num_workers=cfg["num_workers"],pin_memory=device.type=="cuda",drop_last=True)
    valloader=DataLoader(Subset(valset,ids),batch_size=cfg["batch_size"],shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
    model=build_model(config,method).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda" and cfg["precision"]=="fp16")
    start_epoch=0;best=float("inf");global_step=0
    contract={"method":method,"target":target,"seed":seed,"config":config,"data_sha256":sha256(data/"preprocessing.json")}
    if resume and (folder/"last.pt").exists():
        ck=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
        if ck["contract"]!=contract:raise ValueError("恢复配置/数据与断点不一致")
        model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);scaler.load_state_dict(ck["scaler"])
        start_epoch=ck["epoch"]+1;best=ck["best_val"];global_step=ck["global_step"]
        torch.set_rng_state(ck["torch_rng"]);np.random.set_state(ck["numpy_rng"]);random.setstate(ck["python_rng"])
        if device.type=="cuda":torch.cuda.set_rng_state_all(ck["cuda_rng"])
    state={**{k:contract[k] for k in ["method","target","seed"]},"state":"running","started":now(),
           "train_samples":len(trainset),"validation_samples":len(ids),"parameters":sum(p.numel() for p in model.parameters()),"device":str(device)}
    t0=time.time()
    try:
        for epoch in range(start_epoch,cfg["epochs"]):
            model.train();total=0.;seen=0
            for step,(x,c) in enumerate(trainloader):
                x=x.to(device,non_blocking=True);c=c.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
                with amp_context(device,config):loss=model(x,classes=c)
                if not torch.isfinite(loss):raise FloatingPointError("训练损失非有限")
                scaler.scale(loss).backward();scaler.step(opt);scaler.update()
                total+=float(loss.detach())*len(x);seen+=len(x);global_step+=1
                if (step+1)%cfg["log_every"]==0 or step==0:
                    state.update(epoch=epoch+1,step=step+1,steps_per_epoch=len(trainloader),global_step=global_step,
                                 train_loss_so_far=total/seen,updated=now(),elapsed_seconds=time.time()-t0)
                    atomic_json(statefile,state);print(json.dumps(state),flush=True)
            model.eval();vtotal=0.;vseen=0
            # 固定验证噪声与条件丢弃随机数，减少逐轮验证的Monte Carlo干扰。
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []):
                torch.manual_seed(seed+12345)
                with torch.no_grad():
                    for x,c in valloader:
                        x=x.to(device);c=c.to(device)
                        with amp_context(device,config):vl=model(x,classes=c)
                        vtotal+=float(vl)*len(x);vseen+=len(x)
            val=vtotal/vseen
            if not np.isfinite(val):raise FloatingPointError("验证损失非有限")
            if val<best:
                best=val;save_torch(folder/"best.pt",{"contract":contract,"model":model.state_dict(),"epoch":epoch,"val_loss":val})
            ck={"contract":contract,"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),
                "epoch":epoch,"global_step":global_step,"best_val":best,"torch_rng":torch.get_rng_state(),
                "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
                "cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else []}
            save_torch(folder/"last.pt",ck)
            record={"epoch":epoch+1,"train_loss":total/seen,"val_loss":val,"best_val":best,"global_step":global_step,"updated":now()}
            with (folder/"history.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(record)+"\n")
            state.update(record);atomic_json(statefile,state);print(json.dumps(record),flush=True)
        state.update(state="complete",finished=now(),elapsed_seconds=time.time()-t0)
        atomic_json(statefile,state)
        return state
    except BaseException as exc:
        state.update(state="failed",error=repr(exc),updated=now());atomic_json(statefile,state);raise
    finally:
        del model,opt
        if device.type=="cuda":torch.cuda.empty_cache()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--method",choices=["no_control","discrete","continuous"],required=True)
    p.add_argument("--target",choices=TARGETS,default="return");p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cuda");p.add_argument("--resume",action="store_true");a=p.parse_args()
    train_one(read_config(a.config),a.method,a.target,a.seed,a.device,a.resume)
