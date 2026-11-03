"""顺序执行3随机种子×7模型，随后生成控制评估；可直接本地或服务器运行。"""
from __future__ import annotations
import argparse,json,os,platform,time,traceback
from pathlib import Path
from common import read_config,resolve,atomic_json,sha256,TARGETS
from train import train_one,run_dir,now

def verify(config):
    out=resolve(config["output"]);checks=[]
    for seed in config["training"]["seeds"]:
        jobs=[("no_control","return")]+[(m,t) for t in TARGETS for m in ["continuous","discrete"]]
        for method,target in jobs:
            p=run_dir(config,seed,method,target);s=json.loads((p/"status.json").read_text())
            assert s["state"]=="complete" and s["epoch"]==config["training"]["epochs"],str(p)
            rows=[json.loads(x) for x in (p/"history.jsonl").read_text().splitlines()]
            assert len({r["epoch"] for r in rows})==config["training"]["epochs"],str(p)
            checks.append({"run":str(p.relative_to(out)),"last_sha256":sha256(p/"last.pt"),"best_sha256":sha256(p/"best.pt")})
        e=json.loads((out/f"seed_{seed}"/"evaluation/complete.json").read_text())
        assert e["state"]=="complete" and e["rows"]==45
    atomic_json(out/"verification.json",{"verified":True,"models":checks,"seeds":config["training"]["seeds"],"verified_at":now()})
    return checks
def run(config,device,resume=False):
    out=resolve(config["output"]);out.mkdir(parents=True,exist_ok=True)
    statefile=out/"suite_status.json"
    if statefile.exists() and not resume:raise RuntimeError("套件已存在；先check，确认后显式--resume")
    atomic_json(out/"config_frozen.json",config)
    state={"state":"running","pid":os.getpid(),"started":now(),"completed_models":0,
           "total_models":7*len(config["training"]["seeds"]),"device":device,"python":platform.python_version()}
    atomic_json(statefile,state)
    from evaluate import evaluate_seed,aggregate
    try:
        for seed in config["training"]["seeds"]:
            jobs=[("no_control","return")]+[(m,t) for t in TARGETS for m in ["continuous","discrete"]]
            for method,target in jobs:
                state.update(stage="training",seed=seed,method=method,target=target,updated=now());atomic_json(statefile,state)
                train_one(config,method,target,seed,device,resume)
                state["completed_models"]+=1;atomic_json(statefile,state)
            state.update(stage="evaluation",seed=seed,updated=now());atomic_json(statefile,state)
            evaluate_seed(config,seed,device)
        aggregate(config);state.update(stage="verification",updated=now());atomic_json(statefile,state)
        verify(config);state.update(state="complete",stage="complete",finished=now());atomic_json(statefile,state)
    except BaseException as exc:
        state.update(state="failed",error=repr(exc),updated=now());atomic_json(statefile,state);raise
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--device",default="cuda")
    p.add_argument("--resume",action="store_true");p.add_argument("--verify-only",action="store_true");a=p.parse_args()
    c=read_config(a.config)
    verify(c) if a.verify_only else run(c,a.device,a.resume)
