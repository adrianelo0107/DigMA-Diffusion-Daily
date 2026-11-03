"""日频实验的公共路径、原子落盘和随机种子。"""
from __future__ import annotations
import hashlib,json,os,random
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TARGETS=("return","amplitude","volatility")
METHODS=("no_control","discrete","continuous")
BINS=("Lower","Low","Medium","High","Higher")

def read_config(path=None):
    p=Path(path) if path else ROOT/"configs/default.json"
    return json.loads(p.read_text(encoding="utf-8"))
def resolve(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p
def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(q,p)
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def metrics(returns):
    """完全沿用官方三个目标：对数收益和、累计对数路径极差、日对数收益总体标准差。"""
    r=np.asarray(returns,dtype=np.float64)
    c=r.cumsum(axis=-1)
    return np.stack([c[...,-1],c.max(axis=-1)-c.min(axis=-1),r.std(axis=-1,ddof=0)],axis=-1)
def condition_info(values,n_bins=5):
    edges=np.quantile(values,np.linspace(0,1,n_bins+1))
    if np.any(np.diff(edges)<=0):
        raise ValueError("目标分位点不严格递增，无法构造五档；请检查数据")
    labels=np.searchsorted(edges[1:-1],values,side="right")
    return {"edges":edges.tolist(),"medians":[float(np.median(values[labels==i])) for i in range(n_bins)],
            "mean":float(values.mean()),"std":float(values.std()),"counts":[int((labels==i).sum()) for i in range(n_bins)]}
def bin_labels(values,info):
    return np.searchsorted(np.asarray(info["edges"])[1:-1],values,side="right")
