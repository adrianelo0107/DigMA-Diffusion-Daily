"""从股票日频面板构造无跨股票、无跨划分、连续60交易日的两通道样本。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view
from common import ROOT,read_config,resolve,atomic_json,sha256,metrics,condition_info,TARGETS

def prepare(config):
    c=config["data"];src=resolve(c["source"]);out=resolve(c["processed"])
    if (out/"preprocessing.json").exists():
        meta=json.loads((out/"preprocessing.json").read_text())
        if meta["data_config"]!=c or meta["source_sha256"]!=sha256(src):
            raise ValueError("已有缓存与当前配置/原始数据不一致，请改用新的processed目录")
        print("Reuse verified prepared data",flush=True);return meta
    if out.exists() and any(out.iterdir()):raise ValueError("缓存目录有未完成文件，请检查后换一个新目录")
    out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    # 先限制原始记录，再构造交易日历/收益/窗口/训练统计，禁止复用全期参数。
    filters=[]
    if c.get("start_date"):filters.append(("date",">=",pd.Timestamp(c["start_date"])))
    if c.get("end_date"):filters.append(("date","<=",pd.Timestamp(c["end_date"])))
    raw_rows=pq.ParquetFile(src).metadata.num_rows
    d=pq.read_table(src,columns=["order_book_id","date","close","volume"],filters=filters or None).to_pandas()
    if d.empty:raise ValueError("指定日期范围内没有记录")
    d["date"]=pd.to_datetime(d["date"],errors="raise")
    if d["date"].isna().any():raise ValueError("日期不能为空")
    audit={"rows":len(d),"stocks":int(d.order_book_id.nunique()),"nulls":{k:int(v) for k,v in d.isna().sum().items()},
           "zero_or_negative_volume":int((d.volume<=0).sum()),"nonpositive_close":int((d.close<=0).sum())}
    audit.update(raw_source_rows=raw_rows,rows_excluded_by_date=raw_rows-len(d))
    if d.duplicated(["order_book_id","date"]).any():raise ValueError("股票日期主键重复，禁止静默去重")
    d=d.sort_values(["order_book_id","date"]).reset_index(drop=True)
    calendar=np.sort(d.date.unique());T=c["sequence_length"]
    ntrain=int(len(calendar)*c["train_fraction"]);nval=int(len(calendar)*c["validation_fraction"])
    cut1=calendar[ntrain-1];cut2=calendar[ntrain+nval-1]
    states={k:[] for k in ["train","validation","test"]}
    metarows={k:[] for k in states};rejected=0;candidate=0;extreme=0
    for number,(stock,g) in enumerate(d.groupby("order_book_id",sort=False)):
        dates=g.date.to_numpy();close=g.close.to_numpy(dtype=np.float64);vol=g.volume.to_numpy(dtype=np.float64)
        sessions=np.searchsorted(calendar,dates)
        r=np.full(len(g),np.nan,dtype=np.float64)
        with np.errstate(divide="ignore",invalid="ignore"):
            r[1:]=np.log(close[1:]/close[:-1])
        continuous=np.r_[False,np.diff(sessions)==1]
        extreme+=int(np.sum(np.isfinite(r)&(np.abs(r)>c["max_abs_log_return"])))
        valid=continuous&np.isfinite(r)&(close>0)&(vol>0)&np.isfinite(vol)&(np.abs(r)<=c["max_abs_log_return"])
        if len(g)<T:continue
        x=np.stack([r,np.log1p(np.maximum(vol,0))],axis=1).astype(np.float32)
        starts=np.arange(0,len(g)-T+1,c["stride"]);ends=starts+T-1
        candidate+=len(starts)
        bad=np.r_[0,np.cumsum(~valid)]
        ok=(bad[ends+1]-bad[starts]==0)&(sessions[ends]-sessions[starts]==T-1)
        rejected+=int((~ok).sum())
        windows=sliding_window_view(x,T,axis=0)
        for split,mask in [("train",dates[ends]<=cut1),
                           ("validation",(dates[starts]>cut1)&(dates[ends]<=cut2)),
                           ("test",dates[starts]>cut2)]:
            idx=starts[ok&mask]
            if len(idx):
                states[split].append(windows[idx].copy())
                metarows[split].append(pd.DataFrame({"stock":stock,"start":dates[idx],"end":dates[idx+T-1]}))
        if number%500==0:print(f"Prepared {number+1} stocks",flush=True)
    del d
    arrays={};sample_counts={}
    for split,chunks in states.items():
        if not chunks:raise ValueError(f"{split} 无有效样本")
        a=np.concatenate(chunks,axis=0);states[split]=[]
        if not np.isfinite(a).all():raise ValueError("样本存在非有限值")
        np.save(out/f"{split}.npy",a);sample_counts[split]=len(a)
        pd.concat(metarows[split],ignore_index=True).to_parquet(out/f"{split}_windows.parquet",index=False)
        arrays[split]=a
    train=arrays["train"]
    mean=train.mean(axis=(0,2),dtype=np.float64);std=train.std(axis=(0,2),dtype=np.float64)
    if (std<=0).any():raise ValueError("通道标准差必须为正")
    train_metrics=metrics(train[:,0,:])
    infos={target:condition_info(train_metrics[:,j],config["evaluation"]["n_bins"]) for j,target in enumerate(TARGETS)}
    for split,a in arrays.items():np.save(out/f"{split}_metrics.npy",metrics(a[:,0,:]))
    audit.update({"extreme_return_rows_excluded":extreme,"candidate_windows":candidate,"invalid_windows":rejected,
                  "cross_split_windows_excluded":candidate-rejected-sum(sample_counts.values())})
    meta={"data_config":c,"source":str(src.name),"source_sha256":sha256(src),"source_audit":audit,
          "calendar_start":str(calendar[0])[:10],"calendar_end":str(calendar[-1])[:10],
          "train_end":str(cut1)[:10],"validation_end":str(cut2)[:10],"samples":sample_counts,
          "channels":["daily_log_return","log1p_daily_volume"],
          "channel_mean":mean.tolist(),"channel_std":std.tolist(),"targets":infos,
          "volume_unit":"source volume units; no unit conversion",
          "return_basis":"log(close_t / close_previous_stock_session); prev_close deliberately not used",
          "elapsed_seconds":time.time()-t0}
    meta["prepared_sha256"]={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    atomic_json(out/"preprocessing.json",meta)
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    return meta
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config");args=p.parse_args()
    prepare(read_config(args.config))
