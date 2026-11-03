"""DayR/RetAC/VolC/LogV的真实→生成KL；保留范围外比例，不将LogV称为OIR。"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import entropy
from common import resolve,atomic_json,sha256

def state_features(states):
    x=np.asarray(states)
    if x.ndim!=3 or x.shape[1]!=2 or x.shape[2]<3 or not np.isfinite(x).all():raise ValueError("expected finite (N,2,T), T>=3")
    r=x[:,0,:].astype(np.float64);centered=r-r.mean(axis=-1,keepdims=True)
    ac=(centered[:,1:]*centered[:,:-1]).sum(axis=-1)/(r.var(axis=-1)*r.shape[-1]+1e-8)
    sq=r*r;a=sq[:,1:];b=sq[:,:-1]
    za=(a-a.mean(axis=-1,keepdims=True))/(a.std(axis=-1,keepdims=True)+1e-8)
    zb=(b-b.mean(axis=-1,keepdims=True))/(b.std(axis=-1,keepdims=True)+1e-8)
    return {"DayR":r.ravel(),"RetAC":ac,"VolC":(za*zb).mean(axis=-1),"LogV":x[:,1,:].astype(np.float64).ravel()}

def kl_on_reference(real,generated,bins=50):
    p,edges=np.histogram(real,bins=bins,density=True)
    counts,_=np.histogram(generated,bins=edges);inside=int(counts.sum())
    if not inside:raise ValueError("all generated values outside real histogram range")
    q=counts/(inside*np.diff(edges))
    return {"kl_real_to_generated":float(entropy(p,q+1e-9)),"generated_outside_real_range_fraction":float(1-inside/len(generated))}

def read_states(path):
    p=resolve(path)
    if p.suffix==".npz":
        with np.load(p) as z:return z["states"]
    return np.load(p,mmap_mode="r")

def evaluate(real_path,generated_path,output):
    real=read_states(real_path);generated=read_states(generated_path)
    if real.shape[1:]!=generated.shape[1:]:raise ValueError("real and generated channels/length differ")
    rf=state_features(real);gf=state_features(generated)
    result={k:kl_on_reference(rf[k],gf[k]) for k in rf}
    raw_real=np.expm1(rf["LogV"]);raw_gen=np.expm1(gf["LogV"])
    result["volume"]={"median_ratio_to_real":float(np.median(raw_gen)/np.median(raw_real)),
                      "p99_ratio_to_real":float(np.quantile(raw_gen,.99)/np.quantile(raw_real,.99))}
    result["protocol"]={"bins":50,"edges":"real full-range equal-width","smoothing":"1e-9 on generated density",
                        "direction":"real || generated","log_base":"natural","LogV_is_OIR":False,
                        "real_paths":len(real),"generated_paths":len(generated),"real_sha256":sha256(resolve(real_path)),
                        "generated_sha256":sha256(resolve(generated_path))}
    output=resolve(output)
    if output.exists():raise FileExistsError(output)
    atomic_json(output,result);print(json.dumps(result,indent=2));return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--real",required=True);p.add_argument("--generated",required=True)
    p.add_argument("--output",default="outputs/fidelity.json");a=p.parse_args();evaluate(a.real,a.generated,a.output)
