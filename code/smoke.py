"""无需私有行情的CPU端到端烟测：合成面板→预处理→七模型→五档评估→生成/KL。"""
import argparse,copy,json
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch
from common import ROOT,read_config,atomic_json,seed_all
from prepare import prepare
from model import build_model
from train import DailyDataset
from run_suite import run
from generate import generate
from fidelity import evaluate as fidelity_evaluate,state_features

def smoke(device="cpu"):
    torch.set_num_threads(2)
    folder=ROOT/"outputs"/("smoke_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"));folder.mkdir(parents=True)
    c=copy.deepcopy(read_config());c["experiment"]="synthetic_smoke";c["output"]=str(folder/"models")
    c["data"].update(source=str(folder/"synthetic_market.parquet"),processed=str(folder/"processed"),stride=20)
    c["model"].update(dim=8,dim_mults=[1,2,4],diffusion_steps=20,sampling_steps=4,x0_clip=[6.,3.])
    c["training"].update(seeds=[0],epochs=1,batch_size=8,smoke_train_limit=32,validation_samples=16,cpu_threads=2,log_every=4)
    c["evaluation"].update(calibration_samples=8,test_samples=8,guidance_scales=[1,2],sample_batch_size=8)
    rng=np.random.default_rng(4491);dates=pd.bdate_range("2015-01-05",periods=1800);rows=[]
    for stock in range(32):
        sigma=(.008+.002*(stock%8))*(1+.2*np.sin(np.arange(len(dates))/75))
        r=rng.normal((stock%5-2)*.00015,sigma);price=100*np.exp(r.cumsum())
        volume=np.exp(14.7+.03*stock+rng.normal(0,.4,len(dates)))
        rows.append(pd.DataFrame({"order_book_id":f"SYNTHETIC_{stock:03d}","date":dates,"close":price,"volume":volume}))
    pd.concat(rows,ignore_index=True).to_parquet(c["data"]["source"],index=False)
    atomic_json(folder/"config.json",c);meta=prepare(c)
    before_models=[]
    for method in ["no_control","discrete","continuous"]:
        seed_all(71);model=build_model(c,method).to(device)
        assert model.x0_clip.shape==(1,2,1) and model.x0_clip.device==model.betas.device
        assert not any(k.endswith("x0_clip") for k in model.state_dict())
        legacy=copy.deepcopy(c);legacy["model"]["x0_clip"]=6.
        other=build_model(legacy,method).to(device);other.load_state_dict(model.state_dict(),strict=True)
        ds=DailyDataset(Path(c["data"]["processed"])/"train.npy",meta,method,"return")
        x=torch.stack([ds[i][0] for i in range(8)]).to(device);cond=torch.stack([ds[i][1] for i in range(8)]).to(device)
        old={n:p.detach().clone() for n,p in model.named_parameters()}
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3);loss=model(x,classes=cond)
        assert torch.isfinite(loss);loss.backward();opt.step()
        assert any(not torch.equal(old[n],p.detach()) for n,p in model.named_parameters())
        model.eval();model.set_x0_clip([.5,.25])
        with torch.inference_mode():s=model.sample(cond[:2],cond_scale=0 if method=="no_control" else 2,rescaled_phi=.7)
        assert torch.isfinite(s).all() and s.shape==(2,2,60)
        assert bool((s.abs().amax(dim=(0,2))<=torch.tensor([.5,.25],device=device)+1e-5).all())
        before_models.append({"method":method,"loss":float(loss.detach()),"parameters_updated":True})
    run(c,device)
    source=Path(c["data"]["processed"]);checkpoint=Path(c["output"])/"seed_0/volatility_continuous/last.pt"
    generated=folder/"standalone_generated.npz"
    generate(checkpoint,source/"preprocessing.json",c,generated,n=64,bin_name="High",guidance=2,seed=1900,device=device)
    fidelity_evaluate(source/"test.npy",generated,folder/"standalone_fidelity.json")
    zeros=state_features(np.zeros((3,2,60)));assert all(np.isfinite(v).all() for v in zeros.values())
    proof={"verified":True,"synthetic_data_only":True,"real_data_distributed":False,"device":device,
           "seven_models_and_five_levels_evaluated":True,"all_method_parameter_updates":before_models,
           "legacy_state_dict_compatible":True,"per_channel_sampling_bounds_checked":True,
           "generation_and_four_metric_fidelity_checked":True}
    atomic_json(folder/"smoke_verification.json",proof);print("SYNTHETIC_END_TO_END_SMOKE_VERIFIED",folder,flush=True)
    return folder

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--device",default="cpu");a=p.parse_args();smoke(a.device)
