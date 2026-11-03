"""复用官方网络和扩散核心，仅移除原Lightning/分钟数据包装。"""
import torch
from diffusion.cont_ctrl_net import UnetCC
from diffusion.disc_ctrl_net import UnetDC
from diffusion.ddpm import GaussianDiffusion

def build_model(config,method,small=False):
    c=config["model"];dim=8 if small else c["dim"]
    mults=(1,2,4) if small else tuple(c["dim_mults"])
    kw=dict(dim=dim,dim_mults=mults,channels=2)
    if method=="discrete":
        net=UnetDC(num_classes=config["evaluation"]["n_bins"],cond_drop_prob=c["cond_drop_prob"],**kw)
    else:
        net=UnetCC(n_ctrls=1,cond_drop_prob=1.0 if method=="no_control" else c["cond_drop_prob"],**kw)
    return GaussianDiffusion(net,seq_length=config["data"]["sequence_length"],
        timesteps=c["diffusion_steps"],sampling_timesteps=c["sampling_steps"],
        objective="pred_noise",auto_normalize=False,x0_clip=c["x0_clip"])
def condition_tensor(values,method,target_info,device):
    if method=="discrete":return torch.as_tensor(values,dtype=torch.long,device=device).reshape(-1,1)
    if method=="no_control":return torch.zeros((len(values),1),dtype=torch.float32,device=device)
    z=(values-target_info["mean"])/(target_info["std"]+1e-6)
    return torch.as_tensor(z,dtype=torch.float32,device=device).reshape(-1,1)
