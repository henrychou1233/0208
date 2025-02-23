import torch
from noise import *
from utilities import *
from visualize import *
from tqdm import tqdm
from forward_process import *

import torch
from noise import *
from utilities import *
from visualize import *
from tqdm import tqdm
from forward_process import *

def compute_adaptive_noise_scale(xt, et, at):
    """
    更穩定的自適應噪聲尺度計算
    """
    # 使用更穩定的方法計算噪聲尺度
    noise_energy = torch.norm(et)
    signal_energy = torch.norm(xt)
    
    # 避免除零並限制縮放範圍
    signal_to_noise_ratio = signal_energy / (noise_energy + 1e-8)
    
    # 使用雙曲正切函數進行平滑縮放
    adaptive_scale = torch.tanh(signal_to_noise_ratio * 0.1)
    
    return max(0.5, min(1.5, adaptive_scale.item()))

def my_generalized_steps(y, x, seq, model, b, config, eta2, eta3, constants_dict, eraly_stop = True):
    with torch.no_grad():
        n = x.size(0)
        
        m = seq[0].size(0)  # Length of each sequence
        x0_preds = []
        xs = [x]
        
        for idx in reversed(range(m)):
            t = torch.tensor([s[idx] for s in seq]).to(config.model.device)
            if idx == 0:
                next_t = torch.full((n,), -1, device=x.device, dtype=torch.long)
            else:
                next_t = torch.tensor([s[idx-1] for s in seq]).to(x.device)
            
            at = compute_alpha2(b, t.long(), config)
            at_next = compute_alpha2(b, next_t.long(),config)

            xt = xs[-1].to(config.model.device)
            
            et = model(xt, t)
            
            # 自適應噪聲調整
            noise_scale = compute_adaptive_noise_scale(xt, et, at)
            adaptive_eta2 = eta2 * noise_scale

            yt = at.sqrt() * y + (1- at).sqrt() *  et

            #DDAD error correction
            et_hat = et - (1 - at).sqrt() * adaptive_eta2 * (yt-xt)

            x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()

            x0_preds.append(x0_t.to('cpu')) 

            c1 = (
                config.model.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat

            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def DA_generalized_steps(y, x, seq, model, b, config, eta2, eta3, constants_dict, eraly_stop = True):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        
        xs = [x]
        
        for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            t_one = (torch.ones(n)).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long(), config)
            at_half = compute_alpha(b,(t/2).long(), config)
            at_next = compute_alpha(b, next_t.long(),config)
            xt = xs[-1].to(config.model.device)
            
            et = model(xt, t)
            
            # 自適應噪聲調整
            noise_scale = compute_adaptive_noise_scale(xt, et, at)
            adaptive_eta2 = eta2 * noise_scale
            
            yt = at.sqrt() * y + (1- at).sqrt() *  et

            et_hat = et - (1 - at).sqrt() * adaptive_eta2 * (yt-xt)

            
            x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()

            x0_preds.append(x0_t.to('cpu')) 
            if index == 0:
                c1 = torch.zeros_like(x0_t)
                c2 = torch.zeros_like(x0_t)
            else:
                c1 = (
                    config.model.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat

            xs.append(xt_next.to('cpu'))

    return xs, x0_preds