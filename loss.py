import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from forward_process import *
from noise import *

def get_loss(model, constant_dict, x_0, t, config):
    """
    自適應損失函數計算，融合多重策略優化模型性能
    
    參數:
    - model: 擴散模型
    - constant_dict: 常數字典
    - x_0: 原始輸入數據
    - t: 時間步驟
    - config: 模型配置
    
    返回:
    - 自適應計算的損失值
    """
    # 將張量移動到指定設備
    x_0 = x_0.to(config.model.device)
    b = constant_dict['betas'].to(config.model.device)
    
    # 生成隨機噪聲
    e = torch.randn_like(x_0, device=x_0.device)
    
    # 計算alpha_t並添加噪聲到輸入
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = at.sqrt() * x_0 + (1 - at).sqrt() * e
    
    # 模型前向傳播
    output = model(x, t.float())
    
    # 自適應損失計算策略
    # 1. 基礎均方誤差損失
    base_loss = F.mse_loss(e, output)
    
    # 2. 自適應權重策略
    # 根據時間步驟動態調整損失權重
    time_weight = 1.0 / (t.float() + 1)  # 較早期時間步驟獲得更高權重
    adaptive_loss = F.mse_loss(e, output, reduction='none')
    weighted_adaptive_loss = (adaptive_loss * time_weight.view(-1, 1, 1, 1)).mean()
    
    # 3. 噪聲穩定性正則化
    noise_norm_loss = torch.norm(output - e, p=2)
    
    # 自適應權重計算
    # 根據損失的大小和變化動態調整權重
    base_weight = torch.sigmoid(base_loss)  # 將損失映射到[0,1]區間
    adaptive_weight = torch.sigmoid(weighted_adaptive_loss)
    noise_weight = torch.sigmoid(noise_norm_loss)
    
    # 歸一化權重
    total_weight = base_weight + adaptive_weight + noise_weight
    normalized_base_weight = base_weight / total_weight
    normalized_adaptive_weight = adaptive_weight / total_weight
    normalized_noise_weight = noise_weight / total_weight
    
    # 組合最終損失
    final_loss = (
        normalized_base_weight * base_loss +  # 基礎損失
        normalized_adaptive_weight * weighted_adaptive_loss +  # 時間步驟自適應損失
        normalized_noise_weight * noise_norm_loss  # 噪聲穩定性正則項
    )
    
    return final_loss