import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class AdaptiveForwardDiffusion:
    def __init__(self, config):
        self.config = config
        self.device = config.model.device
        self.noise_generator = AdaptiveNoiseGenerator(config)
        
    def get_index_from_list(self, vals: torch.Tensor, t: torch.Tensor, x_shape: tuple, config) -> torch.Tensor:
        """從列表中獲取時間步索引的輔助函數"""
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(config.model.device)

    def compute_alpha(self, beta: torch.Tensor, t: int, config) -> torch.Tensor:
        """計算累積 α 值"""
        beta = beta.to(self.device)
        return torch.prod(1. - beta[:t + 1])

    def forward_diffusion_sample(self, 
                               x_0: torch.Tensor, 
                               t: torch.Tensor, 
                               constant_dict: Dict[str, torch.Tensor], 
                               config,
                               noise_type: str = 'adaptive_gaussian') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自適應前向擴散採樣
        
        Args:
            x_0: 初始圖像
            t: 時間步
            constant_dict: 包含預計算常數的字典
            config: 配置對象
            noise_type: 噪聲類型
            
        Returns:
            x: 噪聲化後的圖像
            noise: 添加的噪聲
        """
        sqrt_alphas_cumprod = constant_dict['sqrt_alphas_cumprod']
        sqrt_one_minus_alphas_cumprod = constant_dict['sqrt_one_minus_alphas_cumprod']
        
        # 使用自適應噪聲生成器
        noise = self.noise_generator.get_noise(x_0, noise_type=noise_type)
        
        # 獲取對應時間步的值
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            sqrt_alphas_cumprod, t, x_0.shape, config
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x_0.shape, config
        )
        
        # 計算噪聲圖像
        x = (sqrt_alphas_cumprod_t * x_0 + 
             sqrt_one_minus_alphas_cumprod_t * noise)
             
        return x.to(self.device), noise.to(self.device)

    def forward_ti_steps(self, 
                        t: int, 
                        ti: int, 
                        x_t_ti: torch.Tensor, 
                        x_0: torch.Tensor, 
                        beta: torch.Tensor, 
                        config,
                        noise_type: str = 'adaptive_gaussian') -> torch.Tensor:
        """
        自適應前向時間插值步驟
        
        Args:
            t: 目標時間步
            ti: 時間步增量
            x_t_ti: 當前噪聲圖像
            x_0: 初始圖像
            beta: beta 調度
            config: 配置對象
            noise_type: 噪聲類型
            
        Returns:
            x_t: 更新後的噪聲圖像
        """
        x_t_ti = x_t_ti.to(self.device)
        
        # 使用自適應噪聲生成器
        noise = self.noise_generator.get_noise(x_t_ti, noise_type=noise_type)
        
        # 計算 alpha 值
        alpha_t = self.compute_alpha(beta, t, config)
        alpha_t_ti = self.compute_alpha(beta, t-ti, config)
        
        # 計算插值係數
        sqrt_alpha_diff = alpha_t.sqrt() - alpha_t_ti.sqrt()
        sqrt_one_minus_alpha_diff = (1 - alpha_t).sqrt() - (1 - alpha_t_ti).sqrt()
        
        # 計算更新後的噪聲圖像
        x_t = (x_t_ti + 
               sqrt_alpha_diff * x_0 + 
               sqrt_one_minus_alpha_diff * noise)
               
        return x_t

class AdaptiveNoiseGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config.model.device
        self.noise_history = []
        self.adaptation_rate = 0.01
        self.min_std = 0.01
        self.max_std = 2.0
        
    def calculate_adaptive_std(self, x: torch.Tensor) -> torch.Tensor:
        """計算自適應標準差"""
        if len(self.noise_history) > 0:
            recent_noise = torch.stack(self.noise_history[-100:])
            current_std = torch.std(recent_noise)
            signal_std = torch.std(x)
            
            adaptive_std = torch.clamp(
                signal_std * self.adaptation_rate,
                self.min_std,
                self.max_std
            )
            return adaptive_std
        return torch.tensor(1.0).to(self.device)

    def get_noise(self, x: torch.Tensor, noise_type: str = 'adaptive_gaussian', seed: Optional[int] = None) -> torch.Tensor:
        """生成自適應噪聲"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if noise_type == 'adaptive_gaussian':
            std = self.calculate_adaptive_std(x)
            noise = torch.randn_like(x).to(self.device) * std
        else:
            noise = torch.randn_like(x).to(self.device)

        # 更新噪聲歷史
        if len(self.noise_history) >= 100:
            self.noise_history.pop(0)
        self.noise_history.append(noise.detach())

        return noise

# 主函數介面
def forward_diffusion_sample(x_0, t, constant_dict, config, noise_type='adaptive_gaussian'):
    """主要的前向擴散採樣函數"""
    diffusion = AdaptiveForwardDiffusion(config)
    return diffusion.forward_diffusion_sample(x_0, t, constant_dict, config, noise_type)

def forward_ti_steps(t, ti, x_t_ti, x_0, beta, config, noise_type='adaptive_gaussian'):
    """主要的前向時間插值步驟函數"""
    diffusion = AdaptiveForwardDiffusion(config)
    return diffusion.forward_ti_steps(t, ti, x_t_ti, x_0, beta, config, noise_type)