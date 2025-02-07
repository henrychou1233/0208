import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torchvision.transforms import transforms
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from torch import Tensor

class AdaptiveWeightCalculator:
    """自適應權重計算器"""
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def calculate_weights(self, maps: List[Tensor]) -> Tensor:
        """計算自適應權重"""
        # 計算每個圖的顯著性得分
        scores = [torch.mean(torch.abs(m)) for m in maps]
        scores = torch.tensor(scores)
        
        # 使用 softmax 獲得權重
        weights = F.softmax(scores / self.temperature, dim=0)
        return weights

def recon_heat_map(output: Tensor, 
                   target: Tensor, 
                   config: object, 
                   detail_enhance: bool = False,
                   frequency_enhance: bool = False) -> Tensor:
    """
    生成重建熱圖，增加頻率域增強
    Args:
        output: 模型輸出張量
        target: 目標張量 
        config: 配置對象
        detail_enhance: 是否啟用細節增強
        frequency_enhance: 是否啟用頻率域增強
    Returns:
        anomaly_map: 異常檢測熱圖
    """
    sigma = 4.0
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    
    output = output.to(config.model.device)
    target = target.to(config.model.device)
    
    # 基礎顏色距離圖
    ano_map = color_distance(output, target, config, config.data.image_size)

    if frequency_enhance:
        # 添加頻率域分析
        freq_map = frequency_domain_analysis(output, target)
        ano_map = 0.7 * ano_map + 0.3 * freq_map

    if detail_enhance:
        maps = []
        scale_params = [
            (sigma * 0.5, 0.3),
            (sigma, 0.4),
            (sigma * 2, 0.3)
        ]
        
        for curr_sigma, weight in scale_params:
            kernel = 2 * int(4 * curr_sigma + 0.5) + 1
            blurred = gaussian_blur2d(
                ano_map,
                kernel_size=(kernel, kernel),
                sigma=(curr_sigma, curr_sigma)
            )
            maps.append(blurred * weight)
        
        ano_map = torch.stack(maps).sum(dim=0)
    else:
        ano_map = gaussian_blur2d(
            ano_map,
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma)
        )
    
    return torch.sum(ano_map, dim=1, keepdim=True)

def frequency_domain_analysis(image1: Tensor, image2: Tensor) -> Tensor:
    """
    頻率域分析
    """
    # 轉換到頻率域
    fft1 = torch.fft.fft2(image1)
    fft2 = torch.fft.fft2(image2)
    
    # 計算頻率域差異
    freq_diff = torch.abs(fft1 - fft2)
    
    # 轉回空間域
    return torch.abs(torch.fft.ifft2(freq_diff))

def feature_heat_map(output: Tensor, 
                     target: Tensor, 
                     fe: torch.nn.Module, 
                     config: object, 
                     use_attention: bool = False,
                     use_multi_scale: bool = False) -> Tensor:
    """
    增強的特徵熱圖生成
    新增多尺度特徵聚合
    """
    sigma = 4.0
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    
    output = output.to(config.model.device)
    target = target.to(config.model.device)
    
    with torch.no_grad():
        f_d = feature_distance_new(output, target, fe, config, use_attention)
        if use_multi_scale:
            # 多尺度特徵聚合
            scales = [1.0, 0.75, 0.5]
            multi_scale_maps = []
            for scale in scales:
                size = int(output.shape[-1] * scale)
                scaled_output = F.interpolate(output, size=size)
                scaled_target = F.interpolate(target, size=size)
                scaled_map = feature_distance_new(scaled_output, scaled_target, fe, config, use_attention)
                scaled_map = F.interpolate(scaled_map, size=output.shape[-1])
                multi_scale_maps.append(scaled_map)
            
            f_d = sum(multi_scale_maps) / len(scales)
    
    f_d = f_d.to(config.model.device)
    anomaly_map = gaussian_blur2d(
        f_d,
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma)
    )
    
    return torch.sum(anomaly_map, dim=1, keepdim=True)

def heatmap_latent(l1_latent: List[Tensor], 
                   cos_list: List[Tensor], 
                   config: object,
                   dynamic_weight: bool = False,
                   channel_attention: bool = False) -> List[Tensor]:
    """
    增強的潛在空間熱圖生成
    新增通道注意力機制
    """
    sigma = 4.0
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    heatmap_latent_list = []

    for l1_map, cos_map in zip(l1_latent, cos_list):
        if dynamic_weight:
            confidence = torch.mean(cos_map)
            weight = config.model.anomap_weighting * (1 + confidence) / 2
        else:
            weight = config.model.anomap_weighting

        if channel_attention:
            # 應用通道注意力
            channel_weights = F.softmax(torch.mean(l1_map, dim=[-2, -1]), dim=1)
            l1_map = l1_map * channel_weights.unsqueeze(-1).unsqueeze(-1)
            cos_map = cos_map * channel_weights.unsqueeze(-1).unsqueeze(-1)

        anomaly_map = weight * l1_map + (1 - weight) * cos_map
        
        anomaly_map = gaussian_blur2d(
            anomaly_map,
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma)
        )
        
        heatmap_latent_list.append(torch.sum(anomaly_map, dim=1, keepdim=True))
    
    return heatmap_latent_list

def color_distance(image1: Tensor, 
                   image2: Tensor, 
                   config: object, 
                   out_size: int = 256) -> Tensor:
    """計算顏色距離，保持原有實現"""
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if config.model.latent:
        image1 = image1.to(config.model.device)
        image2 = image2.to(config.model.device)
        
        distance_map = torch.mean(torch.abs(image1 - image2), dim=1, keepdim=True)
        
        return F.interpolate(
            distance_map,
            size=out_size,
            mode='bilinear',
            align_corners=True
        )
    
    image1, image2 = transform(image1), transform(image2)
    return torch.mean(torch.abs(image1 - image2), dim=1, keepdim=True)

def feature_distance_new(output: Tensor, 
                         target: Tensor, 
                         FE: torch.nn.Module, 
                         config: object,
                         use_attention: bool = False) -> Tensor:
    """增強的特徵距離計算，添加了更多注意力選項"""
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    output = transform(output.to(config.model.device))
    target = transform(target.to(config.model.device))
    
    with torch.no_grad():
        inputs_features = FE(target)
        output_features = FE(output)
    
    out_size = config.data.image_size
    anomaly_map = torch.zeros(
        [inputs_features[0].shape[0], 1, out_size, out_size],
        device=config.model.device
    )
    
    for i, (input_feat, output_feat) in enumerate(zip(inputs_features, output_features)):
        if i in config.model.anomap_excluded_layers:
            continue
            
        if use_attention:
            # 增強的注意力機制
            spatial_attention = F.softmax(torch.norm(input_feat, p=2, dim=1), dim=0)
            channel_attention = F.softmax(torch.norm(input_feat, p=2, dim=[-2, -1]), dim=1)
            
            similarity = 1 - F.cosine_similarity(input_feat, output_feat)
            a_map = similarity * spatial_attention * channel_attention.unsqueeze(-1)
        else:
            a_map = 1 - F.cosine_similarity(input_feat, output_feat)
            
        a_map = F.interpolate(
            a_map.unsqueeze(1),
            size=out_size,
            mode='bilinear',
            align_corners=True
        )
        anomaly_map += a_map
        
    return anomaly_map

def calculate_min_max_of_tensors(tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """計算張量列表的最小值和最大值"""
    min_values = torch.stack([tensor.min() for tensor in tensors])
    max_values = torch.stack([tensor.max() for tensor in tensors])
    return torch.min(min_values), torch.max(max_values)

def scale_values_between_zero_and_one(tensors: List[Tensor]) -> List[Tensor]:
    """張量歸一化"""
    min_value, max_value = calculate_min_max_of_tensors(tensors)
    return [(tensor - min_value) / (max_value - min_value + 1e-8) for tensor in tensors]

def fuse_heat_maps(recon_map: Tensor,
                   feature_map: Tensor,
                   latent_maps: List[Tensor],
                   weights: Optional[Dict[str, float]] = None,
                   adaptive_fusion: bool = False,
                   deep_supervision: bool = False) -> Tensor:
    """
    增強的熱圖融合
    Args:
        recon_map: 重建熱圖
        feature_map: 特徵熱圖
        latent_maps: 潛在空間熱圖列表
        weights: 融合權重字典
        adaptive_fusion: 是否使用自適應融合
        deep_supervision: 是否使用深度監督
    Returns:
        fused_map: 融合後的熱圖
    """
    if weights is None:
        weights = {'recon': 0.3, 'feature': 0.4, 'latent': 0.3}

    # 標準化所有圖
    maps_to_fuse = [
        recon_map,
        feature_map,
        sum(latent_maps)
    ]
    
    if deep_supervision:
        # 添加深度監督分支
        supervised_maps = []
        for m in maps_to_fuse:
            sup_map = apply_deep_supervision(m)
            supervised_maps.append(sup_map)
        maps_to_fuse = supervised_maps
    
    normalized_maps = scale_values_between_zero_and_one(maps_to_fuse)
    
    if adaptive_fusion:
        # 使用自適應權重
        weight_calculator = AdaptiveWeightCalculator()
        adaptive_weights = weight_calculator.calculate_weights(normalized_maps)
        fused_map = sum(w * m for w, m in zip(adaptive_weights, normalized_maps))
    else:
        # 使用固定權重
        fused_map = (
            weights['recon'] * normalized_maps[0] +
            weights['feature'] * normalized_maps[1] +
            weights['latent'] * normalized_maps[2]
        )
    
    return (fused_map - fused_map.min()) / (fused_map.max() - fused_map.min() + 1e-8)

def apply_deep_supervision(feature_map: Tensor) -> Tensor:
    """
    應用深度監督
    Args:
        feature_map: 輸入特徵圖
    Returns:
        supervised_map: 深度監督後的特徵圖
    """
    # 多尺度特徵提取
    scales = [1.0, 0.75, 0.5]
    supervised_maps = []
    
    for scale in scales:
        size = int(feature_map.shape[-1] * scale)
        scaled_map = F.interpolate(feature_map, size=size, mode='bilinear', align_corners=True)
        
        # 添加殘差連接
        if scale != 1.0:
            scaled_map = scaled_map + F.interpolate(feature_map, size=size, mode='bilinear', align_corners=True)
            
        scaled_map = F.interpolate(scaled_map, size=feature_map.shape[-1], mode='bilinear', align_corners=True)
        supervised_maps.append(scaled_map)
    
    # 融合多尺度特徵
    return sum(supervised_maps) / len(supervised_maps)