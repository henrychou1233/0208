import torch
from torchmetrics import ROC, AUROC, F1Score, AveragePrecision
import os
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from skimage import measure
from statistics import mean
import time
from sklearn.metrics import auc, confusion_matrix, classification_report, accuracy_score
from sklearn import metrics
from sklearn.cluster import KMeans

def calculate_fps_latency(func):
    """計算FPS和延遲時間的裝飾器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        latency = execution_time * 1000
        fps = 1.0 / execution_time if execution_time > 0 else 0
        
        print(f"\n=== 效能指標 ===")
        print(f"FPS: {fps:.2f}")
        print(f"延遲時間: {latency:.2f} ms")
        return result
    return wrapper

def cluster_samples(predictions, labels_list, predictions0_1, config):
    """對樣本進行分群分析"""
    anomaly_scores = predictions.cpu().numpy()
    
    # 使用KMeans進行分群
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(anomaly_scores.reshape(-1, 1))
    
    # 根據中心點排序群集
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)
    mapping = {old: new for new, old in enumerate(sorted_indices)}
    cluster_labels = np.array([mapping[label] for label in cluster_labels])
    sorted_centers = centers[sorted_indices]
    
    # 定義群集類型
    cluster_types = {
        0: "低異常分數群集 (正常)",
        1: "中異常分數群集",
        2: "高異常分數群集 (異常)"
    }
    
    print("\n=== 樣本分群分析 ===")
    print("-" * 50)
    
    # 顯示群集中心點
    print("\n群集中心點:")
    for i, center in enumerate(sorted_centers):
        print(f"{cluster_types[i]}: {center:.4f}")
    
    # 分析每個群集
    for cluster_id in range(3):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_scores = anomaly_scores[cluster_mask]
        cluster_true_labels = labels_list[cluster_mask].cpu().numpy()
        cluster_pred_labels = predictions0_1[cluster_mask].cpu().numpy()
        
        if len(cluster_scores) > 0:
            print(f"\n{cluster_types[cluster_id]}")
            print("-" * 30)
            print(f"樣本數量: {len(cluster_scores)}")
            print(f"分數範圍: {cluster_scores.min():.4f} - {cluster_scores.max():.4f}")
            print(f"平均分數: {cluster_scores.mean():.4f}")
            print(f"中位數: {np.median(cluster_scores):.4f}")
            print(f"標準差: {cluster_scores.std():.4f}")
            
            # 計算真實標籤分布
            normal_count = np.sum(cluster_true_labels == 0)
            anomaly_count = np.sum(cluster_true_labels == 1)
            print(f"正常樣本數: {normal_count}")
            print(f"異常樣本數: {anomaly_count}")
            
            # 列出樣本詳情
            print("\n樣本詳情:")
            print("索引    異常分數    真實標籤    預測標籤    預測狀態")
            print("-" * 60)
            
            # 獲取原始索引
            original_indices = np.where(cluster_mask)[0]
            for idx, score, true_label, pred_label in zip(
                original_indices, cluster_scores, cluster_true_labels, cluster_pred_labels):
                pred_status = "正確" if true_label == pred_label else "錯誤"
                true_label_str = "正常" if true_label == 0 else "異常"
                pred_label_str = "正常" if pred_label == 0 else "異常"
                print(f"{idx:<7d} {score:.4f}    {true_label_str:<8s} {pred_label_str:<8s} {pred_status}")

def metric(labels_list, predictions, anomaly_map_list, GT_list, config):
    """計算評估指標"""
    labels_list = torch.tensor(labels_list)
    predictions = torch.tensor(predictions)
    pro = compute_pro(GT_list, anomaly_map_list, num_th=200)
    
    results_embeddings = anomaly_map_list[0]
    for feature in anomaly_map_list[1:]:
        results_embeddings = torch.cat((results_embeddings, feature), 0)
    results_embeddings = ((results_embeddings - results_embeddings.min()) / 
                         (results_embeddings.max() - results_embeddings.min()))
    
    GT_embeddings = GT_list[0]
    for feature in GT_list[1:]:
        GT_embeddings = torch.cat((GT_embeddings, feature), 0)

    results_embeddings = results_embeddings.clone().detach().requires_grad_(False)
    GT_embeddings = GT_embeddings.clone().detach().requires_grad_(False)

    roc = ROC(task="binary")
    auroc = AUROC(task="binary")
    fpr, tpr, thresholds = roc(predictions, labels_list)
    auroc_score = auroc(predictions, labels_list)

    GT_embeddings = torch.flatten(GT_embeddings).type(torch.bool).cpu().detach()
    results_embeddings = torch.flatten(results_embeddings).cpu().detach()
    auroc_pixel = auroc(results_embeddings, GT_embeddings)
    
    thresholdOpt_index = torch.argmax(tpr - fpr)
    thresholdOpt = thresholds[thresholdOpt_index]

    f1 = F1Score(task="binary")
    ap = AveragePrecision(task="binary")
    ap_image = ap(predictions, labels_list)
    ap_pixel = ap(results_embeddings, GT_embeddings)
    
    predictions0_1 = (predictions > thresholdOpt).int()
    
    for i, (l, p) in enumerate(zip(labels_list, predictions0_1)):
        if l != p:
            print(f'樣本 {i}: 預測值={p.item()}, 實際值={l.item()}, 預測分數={predictions[i].item()}')

    f1_score = f1(predictions0_1, labels_list)

    print("\n=== 評估指標 ===")
    if config.metrics.image_level_AUROC:
        print(f'影像層級 AUROC: {auroc_score:.4f}')
    if config.metrics.pixel_level_AUROC:
        print(f"像素層級 AUROC: {auroc_pixel:.4f}")
    if config.metrics.image_level_F1Score:
        print(f'F1分數: {f1_score:.4f}')
    if config.metrics.pro:
        print(f'PRO分數: {pro:.4f}')
        print(f"影像層級平均精確度: {ap_image:.4f}")
        print(f"像素層級平均精確度: {ap_pixel:.4f}")

    cm = confusion_matrix(labels_list.cpu(), predictions0_1.cpu())
    print("\n=== 混淆矩陣 ===")
    print("預測        Normal  Anomaly")
    print(f"實際 Normal  {cm[0,0]:<6d} {cm[0,1]:<6d}")
    print(f"實際 Anomaly {cm[1,0]:<6d} {cm[1,1]:<6d}")
    
    # 執行分群分析
    cluster_samples(predictions, labels_list, predictions0_1, config)
    
    with open('readme.txt', 'a') as f:
        f.write(f"{config.data.category}\n")
        f.write(f"AUROC: {auroc_score:.4f} | AUROC_pixel: {auroc_pixel:.4f} | "
                f"F1_SCORE: {f1_score:.4f} | PRO_AUROC: {pro:.4f}\n")
    
    roc = roc.reset()
    auroc = auroc.reset()
    f1 = f1.reset()
    
    return thresholdOpt

@calculate_fps_latency
def compute_pro(masks, amaps, num_th=200):
    """計算PRO (Per-Region Overlap)分數"""
    results_embeddings = amaps[0]
    for feature in amaps[1:]:
        results_embeddings = torch.cat((results_embeddings, feature), 0)
    amaps = ((results_embeddings - results_embeddings.min()) /
             (results_embeddings.max() - results_embeddings.min()))
    
    amaps = amaps.squeeze(1).cpu().detach().numpy()
    gt_embeddings = masks[0]
    for feature in masks[1:]:
        gt_embeddings = torch.cat((gt_embeddings, feature), 0)
    masks = gt_embeddings.squeeze(1).cpu().detach().numpy()

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({
            "pro": mean(pros),
            "fpr": fpr,
            "threshold": th
        }, index=[0])], ignore_index=True)

    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc