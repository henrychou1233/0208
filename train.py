import torch
import os
import torch.nn as nn
from forward_process import *
from dataset import *
from diffusers import AutoencoderKL
from torch.optim import Adam
from dataset import *
from noise import *
from visualize import show_tensor_image

from test import *
from loss import *
from optimizer import *
from sample import *

def trainer(model, constants_dict, ema_helper, config):
    optimizer = build_optimizer(model, config)
    
    # 檢查是否有 checkpoint
    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
    os.makedirs(model_save_dir, exist_ok=True)

    # 自動載入最新的 checkpoint
    latest_checkpoint = None
    start_epoch = 0  # 預設從 0 開始

    checkpoint_files = sorted([
        f for f in os.listdir(model_save_dir) if f.endswith(".pth")
    ], key=lambda x: int(x.split("_")[-1]))  # 依據 epoch 數排序

    if checkpoint_files:
        latest_checkpoint = os.path.join(model_save_dir, checkpoint_files[-1])
        print(f"✅ 找到最新的 checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=config.model.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # 從下一個 epoch 開始
        print(f"🔄 從 epoch {start_epoch} 繼續訓練")

    # 訓練資料載入
    if config.data.name in ['MVTec', 'BTAD', 'MTD', 'VisA']:
        train_dataset = MVTecDataset(
            root=config.data.data_dir,
            category=config.data.category,
            config=config,
            is_train=True,
        )
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.model.num_workers,
            drop_last=True,
        )
    elif config.data.name == 'cifar10':
        trainloader, testloader = load_data(dataset_name='cifar10')

    # 設定 VAE 模型
    if config.model.latent:
        if config.model.latent_backbone == "VAE":
            vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
            vae.to(config.model.device)
            vae.eval()
        else:
            raise ValueError("error: backbone needs to be VAE")

    # 訓練迴圈
    best_loss = float('inf')
    for epoch in range(start_epoch, config.model.epochs):
        epoch_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(trainloader):
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
            optimizer.zero_grad()
            
            if config.model.latent:
                if config.model.latent_backbone == "VAE":     
                    features = vae.encode(batch[0].to(config.model.device)).latent_dist.sample() * 0.18215
                    loss = get_loss(model, constants_dict, features, t, config)
                else:
                    raise ValueError("error: backbone needs to be VAE")
            else:
                loss = get_loss(model, constants_dict, batch[0], t, config) 

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            step_count += 1

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
            
            # 每 25 個 epoch 存一次模型
            if epoch % 1 == 0 and step == 0 and config.model.save_model:
                save_path = os.path.join(
                    model_save_dir,
                    f"{config.model.latent_size}_{config.model.unet_channel}_{config.model.n_head}_{config.model.head_channel}_diffusers_unet_{epoch}.pth"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, save_path)

        avg_loss = epoch_loss / step_count
        print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss and config.model.save_model:
            best_loss = avg_loss
            save_path = os.path.join(
                model_save_dir,
                f"{config.model.latent_size}_{config.model.unet_channel}_{config.model.n_head}_{config.model.head_channel}_diffusers_unet_best.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, save_path)
                
    # 最後一個 epoch 存檔
    if config.model.save_model:
        final_save_path = os.path.join(
            model_save_dir,
            f"{config.model.latent_size}_{config.model.unet_channel}_{config.model.n_head}_{config.model.head_channel}_diffusers_unet_{config.model.epochs-1}.pth"
        )
        torch.save({
            "epoch": config.model.epochs-1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, final_save_path)

    return model
