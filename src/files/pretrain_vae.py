"""
VAE预训练脚本
用于预训练感知质量map编码器
"""
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

from config import Config
from vae_module import PerceptionMapVAE, vae_loss
from environment import VehicularEnvironment


class PerceptionMapDataset(Dataset):
    """感知质量map数据集"""
    def __init__(self, config, num_samples=10000):
        self.config = config
        self.samples = []
        
        print("Generating perception map dataset...")
        env = VehicularEnvironment(config)
        
        for _ in tqdm(range(num_samples)):
            state = env.reset()
            
            # 收集多个时间步
            for _ in range(20):
                # 随机动作
                action = torch.rand(config.NUM_VEHICLES, config.MAP_HEIGHT, config.MAP_WIDTH)
                action = (action > 0.75).float()  # top-25%
                
                state, _, done, _ = env.step(action)
                
                # 保存感知maps
                perception_maps = state['perception_maps_history'][:, -1, :, :]  # [N_v, H, W]
                for i in range(config.NUM_VEHICLES):
                    self.samples.append(perception_maps[i].unsqueeze(0))  # [1, H, W]
                
                if done:
                    break
        
        print(f"Dataset size: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def train_vae(config):
    """训练VAE"""
    print("\n" + "="*60)
    print("VAE Pretraining")
    print("="*60)
    
    # 创建模型
    vae = PerceptionMapVAE(
        map_size=(config.MAP_HEIGHT, config.MAP_WIDTH),
        latent_dim=config.VAE_LATENT_DIM
    ).to(config.DEVICE)
    
    # 优化器
    optimizer = optim.Adam(vae.parameters(), lr=config.VAE_PRETRAIN_LR)
    
    # 数据集
    dataset = PerceptionMapDataset(config, num_samples=5000)
    dataloader = DataLoader(
        dataset,
        batch_size=config.VAE_PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    # 训练循环
    vae.train()
    for epoch in range(config.VAE_PRETRAIN_EPOCHS):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.VAE_PRETRAIN_EPOCHS}")
        
        for batch in pbar:
            batch = batch.to(config.DEVICE)
            
            # 前向传播
            recon, mu, logvar = vae(batch)
            
            # 计算损失
            loss = vae_loss(recon, batch, mu, logvar) / batch.size(0)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'vae_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # 保存最终模型
    final_path = config.VAE_CHECKPOINT_PATH
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({
        'epoch': config.VAE_PRETRAIN_EPOCHS,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, final_path)
    
    print(f"\n✓ VAE training complete! Saved to {final_path}")
    
    return vae


if __name__ == '__main__':
    config = Config()
    config.print_config()
    
    train_vae(config)
