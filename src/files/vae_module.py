"""
VAE模块：用于预训练感知质量map编码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PerceptionMapVAE(nn.Module):
    """
    变分自编码器用于学习感知质量map的紧凑表示
    """
    def __init__(self, map_size: Tuple[int, int] = (64, 64), latent_dim: int = 128):
        super().__init__()
        self.map_size = map_size
        self.latent_dim = latent_dim
        
        # ====== Encoder ======
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # 均值和方差
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # ====== Decoder ======
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码为潜在分布
        Args:
            x: [B, 1, H, W]
        Returns:
            mu, logvar: [B, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码
        Args:
            z: [B, latent_dim]
        Returns:
            recon: [B, 1, H, W]
        """
        h = self.decoder_input(z)
        h = h.view(-1, 256, 4, 4)
        recon = self.decoder(h)
        return recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: [B, 1, H, W]
        Returns:
            recon: [B, 1, H, W]
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取潜在表示（用于下游任务）
        Args:
            x: [B, 1, H, W]
        Returns:
            z: [B, latent_dim]
        """
        mu, _ = self.encode(x)
        return mu


class InterestMapEncoder(nn.Module):
    """
    兴趣度map编码器
    """
    def __init__(self, map_size: Tuple[int, int] = (64, 64), output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # 全连接
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] 兴趣度map
        Returns:
            features: [B, output_dim]
        """
        return self.encoder(x)


def vae_loss(recon: torch.Tensor, x: torch.Tensor, 
             mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    VAE损失函数
    Args:
        recon: [B, 1, H, W] 重构
        x: [B, 1, H, W] 原始
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
    Returns:
        loss: 标量
    """
    # 重构损失
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    
    # KL散度
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kld_loss
