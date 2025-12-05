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
    def __init__(self, map_size: Tuple[int, int] = (64, 64), latent_dim: int = 128, beta: float = 0.1):
        super().__init__()
        self.map_size = map_size
        self.latent_dim = latent_dim
        self.beta = beta
        
        # ===== Encoder =====
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),   # 16×16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 8×8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 4×4
            nn.ReLU(),
            nn.Flatten()
        )

        # 根据输入尺寸动态推算 flatten 维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *map_size)
            flat_dim = self.encoder(dummy).shape[1]
        
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        
        # ===== Decoder =====
        self.decoder_input = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 8→16
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self._flat_dim = flat_dim
    
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
        h = h.view(-1, 128, 4, 4)
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
    适配 128×128 输入的兴趣度 map 编码器（β-VAE 结构）
    """
    def __init__(self, map_size: Tuple[int, int] = (128, 128),
                 latent_dim: int = 128, beta: float = 0.1):
        super().__init__()
        self.map_size = map_size
        self.latent_dim = latent_dim
        self.beta = beta

        # ===== Encoder =====
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),    # 128×128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   # 64×64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32×32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 16×16
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1), # 8×8
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *map_size)
            # print(f"dummy {dummy.shape}")
            flat_dim = self.encoder(dummy).shape[1]

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # ===== Decoder =====
        self.decoder_input = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 64→128
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self._flat_dim = flat_dim

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon: torch.Tensor, x: torch.Tensor, 
             mu: torch.Tensor, logvar: torch.Tensor, beta = 0.1) -> torch.Tensor:
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
    
    return recon_loss + beta * kld_loss
