"""
Q值生成器模块
从车辆嵌入生成Q值map
"""
import torch
import torch.nn as nn


class CNNQGenerator(nn.Module):
    """
    基于CNN Decoder的Q值map生成器
    从车辆嵌入向量上采样为2D Q值map
    """
    def __init__(
        self,
        vehicle_feature_dim: int,
        map_size: tuple = (64, 64)
    ):
        super().__init__()
        self.map_size = map_size
        
        # 投影到初始feature map
        self.fc = nn.Sequential(
            nn.Linear(vehicle_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 4 * 4)
        )
        
        # 上采样解码器
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
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 输出层
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, vehicle_features: torch.Tensor) -> torch.Tensor:
        """
        生成Q值map
        
        Args:
            vehicle_features: [N_v, vehicle_feature_dim]
        
        Returns:
            q_maps: [N_v, H, W] 每辆车的Q值map
        """
        N_v = vehicle_features.shape[0]
        
        # 投影到feature map
        x = self.fc(vehicle_features)  # [N_v, 256*4*4]
        x = x.view(N_v, 256, 4, 4)  # [N_v, 256, 4, 4]
        
        # 解码上采样
        x = self.decoder(x)  # [N_v, 1, H, W]
        
        # 去掉通道维度
        q_maps = x.squeeze(1)  # [N_v, H, W]
        
        return q_maps
