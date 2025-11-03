"""
特征提取和融合模块
处理感知质量map + 兴趣度map + 位置信息
"""
import torch
import torch.nn as nn
from vae_module import PerceptionMapVAE, InterestMapEncoder


class ComprehensiveFeatureExtractor(nn.Module):
    """
    综合特征提取器
    融合：感知质量map（车辆视角） + 兴趣度map（全局视角） + 位置信息
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ====== 1. 感知质量map编码器（VAE） ======
        self.perception_vae = PerceptionMapVAE(
            map_size=(config.MAP_HEIGHT, config.MAP_WIDTH),
            latent_dim=config.VAE_LATENT_DIM
        )
        
        # ====== 2. 兴趣度map编码器 ======
        self.interest_encoder = InterestMapEncoder(
            map_size=(config.MAP_HEIGHT, config.MAP_WIDTH),
            output_dim=config.INTEREST_ENCODER_DIM
        )
        
        # ====== 3. 位置编码器 ======
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, config.POSITION_ENCODER_DIM),
            nn.ReLU()
        )
        
        # ====== 4. 特征融合网络 ======
        total_dim = (config.VAE_LATENT_DIM + 
                    config.INTEREST_ENCODER_DIM + 
                    config.POSITION_ENCODER_DIM)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, config.FUSED_FEATURE_DIM * 2),
            nn.LayerNorm(config.FUSED_FEATURE_DIM * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.FUSED_FEATURE_DIM * 2, config.FUSED_FEATURE_DIM),
            nn.LayerNorm(config.FUSED_FEATURE_DIM),
            nn.ReLU()
        )
    
    def forward(
        self,
        perception_maps: torch.Tensor,
        interest_map: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        提取并融合特征
        
        Args:
            perception_maps: [N_v, 1, H, W] 每辆车的感知质量map
            interest_map: [1, H, W] 全局兴趣度map
            positions: [N_v, 2] 车辆位置
        
        Returns:
            fused_features: [N_v, fused_feature_dim] 融合后的特征
        """
        N_v = perception_maps.shape[0]
        
        # ------ 1. 编码感知质量map（个体特征） ------
        perception_features = self.perception_vae.get_latent(perception_maps)  # [N_v, 128]
        
        # ------ 2. 编码兴趣度map（全局特征 → 复制给每辆车） ------
        interest_features = self.interest_encoder(interest_map.unsqueeze(0))  # [1, 64]
        interest_features = interest_features.repeat(N_v, 1)  # [N_v, 64]
        
        # ------ 3. 编码位置 ------
        position_features = self.position_encoder(positions)  # [N_v, 32]
        
        # ------ 4. 拼接 ------
        combined = torch.cat([
            perception_features,
            interest_features,
            position_features
        ], dim=-1)  # [N_v, 128+64+32=224]
        
        # ------ 5. 融合 ------
        fused_features = self.feature_fusion(combined)  # [N_v, 256]
        
        return fused_features
    
    def load_pretrained_vae(self, checkpoint_path: str):
        """加载预训练的VAE权重"""
        checkpoint = torch.load(checkpoint_path)
        self.perception_vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded pretrained VAE from {checkpoint_path}")
        
        # 冻结VAE编码器
        for param in self.perception_vae.encoder.parameters():
            param.requires_grad = False
        for param in self.perception_vae.fc_mu.parameters():
            param.requires_grad = False
        for param in self.perception_vae.fc_logvar.parameters():
            param.requires_grad = False
        
        print("✓ VAE encoder frozen")
