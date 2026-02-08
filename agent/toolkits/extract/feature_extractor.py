"""
特征提取和融合模块
处理感知质量map + 兴趣度map + 位置信息
"""
import torch
import torch.nn as nn
from src.config import Config
from src.extract.vae_module import PerceptionMapVAE, InterestMapEncoder


class ComprehensiveFeatureExtractor(nn.Module):
    """
    综合特征提取器
    融合：感知质量map（车辆视角） + 兴趣度map（全局视角） + 位置信息
    """
    def __init__(self, env_config, model_config:Config):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        
        # ====== 1. 感知质量map编码器（VAE） ======
        self.perception_vae = PerceptionMapVAE(
            map_size=(model_config.MAP_HEIGHT, model_config.MAP_WIDTH),
            latent_dim=model_config.VAE_LATENT_DIM
        )
        
        # ====== 2. 兴趣度map编码器 ======
        self.interest_encoder = InterestMapEncoder(
            map_size=(model_config.GLOBAL_MAP_H, model_config.GLOBAL_MAP_W),
            latent_dim=model_config.INTEREST_ENCODER_DIM
        )
        
        # ====== 3. 位置编码器 ======
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, model_config.POSITION_ENCODER_DIM),
            nn.ReLU()
        )
        
        # ====== 4. 特征融合网络 ======
        total_dim = (model_config.VAE_LATENT_DIM *2 +   # local + fused
                    model_config.INTEREST_ENCODER_DIM + 
                    model_config.POSITION_ENCODER_DIM)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, model_config.FUSED_FEATURE_DIM * 2),
            nn.LayerNorm(model_config.FUSED_FEATURE_DIM * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_config.FUSED_FEATURE_DIM * 2, model_config.FUSED_FEATURE_DIM),
            nn.LayerNorm(model_config.FUSED_FEATURE_DIM),
            nn.ReLU()
        )
    
    def forward(
        self,
        local_maps: torch.Tensor,
        fused_maps: torch.Tensor,
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
        N_v = local_maps.shape[0]
        
        # ------ 1. 编码感知质量map（个体特征） ------
        local_features = self.perception_vae.get_latent(local_maps)  # [N_v, 32]
        fused_features = self.perception_vae.get_latent(fused_maps)  # [N_v, 32]

        # ------ 2. 编码兴趣度map（全局特征 → 复制给每辆车） ------
        interest_features = self.interest_encoder.get_latent(interest_map.unsqueeze(0))  # [1, 64]
        interest_features = interest_features.repeat(N_v, 1)  # [N_v, 32]
        
        # ------ 3. 编码位置 ------
        position_features = self.position_encoder(positions)  # [N_v, 32]
        
        # print(f"local_features: {local_features.shape}, fused_features: {fused_features.shape}, \
                    # interest_features: {interest_features.shape}, position_features: {position_features.shape}")

        # ------ 4. 拼接 ------
        combined = torch.cat([
            local_features,
            fused_features,
            interest_features,
            position_features
        ], dim=-1)  # [N_v, 32+32+32+32=128]
        
        # ------ 5. 融合 ------
        fused_features = self.feature_fusion(combined)  # [N_v, 128]
        
        return fused_features
    
    def load_pretrained_vae(self, checkpoint_path: str, checkpoint_n_path: str):
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

        checkpoint_n = torch.load(checkpoint_n_path)
        self.interest_encoder.load_state_dict(checkpoint_n['model_state_dict'])
        print(f"✓ Loaded pretrained interested map VAE from {checkpoint_n_path}")
        
        # 冻结VAE编码器
        for param in self.interest_encoder.encoder.parameters():
            param.requires_grad = False
        for param in self.interest_encoder.fc_mu.parameters():
            param.requires_grad = False
        for param in self.interest_encoder.fc_logvar.parameters():
            param.requires_grad = False
        
        print("✓ interested map VAE encoder frozen")
