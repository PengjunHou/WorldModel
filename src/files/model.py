"""
完整的ST-GAT Q网络模型
"""
import torch
import torch.nn as nn
from typing import Dict, List
from feature_extractor import ComprehensiveFeatureExtractor
from stgat_module import SpatioTemporalGAT, EdgeBuilder
from q_generator import CNNQGenerator


class STGATQNetwork(nn.Module):
    """
    完整的基于ST-GAT的Q网络
    
    Pipeline:
    1. 特征提取（感知map + 兴趣map + 位置） → [N_v, T, feature_dim]
    2. ST-GAT（时空建模） → [N_v, hidden_dim]
    3. Q值生成器 → [N_v, H, W]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ====== 模块1：特征提取器 ======
        self.feature_extractor = ComprehensiveFeatureExtractor(config)
        
        # ====== 模块2：ST-GAT ======
        self.stgat = SpatioTemporalGAT(
            feature_dim=config.FUSED_FEATURE_DIM,
            hidden_dim=config.STGAT_HIDDEN_DIM,
            num_spatial_layers=config.STGAT_NUM_SPATIAL_LAYERS,
            num_temporal_layers=config.STGAT_NUM_TEMPORAL_LAYERS,
            num_heads=config.STGAT_NUM_HEADS,
            dropout=config.STGAT_DROPOUT
        )
        
        # ====== 模块3：Q值生成器 ======
        self.q_generator = CNNQGenerator(
            vehicle_feature_dim=config.STGAT_HIDDEN_DIM,
            map_size=(config.MAP_HEIGHT, config.MAP_WIDTH)
        )
        
        # ====== 边构建器 ======
        self.edge_builder = EdgeBuilder()
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            batch_data: {
                'perception_maps_history': [N_v, T, H, W],
                'interest_maps_history': [T, H, W],
                'positions_history': [N_v, T, 2]
            }
        
        Returns:
            q_maps: [N_v, H, W] 每辆车的Q值map
        """
        N_v = batch_data['perception_maps_history'].shape[0]
        T = batch_data['perception_maps_history'].shape[1]
        
        # ====== 步骤1：逐时间步提取特征 ======
        features_history = []
        edge_indices_history = []
        
        for t in range(T):
            # 当前时刻的数据
            perception_maps_t = batch_data['perception_maps_history'][:, t]  # [N_v, H, W]
            interest_map_t = batch_data['interest_maps_history'][t]  # [H, W]
            positions_t = batch_data['positions_history'][:, t]  # [N_v, 2]
            
            # 添加通道维度
            perception_maps_t = perception_maps_t.unsqueeze(1)  # [N_v, 1, H, W]
            
            # 提取特征
            features_t = self.feature_extractor(
                perception_maps_t,
                interest_map_t,
                positions_t
            )  # [N_v, 256]
            
            features_history.append(features_t)
            
            # 构建当前时刻的通信边
            edge_index_t = self.edge_builder.build_communication_edges(
                positions_t,
                comm_range=self.config.COMM_RANGE
            )
            edge_indices_history.append(edge_index_t)
        
        # 堆叠为时序特征
        features_history = torch.stack(features_history, dim=1)  # [N_v, T, 256]
        
        # ====== 步骤2：ST-GAT时空建模 ======
        vehicle_embeddings = self.stgat(
            features_history,
            edge_indices_history
        )  # [N_v, 512]
        
        # ====== 步骤3：生成Q值map ======
        q_maps = self.q_generator(vehicle_embeddings)  # [N_v, H, W]
        
        return q_maps
    
    def compute_whittle_index(
        self,
        q_maps: torch.Tensor,
        confidence_current: torch.Tensor,
        confidence_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Whittle Index
        
        Args:
            q_maps: [N_v, H, W] Q值map
            confidence_current: [N_v, H, W] 当前时刻confidence
            confidence_prev: [N_v, H, W] 上一时刻confidence
        
        Returns:
            whittle_indices: [N_v, H, W] Whittle Index map
        """
        beta = self.config.DISCOUNT_FACTOR
        
        # C_t = (r_t - β*r_{t-1}) * Q_t
        confidence_diff = confidence_current - beta * confidence_prev
        whittle_indices = confidence_diff * q_maps
        
        return whittle_indices
    
    def select_top_k_regions(
        self,
        whittle_indices: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        为每辆车选择top-K区域
        
        Args:
            whittle_indices: [N_v, H, W]
            k: int top-K数量
        
        Returns:
            actions: [N_v, H, W] 二值mask，1表示选择该区域
        """
        N_v, H, W = whittle_indices.shape
        
        # 展平
        whittle_flat = whittle_indices.view(N_v, -1)  # [N_v, H*W]
        
        # 找到top-K
        _, top_k_indices = torch.topk(whittle_flat, k, dim=1)  # [N_v, k]
        
        # 构建二值mask
        actions = torch.zeros_like(whittle_flat)  # [N_v, H*W]
        actions.scatter_(1, top_k_indices, 1)
        
        # 恢复形状
        actions = actions.view(N_v, H, W)
        
        return actions
    
    def load_pretrained_vae(self, checkpoint_path: str):
        """加载预训练的VAE"""
        self.feature_extractor.load_pretrained_vae(checkpoint_path)
