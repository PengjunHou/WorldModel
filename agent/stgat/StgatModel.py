"""
完整的ST-GAT Q网络模型
"""
import torch
import torch.nn as nn
from typing import Dict, List
from ..toolkits import CNNQGenerator, SpatioTemporalGAT, EdgeBuilder, ComprehensiveFeatureExtractor


class STGATQNetwork(nn.Module):
    """
    完整的基于ST-GAT的Q网络
    
    Pipeline:
    1. 特征提取（感知map + 兴趣map + 位置） → [N_v, T, feature_dim]
    2. ST-GAT（时空建模） → [N_v, hidden_dim]
    3. Q值生成器 → [N_v, H, W]
    """
    def __init__(self, env_config, model_config):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        
        # ====== 模块1：特征提取器 ======
        self.feature_extractor = ComprehensiveFeatureExtractor(env_config=env_config, model_config=model_config)
        
        # ====== 模块2：ST-GAT ======
        self.stgat = SpatioTemporalGAT(
            feature_dim=model_config.FUSED_FEATURE_DIM,
            hidden_dim=model_config.STGAT_HIDDEN_DIM,
            num_spatial_layers=model_config.STGAT_NUM_SPATIAL_LAYERS,
            num_temporal_layers=model_config.STGAT_NUM_TEMPORAL_LAYERS,
            num_heads=model_config.STGAT_NUM_HEADS,
            dropout=model_config.STGAT_DROPOUT,
            use_edge_weights=model_config.STGAT_USEWEIGHTS
        )
        
        # ====== 模块3：Q值生成器 ======
        self.q_generator = CNNQGenerator(
            vehicle_feature_dim=model_config.STGAT_HIDDEN_DIM,
            map_size=(model_config.MAP_HEIGHT, model_config.MAP_WIDTH)
        )
        
        # ====== 边构建器 ======
        self.edge_builder = EdgeBuilder()
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            batch_data: {
                local_maps: [N_v, T, H, W] (float32)
                fused_maps: [N_v, T, H, W] (float32)
                adjacency_matrix: [T, N_v, N_v] (float32)
                positions: [N_v, T, 2]   (float32)
                interest_maps: [T, H_g, W_g]  (float32)
            }
        
        Returns:
            q_maps: [N_v, H, W] 每辆车的Q值map
        """
        N_v = batch_data['local_maps'].shape[0]
        T = batch_data['local_maps'].shape[1]
        
        # ====== 步骤1：逐时间步提取特征 ======
        features_history = []
        edge_indices_history = []
        edge_weights_history = []  # 存储边权重
        
        for t in range(T):
            # 当前时刻的数据
            local_maps_t = batch_data['local_maps'][:, t]  # [N_v, H, W]
            fused_maps_t = batch_data['fused_maps'][:, t]  # [N_v, H, W]
            interest_map_t = batch_data['interest_maps'][t]  # [H, W]
            positions_t = batch_data['positions'][:, t]  # [N_v, 3]
            
            # 添加通道维度
            local_maps_t = local_maps_t.unsqueeze(1)  # [N_v, 1, H, W]
            fused_maps_t = fused_maps_t.unsqueeze(1)
            interest_map_t = interest_map_t.unsqueeze(0)    # [1, H, W]

            # 提取特征
            features_t = self.feature_extractor(
                local_maps_t,
                fused_maps_t,
                interest_map_t,
                positions_t
            )  # [N_v, 256]
            
            features_history.append(features_t)
            
            # 构建当前时刻的通信边（支持边权重）
            use_edge_weights = getattr(self.model_config, 'STGAT_USEWEIGHTS', False)
            
            if use_edge_weights:
                # 使用边权重
                weight_type = getattr(self.model_config, 'EDGE_WEIGHT_TYPE', 'linear')
                edge_index_t, edge_weights_t = self.edge_builder.build_communication_edges(
                    positions_t,
                    comm_range=self.model_config.COMM_RANGE,
                    return_weights=True,
                    weight_type=weight_type
                )
                edge_weights_history.append(edge_weights_t)
            else:
                # 不使用边权重（默认行为）
                edge_index_t = self.edge_builder.build_communication_edges(
                    positions_t,
                    comm_range=self.model_config.COMM_RANGE
                )

            edge_indices_history.append(edge_index_t)
        
        # 堆叠为时序特征
        features_history = torch.stack(features_history, dim=1)  # [N_v, T, 256]
        
        # ====== 步骤2：ST-GAT时空建模 ======
        vehicle_embeddings = self.stgat(
            features_history,
            edge_indices_history,
            edge_weights_history
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
        beta = self.model_config.DISCOUNT_FACTOR
        
        # C_t = (r_t - β*r_{t-1}) * Q_t， TODO：确认这里的r_t是local confidence还是fused confidence
        confidence_diff = confidence_current - beta * confidence_prev
        
        print(f'q_maps: {q_maps.shape}, confidence_current: {confidence_current.shape}, \
               confidence_prev: {confidence_prev.shape},  confidence_diff: {confidence_diff.shape}')

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
    
    def load_pretrained_vae(self, checkpoint_path: str, checkpoint_n_path: str):
        """加载预训练的VAE"""
        self.feature_extractor.load_pretrained_vae(checkpoint_path, checkpoint_n_path)
