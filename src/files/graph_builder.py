"""
图数据结构：定义车辆-区域异构图
"""
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional


class CooperativePerceptionGraph:
    """
    车联网协同感知的异构图结构
    包含：车辆节点、区域节点、以及它们之间的边
    """
    
    def __init__(self, config):
        self.config = config
        
    def build_graph(
        self,
        vehicle_data: Dict,
        region_data: Dict,
        semantic_map: Optional[torch.Tensor] = None
    ) -> HeteroData:
        """
        构建异构图
        
        Args:
            vehicle_data: 车辆数据字典
                - positions: [N_v, 2] 车辆位置
                - velocities: [N_v, 2] 车辆速度
                - perception_maps: [N_v, K, H, W] 历史感知地图
                - headings: [N_v] 车辆朝向
            region_data: 区域数据字典
                - confidence_maps: [N_r] 每个区域的confidence值
                - interest_counts: [N_r] 每个区域感兴趣的车辆数
                - positions: [N_r, 2] 区域中心位置
            semantic_map: [C, H, W] 可选的语义地图
            
        Returns:
            HeteroData: PyG异构图对象
        """
        data = HeteroData()
        
        # ============ 车辆节点特征 ============
        vehicle_features = self._encode_vehicle_features(vehicle_data)
        data['vehicle'].x = vehicle_features
        data['vehicle'].pos = vehicle_data['positions']
        
        # ============ 区域节点特征 ============
        region_features = self._encode_region_features(region_data, semantic_map)
        data['region'].x = region_features
        data['region'].pos = region_data['positions']
        
        # ============ 构建边 ============
        # 1. 车辆-车辆通信边 (基于通信范围)
        v2v_edges = self._build_vehicle_edges(vehicle_data['positions'])
        data['vehicle', 'communicates', 'vehicle'].edge_index = v2v_edges
        
        # 2. 区域-区域邻接边 (基于空间邻接)
        r2r_edges = self._build_region_edges(region_data['positions'])
        data['region', 'adjacent', 'region'].edge_index = r2r_edges
        
        # 3. 车辆-区域感知边 (车辆可以感知哪些区域)
        v2r_edges = self._build_vehicle_region_edges(
            vehicle_data['positions'],
            region_data['positions']
        )
        data['vehicle', 'perceives', 'region'].edge_index = v2r_edges
        
        # 4. 区域-车辆兴趣边 (哪些车辆对该区域感兴趣)
        r2v_edges = self._build_region_vehicle_edges(
            region_data['interest_counts'],
            vehicle_data['positions'],
            region_data['positions']
        )
        data['region', 'interests', 'vehicle'].edge_index = r2v_edges
        
        # ============ 附加信息 ============
        data.num_vehicles = vehicle_data['positions'].shape[0]
        data.num_regions = region_data['positions'].shape[0]
        
        return data
    
    def _encode_vehicle_features(self, vehicle_data: Dict) -> torch.Tensor:
        """
        编码车辆特征
        包括：位置、速度、历史感知地图、朝向等
        """
        features_list = []
        
        # 1. 位置特征 (归一化)
        positions = vehicle_data['positions'] / self.config.PERCEPTION_RANGE
        features_list.append(positions)
        
        # 2. 速度特征
        velocities = vehicle_data['velocities']
        features_list.append(velocities)
        
        # 3. 朝向特征 (转换为sin/cos)
        headings = vehicle_data['headings']
        heading_features = torch.stack([
            torch.cos(headings),
            torch.sin(headings)
        ], dim=-1)
        features_list.append(heading_features)
        
        # 4. 历史感知地图特征 (通过CNN编码)
        perception_maps = vehicle_data['perception_maps']  # [N_v, K, H, W]
        encoded_maps = self._encode_perception_history(perception_maps)
        features_list.append(encoded_maps)
        
        # 合并所有特征
        vehicle_features = torch.cat(features_list, dim=-1)
        
        return vehicle_features
    
    def _encode_region_features(
        self,
        region_data: Dict,
        semantic_map: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        编码区域特征
        包括：confidence值、感兴趣车辆数、语义信息等
        """
        features_list = []
        
        # 1. Confidence值
        confidence = region_data['confidence_maps'].unsqueeze(-1)
        features_list.append(confidence)
        
        # 2. 感兴趣的车辆数 (归一化)
        interest_counts = region_data['interest_counts'].unsqueeze(-1)
        interest_counts = interest_counts / self.config.NUM_VEHICLES
        features_list.append(interest_counts)
        
        # 3. 区域位置 (归一化)
        positions = region_data['positions'] / self.config.PERCEPTION_RANGE
        features_list.append(positions)
        
        # 4. 语义特征 (如果提供)
        if semantic_map is not None:
            semantic_features = self._extract_region_semantics(
                semantic_map,
                region_data['positions']
            )
            features_list.append(semantic_features)
        
        region_features = torch.cat(features_list, dim=-1)
        
        return region_features
    
    def _encode_perception_history(self, perception_maps: torch.Tensor) -> torch.Tensor:
        """
        使用简单的CNN编码历史感知地图
        
        Args:
            perception_maps: [N_v, K, H, W]
        Returns:
            encoded: [N_v, feature_dim]
        """
        N_v, K, H, W = perception_maps.shape
        
        # 简单的平均池化 + 展平
        # 在实际应用中，可以使用更复杂的CNN
        pooled = torch.mean(perception_maps, dim=1)  # [N_v, H, W]
        encoded = torch.flatten(pooled, start_dim=1)  # [N_v, H*W]
        
        return encoded
    
    def _extract_region_semantics(
        self,
        semantic_map: torch.Tensor,
        region_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        从语义地图中提取每个区域的语义特征
        
        Args:
            semantic_map: [C, H, W]
            region_positions: [N_r, 2]
        Returns:
            semantic_features: [N_r, C]
        """
        # 简化版本：使用最近邻采样
        # 实际应用中可以使用更复杂的采样策略
        C, H, W = semantic_map.shape
        N_r = region_positions.shape[0]
        
        # 将位置映射到像素坐标
        pixel_coords = (region_positions / self.config.PERCEPTION_RANGE * torch.tensor([W, H])).long()
        pixel_coords = torch.clamp(pixel_coords, 0, torch.tensor([W-1, H-1]))
        
        # 提取特征
        semantic_features = semantic_map[:, pixel_coords[:, 1], pixel_coords[:, 0]].T
        
        return semantic_features
    
    def _build_vehicle_edges(self, positions: torch.Tensor) -> torch.Tensor:
        """
        构建车辆间通信边 (基于距离阈值)
        
        Args:
            positions: [N_v, 2]
        Returns:
            edge_index: [2, E]
        """
        N_v = positions.shape[0]
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(positions, positions)
        
        # 通信范围阈值
        comm_range = self.config.PERCEPTION_RANGE * 1.5
        
        # 找到在通信范围内的边
        adj_matrix = (dist_matrix < comm_range) & (dist_matrix > 0)
        edge_index = adj_matrix.nonzero().t()
        
        return edge_index
    
    def _build_region_edges(self, positions: torch.Tensor) -> torch.Tensor:
        """
        构建区域间邻接边 (网格邻接关系)
        
        Args:
            positions: [N_r, 2]
        Returns:
            edge_index: [2, E]
        """
        grid_size = self.config.GRID_SIZE
        N_r = grid_size * grid_size
        
        edge_list = []
        
        # 构建8邻接关系
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                
                # 上下左右邻居
                neighbors = []
                if i > 0:
                    neighbors.append((i-1) * grid_size + j)
                if i < grid_size - 1:
                    neighbors.append((i+1) * grid_size + j)
                if j > 0:
                    neighbors.append(i * grid_size + (j-1))
                if j < grid_size - 1:
                    neighbors.append(i * grid_size + (j+1))
                
                # 对角邻居
                if i > 0 and j > 0:
                    neighbors.append((i-1) * grid_size + (j-1))
                if i > 0 and j < grid_size - 1:
                    neighbors.append((i-1) * grid_size + (j+1))
                if i < grid_size - 1 and j > 0:
                    neighbors.append((i+1) * grid_size + (j-1))
                if i < grid_size - 1 and j < grid_size - 1:
                    neighbors.append((i+1) * grid_size + (j+1))
                
                for neighbor in neighbors:
                    edge_list.append([node_id, neighbor])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        return edge_index
    
    def _build_vehicle_region_edges(
        self,
        vehicle_positions: torch.Tensor,
        region_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        构建车辆到区域的感知边
        
        Args:
            vehicle_positions: [N_v, 2]
            region_positions: [N_r, 2]
        Returns:
            edge_index: [2, E] 从车辆到区域
        """
        N_v = vehicle_positions.shape[0]
        N_r = region_positions.shape[0]
        
        # 计算车辆到每个区域的距离
        dist_matrix = torch.cdist(vehicle_positions, region_positions)
        
        # 车辆可以感知在感知范围内的区域
        perception_mask = dist_matrix < self.config.PERCEPTION_RANGE
        
        edge_index = perception_mask.nonzero().t()
        
        return edge_index
    
    def _build_region_vehicle_edges(
        self,
        interest_counts: torch.Tensor,
        vehicle_positions: torch.Tensor,
        region_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        构建区域到车辆的兴趣边 (反向边)
        
        Args:
            interest_counts: [N_r] 每个区域感兴趣的车辆数
            vehicle_positions: [N_v, 2]
            region_positions: [N_r, 2]
        Returns:
            edge_index: [2, E] 从区域到车辆
        """
        # 这里简化处理：与vehicle_region_edges相反
        v2r_edges = self._build_vehicle_region_edges(vehicle_positions, region_positions)
        r2v_edges = v2r_edges.flip(0)  # 反转边方向
        
        return r2v_edges


class GraphBatch:
    """批处理多个图的工具类"""
    
    @staticmethod
    def collate_graphs(graph_list: List[HeteroData]) -> HeteroData:
        """
        将多个图合并为一个batch
        PyG会自动处理这个，这里提供自定义逻辑的接口
        """
        from torch_geometric.data import Batch
        return Batch.from_data_list(graph_list)
