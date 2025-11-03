'''
Finish the data processing and detection code with the following changes:
把每个时间步所有车辆的local BEV置信度图, 检测结果, 融合的置信度图，融合后的检测结果，每个区域的车辆数量记录下来，保存成npz文件，方便后续分析。
每个时间步信息格式:
{
    'time_step': int,
    'vehicles': [
        {
            'agent_id': str,
            "vehicle_global_position": [x, y, z],  # 车辆位置
            'bev_confidence_local': np.ndarray (H, W),  # 局部BEV置信度图
            'detections': [  # 目标检测结果
                {
                    'box': [x1, y1, x2, y2],
                    'score': float,
                    'label': str
                },
                ...
            ]
        },
        ...
    ],
    'fused_bev_confidence': np.ndarray (H_global, W_global),  # 融合后的全局BEV置信度图
    'fused_detections': [  # 融合后的目标检测结果
        {
            'box': [x1, y1, x2, y2],
            'score': float,
            'label': str
        },
        ...
    ],
    'area_vehicle_counts': np.ndarray (H_global, W_global)
}
'''
import numpy as np
import os
from pathlib import Path
import cv2
from typing import List, Dict, Any
from src.det.ImageToBEV import ImageToBEVProjectorWithGlobal
from src.det.MultiDet import MultiAgentBEVFusion
from src.det.FastCNNDet import FasterRCNNDetector
from envs.CarlaVehEnv.CarlaDataCollector import V2XSimReader

class MultiAgentBEVDataProcessor:
    """
    多智能体BEV数据处理器
    用于保存每个时间步的所有车辆检测、BEV置信度图和融合结果
    """
    
    def __init__(self, save_dir: str, local_bev_config: dict, global_bev_config: dict):
        """
        参数:
            save_dir: 保存目录
            local_bev_config: 局部BEV配置
            global_bev_config: 全局BEV配置
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.local_bev_config = local_bev_config
        self.global_bev_config = global_bev_config
        
        # 存储所有时间步的数据
        self.time_steps_data = []
        
        print(f"✓ 数据处理器初始化完成")
        print(f"  保存目录: {self.save_dir}")
        print(f"  局部BEV大小: {local_bev_config['grid_size']}")
        print(f"  全局BEV大小: {global_bev_config['grid_size']}")
    
    def add_time_step(self, 
                      time_step: int,
                      vehicles_data: List[Dict],
                      fused_bev_confidence: np.ndarray,
                      fused_detections: List[Dict],
                      area_vehicle_counts: np.ndarray):
        """
        添加一个时间步的数据
        
        参数:
            time_step: 时间步索引
            vehicles_data: 所有车辆的数据列表
            fused_bev_confidence: 融合后的全局BEV置信度图
            fused_detections: 融合后的检测结果
            area_vehicle_counts: 每个区域的车辆数量
        """
        time_step_data = {
            'time_step': time_step,
            'vehicles': vehicles_data,
            'fused_bev_confidence': fused_bev_confidence,
            'fused_detections': fused_detections,
            'area_vehicle_counts': area_vehicle_counts
        }
        
        self.time_steps_data.append(time_step_data)
        
        print(f"✓ 已添加时间步 {time_step} 的数据")
        print(f"  车辆数: {len(vehicles_data)}")
        print(f"  融合后检测数: {len(fused_detections)}")
        print(f"  区域最大车辆数: {area_vehicle_counts.max()}")
    
    def save_to_npz(self, filename: str = None):
        """
        保存所有数据到npz文件
        
        参数:
            filename: 文件名，默认为 'multi_agent_bev_data.npz'
        """
        if filename is None:
            filename = 'multi_agent_bev_data.npz'
        
        save_path = self.save_dir / filename
        
        # 准备保存的数据字典
        save_dict = {
            'local_bev_config': self.local_bev_config,
            'global_bev_config': self.global_bev_config,
            'num_time_steps': len(self.time_steps_data)
        }
        
        # 保存每个时间步的数据
        for i, time_step_data in enumerate(self.time_steps_data):
            prefix = f'time_step_{i}'
            
            # 基本信息
            save_dict[f'{prefix}_time_step'] = time_step_data['time_step']
            save_dict[f'{prefix}_num_vehicles'] = len(time_step_data['vehicles'])
            
            # 车辆数据
            for j, vehicle_data in enumerate(time_step_data['vehicles']):
                vehicle_prefix = f'{prefix}_vehicle_{j}'
                
                save_dict[f'{vehicle_prefix}_id'] = vehicle_data['agent_id']
                save_dict[f'{vehicle_prefix}_bev_confidence_local'] = vehicle_data['bev_confidence_local']
                save_dict[f'{vehicle_prefix}_pos'] = vehicle_data['vehicle_global_position']

                # 检测结果
                if len(vehicle_data['detections']) > 0:
                    # 将检测结果转换为结构化数组
                    det_boxes = np.array([d['box'] for d in vehicle_data['detections']])
                    det_scores = np.array([d['score'] for d in vehicle_data['detections']])
                    det_labels = np.array([d['label'] for d in vehicle_data['detections']], dtype='U20')
                    
                    save_dict[f'{vehicle_prefix}_detection_boxes'] = det_boxes
                    save_dict[f'{vehicle_prefix}_detection_scores'] = det_scores
                    save_dict[f'{vehicle_prefix}_detection_labels'] = det_labels
                else:
                    save_dict[f'{vehicle_prefix}_detection_boxes'] = np.array([])
                    save_dict[f'{vehicle_prefix}_detection_scores'] = np.array([])
                    save_dict[f'{vehicle_prefix}_detection_labels'] = np.array([])
            
            # 融合结果
            save_dict[f'{prefix}_fused_bev_confidence'] = time_step_data['fused_bev_confidence']
            save_dict[f'{prefix}_area_vehicle_counts'] = time_step_data['area_vehicle_counts']
            
            # 融合后的检测结果
            if len(time_step_data['fused_detections']) > 0:
                fused_boxes = np.array([d['box'] for d in time_step_data['fused_detections']])
                fused_scores = np.array([d['score'] for d in time_step_data['fused_detections']])
                fused_labels = np.array([d['label'] for d in time_step_data['fused_detections']], dtype='U20')
                
                save_dict[f'{prefix}_fused_detection_boxes'] = fused_boxes
                save_dict[f'{prefix}_fused_detection_scores'] = fused_scores
                save_dict[f'{prefix}_fused_detection_labels'] = fused_labels
            else:
                save_dict[f'{prefix}_fused_detection_boxes'] = np.array([])
                save_dict[f'{prefix}_fused_detection_scores'] = np.array([])
                save_dict[f'{prefix}_fused_detection_labels'] = np.array([])
        
        # 保存
        np.savez_compressed(save_path, **save_dict)
        
        print(f"\n{'='*80}")
        print(f"✓ 数据已保存到: {save_path}")
        print(f"  时间步数: {len(self.time_steps_data)}")
        print(f"  文件大小: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"{'='*80}")
        
        return save_path
    
    @staticmethod
    def load_from_npz(npz_path: str):
        """
        从npz文件加载数据
        
        返回:
            data: 解析后的数据字典
        """
        data = np.load(npz_path, allow_pickle=True)
        
        # 重构数据结构
        num_time_steps = int(data['num_time_steps'])
        
        time_steps_data = []
        
        for i in range(num_time_steps):
            prefix = f'time_step_{i}'
            
            time_step = int(data[f'{prefix}_time_step'])
            num_vehicles = int(data[f'{prefix}_num_vehicles'])
            
            # 重构车辆数据
            vehicles = []
            for j in range(num_vehicles):
                vehicle_prefix = f'{prefix}_vehicle_{j}'
                
                agent_id = str(data[f'{vehicle_prefix}_id'])
                vehicle_pos = data[f'{vehicle_prefix}_pos']
                local_bev = data[f'{vehicle_prefix}_bev_confidence_local']
                
                # 重构检测结果
                boxes = data[f'{vehicle_prefix}_detection_boxes']
                scores = data[f'{vehicle_prefix}_detection_scores']
                labels = data[f'{vehicle_prefix}_detection_labels']
                
                detections = []
                if len(boxes) > 0:
                    for box, score, label in zip(boxes, scores, labels):
                        detections.append({
                            'box': box.tolist(),
                            'score': float(score),
                            'label': str(label)
                        })
                
                vehicles.append({
                    'agent_id': agent_id,
                    'vehicle_global_position': vehicle_pos,
                    'bev_confidence_local': local_bev,
                    'detections': detections
                })
            
            # 重构融合结果
            fused_bev = data[f'{prefix}_fused_bev_confidence']
            area_counts = data[f'{prefix}_area_vehicle_counts']
            
            fused_boxes = data[f'{prefix}_fused_detection_boxes']
            fused_scores = data[f'{prefix}_fused_detection_scores']
            fused_labels = data[f'{prefix}_fused_detection_labels']
            
            fused_detections = []
            if len(fused_boxes) > 0:
                for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
                    fused_detections.append({
                        'box': box.tolist(),
                        'score': float(score),
                        'label': str(label)
                    })
            
            time_steps_data.append({
                'time_step': time_step,
                'vehicles': vehicles,
                'fused_bev_confidence': fused_bev,
                'fused_detections': fused_detections,
                'area_vehicle_counts': area_counts
            })
        
        result = {
            'local_bev_config': data['local_bev_config'].item(),
            'global_bev_config': data['global_bev_config'].item(),
            'time_steps_data': time_steps_data
        }
        
        print(f"✓ 已加载数据: {npz_path}")
        print(f"  时间步数: {len(time_steps_data)}")
        
        return result

def process_single_vehicle_bev(agent_id: str,
                                image_rgb,
                                detections: List[Dict],
                                camera_params: Dict,
                                ego_vehicle_params: Dict,
                                local_bev_config: Dict,
                                global_bev_config: Dict) -> Dict:
    """
    处理单个车辆的BEV数据
    
    参数:
        agent_id: 车辆ID
        image_rgb: RGB图像
        detections: 检测结果
        camera_params: 相机参数
        ego_vehicle_params: 车辆参数
        local_bev_config: 局部BEV配置
        global_bev_config: 全局BEV配置
    
    返回:
        vehicle_data: 包含局部BEV和检测结果的字典
    """
    # 创建投影器
    projector = ImageToBEVProjectorWithGlobal(
        camera_params=camera_params,
        ego_vehicle_params=ego_vehicle_params,
        global_bev_config=global_bev_config,
        local_bev_config=local_bev_config
    )
    
    # 投影到BEV
    (bev_confidence_global, bev_coverage_global, projected_positions,
     bev_confidence_local, bev_coverage_local, local_info) = projector.project_detections_to_bev(
        detections=detections,
        method='depth_estimation',
        local_bev_config=local_bev_config,
        local_center='vehicle'  # 以车辆为中心
    )

    # agents_data.append({
    #     'agent_id': agent_config['agent_id'],
    #     'camera_params': agent_config['camera_params'],
    #     'ego_params': agent_config['ego_params'],
    #     'camera_global_position': projector.camera_global_position,
    #     'image_path': agent_config['image_path'],
    #     'detections': detections,
    #     'bev_confidence_global': bev_confidence_global,
    #     'bev_coverage_global': bev_coverage_global,
    #     'bev_confidence_local': bev_confidence_local,
    #     'bev_coverage_local': bev_coverage_local,
    #     'projected_positions': projected_positions,
    #     'local_info': local_info
    # })


    # 构建车辆数据
    vehicle_data = {
        'agent_id': agent_id,
        'image_path':image_rgb,
        "camera_params":camera_params,
        'camera_global_position': projector.camera_global_position,
        'vehicle_global_position': projector.ego_vehicle_params['translation'],
        'bev_confidence_global': bev_confidence_global,
        'bev_coverage_global': bev_coverage_global,
        'bev_confidence_local': bev_confidence_local,
        'bev_coverage_local': bev_coverage_local,
        'detections': detections,
        'ego_params': ego_vehicle_params,
        'projected_positions': projected_positions,  # 用于后续融合
        'local_info': local_info
    }
    
    return vehicle_data, projector


def fuse_multi_vehicle_bev(vehicles_data: List[Dict],
                           global_bev_config: Dict,
                           local_bev_config: Dict,
                           fusion_method: str = 'max',
                           visualize = False) -> tuple:
    """
    融合多个车辆的BEV置信度图
    
    参数:
        vehicles_data: 所有车辆的数据列表
        global_bev_config: 全局BEV配置
        fusion_method: 融合方法 'max', 'average', 'weighted'
    
    返回:
        fused_bev_confidence: 融合后的BEV置信度图
        area_vehicle_counts: 每个区域的车辆覆盖数量
    """

    fusion_system = MultiAgentBEVFusion(bev_config=global_bev_config, local_bev_config=local_bev_config)


    # 融合所有车辆的BEV数据
    fused_result = fusion_system.fuse_multi_agent_bev(vehicles_data)

    # =============== 4. 可视化 ===============
    if visualize:
        fusion_system.visualize_multi_agent_bev(
            vehicles_data,
            fused_result,
            save_path='multi_agent_bev_fusion.png'
        )

    # =============== 5. 统计分析 ===============

    fusion_system.generate_comparison_stats(vehicles_data, fused_result)
    
    return fused_result['fused_confidence'], fused_result['area_vehicle_counts']


def fuse_multi_vehicle_detections(vehicles_data: List[Dict],
                                   iou_threshold: float = 0.5,
                                   score_threshold: float = 0.3) -> List[Dict]:
    """
    融合多个车辆的检测结果（去重和NMS）
    
    参数:
        vehicles_data: 所有车辆的数据列表
        iou_threshold: IOU阈值用于NMS
        score_threshold: 置信度阈值
    
    返回:
        fused_detections: 融合后的检测结果
    """
    all_detections = []
    
    # 收集所有检测
    for vehicle_data in vehicles_data:
        if 'projected_positions' in vehicle_data:
            for proj in vehicle_data['projected_positions']:
                if proj['score'] >= score_threshold:
                    all_detections.append({
                        'box': proj['grid_pos'],  # BEV网格位置
                        'world_pos': proj['world_pos'],
                        'score': proj['score'],
                        'label': proj['label'],
                        'agent_id': vehicle_data['agent_id']
                    })
    
    if len(all_detections) == 0:
        return []
    
    # 简单的去重：基于世界坐标的距离
    fused_detections = []
    used = [False] * len(all_detections)
    
    for i, det1 in enumerate(all_detections):
        if used[i]:
            continue
        
        # 找到所有与det1接近的检测
        cluster = [det1]
        used[i] = True
        
        for j, det2 in enumerate(all_detections):
            if used[j] or i == j:
                continue
            
            # 计算世界坐标距离
            pos1 = np.array(det1['world_pos'][:2])  # X, Y
            pos2 = np.array(det2['world_pos'][:2])
            dist = np.linalg.norm(pos1 - pos2)
            
            # 如果距离小于阈值（例如2米），认为是同一个物体
            if dist < 2.0 and det1['label'] == det2['label']:
                cluster.append(det2)
                used[j] = True
        
        # 对cluster取最高置信度
        best_det = max(cluster, key=lambda x: x['score'])
        fused_detections.append(best_det)
    
    return fused_detections


def load_v2x_dataset(dataset_path: str):
    """
    加载V2X数据集的占位函数
    """
    dataset = V2XSimReader(dataset_path)

    return dataset


def process_multi_agent_bev_data(dataset_path: str,
                                  save_dir: str,
                                  local_bev_config: Dict,
                                  global_bev_config: Dict,
                                  time_steps: List[int] = None,
                                  max_vehicles_per_step: int = None):
    """
    处理多智能体BEV数据的主函数
    
    参数:
        dataset_path: 数据集路径
        save_dir: 保存目录
        local_bev_config: 局部BEV配置
        global_bev_config: 全局BEV配置
        time_steps: 要处理的时间步列表（None表示全部）
        max_vehicles_per_step: 每个时间步最大车辆数（None表示全部）
    """
    # 初始化数据处理器
    processor = MultiAgentBEVDataProcessor(
        save_dir=save_dir,
        local_bev_config=local_bev_config,
        global_bev_config=global_bev_config
    )
    
    # 加载数据集
    dataset = load_v2x_dataset(dataset_path)
    
    # 示例：处理每个时间步
    if time_steps is None:
        time_steps = range(len(dataset))  # 假设dataset是可迭代的
    
    for time_step in time_steps:
        print(f"\n{'='*80}")
        print(f"处理时间步 {time_step}")
        print(f"{'='*80}")
        
        # TODO: 获取该时间步的所有车辆数据
        vehicles_at_time_step = dataset.get_vehicles_at_time(time_step)
        print(f"检测到 {len(vehicles_at_time_step)} 辆车")
        print(f"{'-'*40}")
        print(f"{vehicles_at_time_step}")
        
        vehicles_data_list = []
        
        # 处理每个车辆
        for agent_idx in vehicles_at_time_step.keys():
            # if max_vehicles_per_step and agent_idx >= max_vehicles_per_step:
            #     break

            vehicle_info = vehicles_at_time_step[agent_idx]
            
            print(f"\n处理车辆 {vehicle_info['agent_id']}")
            
            # TODO: 获取车辆的相机图像和参数
            image_path = vehicle_info['image']
            camera_params = vehicle_info['camera_params']
            ego_params = vehicle_info['ego_params']
            
            # 运行目标检测
            detector = FasterRCNNDetector(device='cuda', conf_threshold=0.2)
            image = cv2.imread(image_path)
            detections = detector.detect(image)
            
            # 处理单个车辆的BEV
            vehicle_data, projector = process_single_vehicle_bev(
                agent_id=vehicle_info['agent_id'],
                image_rgb=image_path,
                detections=detections,
                camera_params=camera_params,
                ego_vehicle_params=ego_params,
                local_bev_config=local_bev_config,
                global_bev_config=global_bev_config
            )
            
            vehicles_data_list.append(vehicle_data)
        
        # 融合所有车辆的BEV
        print(f"\n融合 {len(vehicles_data_list)} 个车辆的BEV...")
        
        fused_bev_confidence, area_vehicle_counts = fuse_multi_vehicle_bev(
            vehicles_data=vehicles_data_list,
            global_bev_config=global_bev_config,
            local_bev_config=local_bev_config,
            fusion_method='max'
        )
        
        # 融合检测结果
        fused_detections = fuse_multi_vehicle_detections(
            vehicles_data=vehicles_data_list,
            iou_threshold=0.5,
            score_threshold=0.3
        )
        
        # 添加到处理器
        processor.add_time_step(
            time_step=time_step,
            vehicles_data=vehicles_data_list,
            fused_bev_confidence=fused_bev_confidence,
            fused_detections=fused_detections,
            area_vehicle_counts=area_vehicle_counts
        )
    
    # 保存所有数据
    save_path = processor.save_to_npz('multi_agent_bev_data.npz')
    
    return processor, save_path

if __name__ == "__main__":
    # 测试代码放在这里
    data_path = "/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini"
    v2x_data_path = "/home/peh324/Codes/V2X-Sim-2.0-mini/V2X-Sim-det"
    # BEV配置
    bev_config = {
        'area_extents': [[-256, 256], [-256, 256]],  # 覆盖64m x 64m区域
        'voxel_size': [0.25, 0.25],              # 每个网格0.25m
        'grid_size': [2048, 2048]                  # 256x256网格
    }

    local_bev_config = {
        'area_extents': [[-32, 32], [-32, 32]],  
        'voxel_size': [0.25, 0.25],              
        'grid_size': [256, 256]                  
    }

    process_multi_agent_bev_data(
        dataset_path=data_path,
        save_dir='./tmp',
        local_bev_config=local_bev_config,
        global_bev_config=bev_config,
        time_steps=range(2),  # 仅处理前10个时间步},
        max_vehicles_per_step=5  # 每个时间步最多处理5辆车
    )