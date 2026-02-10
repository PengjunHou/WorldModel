from envs.CarlaVehEnv.Sensor import Lidar, Camera
from envs.CarlaVehEnv.CarlaDataCollector import V2XSimReader
from envs.CarlaVehEnv.Visualize import visualize_step
from envs.CarlaVehEnv.utils import *
from envs.CarlaVehEnv.Comm_Comp_settings import Comm_Comp_Base
from nuscenes.utils.geometry_utils import transform_matrix
from v2x_sim_visualizer import render_sample_data, render_scene_lidar
from pyquaternion import Quaternion
from functools import reduce
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from src.det.ImageToBEV import ImageToBEVProjectorWithGlobal
from src.det.process import process_single_vehicle_bev, fuse_multi_vehicle_detections, fuse_multi_vehicle_bev
from src.det.utils import plot_detections
from src.det.MultiDet import MultiAgentBEVFusion
from src.gode.gode_model import VehicleGODEPredictor
from src.extract.vae_module import PerceptionMapVAE
from src.train_odemodel import GODETrainer
# from src.det.DetectionVis_old import BEVFusionDetectionVisualizer

import logging
LOG = logging.getLogger(__name__)

class Car():
    def __init__(self, vid, dataset, config, detector, carla_vehicle = None):
        self.carla_vehicle = carla_vehicle
        self.detector = detector
        self.config = config
        self.visualize = config.visualize          # 可视化的数据
        self.dataset : V2XSimReader = dataset# V2XSimDet dataset for detection
        self.cluster_id = -1
        self.vid = vid
        self.results_path = os.path.join(self.config.result_path, f"vehicle_{self.vid:02d}")
        self.setup()
        self.strategy = config.strategy
        self.seqs_len = config.seqs_len # T = 5
        # store the sequences of data
        self.local_map_seqs = []
        self.fused_map_seqs = []
        self.position_seqs = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_slices_cnt = 0
        self.current_slices_limits = 0
        
        # define the map
        self.local_conf_map = np.zeros((self.dataset.local_bev_config[ 'grid_size'][0], \
                                        self.dataset.local_bev_config[ 'grid_size'][1]), dtype=np.float32)
        self.cur_fused_conf_map = np.zeros((self.dataset.local_bev_config[ 'grid_size'][0], \
                                        self.dataset.local_bev_config[ 'grid_size'][1]), dtype=np.float32)
        self.last_fused_conf_map = np.zeros((self.dataset.local_bev_config[ 'grid_size'][0], \
                                        self.dataset.local_bev_config[ 'grid_size'][1]), dtype=np.float32)

        self.comm_comp_model = Comm_Comp_Base(self.config)
        
        # self.detVis = BEVFusionDetectionVisualizer(self.results_path)
        # self.gode_net = GodeNet()
        vae = PerceptionMapVAE(
            map_size=(16, 16),
            latent_dim=64
        ).to(self.device)
        vae_model_path = '/home/peh324/Codes/WorldModel/src/extract/checkpoints/perception_vae.pth'
        if os.path.exists(vae_model_path):
            checkpoint = torch.load(vae_model_path, map_location=self.device)
            vae.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded VAE from {vae_model_path}")
        else:
            print(f"⚠️  VAE checkpoint not found: {vae_model_path}")
            print("    Using randomly initialized VAE (not recommended)")
            # raise "vae error"
        
        self.gode_net = VehicleGODEPredictor(
            vae_encoder=vae,
            map_size=(16, 16),
            vae_latent_dim=64,
            temporal_hidden_dim=128,
            gnn_hidden_dim=256,
            num_gnn_layers=3,
            k_embed_dim=16,
            comm_range=100
        ).to(self.device)
        self.gode_net.eval()

        gode_model_path = '/home/peh324/Codes/WorldModel/checkpoints/gode/best_model.pth'
        check_points = torch.load(gode_model_path, map_location=self.device)
        self.gode_net.load_state_dict(check_points['model_state_dict'])

        

    def setup(self):
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            
    def compare_diff(self, pred_fused_map, groudth):
        pred_fused_map

    def apply_control(self, time_step, actions, area_cnt, vehicles_info):
        """
        应用控制：融合其他车辆上传的局部BEV置信度图到自己的置信度图中
        
        由于所有车辆使用相同的局部BEV配置，坐标转换大大简化
        
        参数:
            time_step: 当前时间步
            actions: dict {vehicle_id: [(p, q, score), ...]}
                    每辆车上传的局部BEV网格索引和置信度值
            area_cnt: 区域计数
            vehicles_info: dict {vehicle_id: vehicle_data}
                        所有车辆的信息，用于获取位置
        """
        # 1. 复制上一时刻的融合地图
        self.cur_fused_conf_map[...] = self.last_fused_conf_map
        
        # 2. 将自己的局部BEV置信度图融合进来
        np.maximum(self.cur_fused_conf_map, self.local_conf_map, out=self.cur_fused_conf_map)
        
        # 3. 获取当前车辆的位置信息
        self_vehicle_info = vehicles_info[self.vid-1]
        
        # 当前车辆的全局位置
        self_vehicle_x = self_vehicle_info['vehicle_global_position'][0]  # X: 前后
        self_vehicle_y = self_vehicle_info['vehicle_global_position'][1]  # Y: 左右
        self_vehicle_z = self_vehicle_info['vehicle_global_position'][2]  # Z: 高度
        
        # 局部BEV配置（所有车辆相同）
        local_x_min, local_x_max = self.dataset.local_bev_config['area_extents'][0]  # [-32, 32]
        local_y_min, local_y_max = self.dataset.local_bev_config['area_extents'][1]  # [-32, 32]
        voxel_size_x, voxel_size_y = self.dataset.local_bev_config['voxel_size']     # [4, 4]
        grid_h, grid_w = self.dataset.local_bev_config['grid_size']                  # [16, 16]
        
        # print(f"\n[Vehicle {self.vid}] 时间步 {time_step} - 融合其他车辆的BEV数据")
        # print(f"  自车位置: X={self_vehicle_x:.2f}, Y={self_vehicle_y:.2f}, Z={self_vehicle_z:.2f}")
        
        # 4. 融合其他车辆上传的数据
        total_points_received = 0
        total_points_used = 0
        
        for vid, action in actions.items():
            if vid == self.vid:
                continue  # 跳过自己
            
            # 获取其他车辆的位置信息
            other_vehicle_info = vehicles_info[vid-1]
            
            other_vehicle_x = other_vehicle_info['vehicle_global_position'][0]
            other_vehicle_y = other_vehicle_info['vehicle_global_position'][1]  # ⭐ 直接用Y
            other_vehicle_z = other_vehicle_info['vehicle_global_position'][2]
            
            # print(f"\n  融合来自车辆 {vid} 的数据:")
            # print(f"    对方位置: X={other_vehicle_x:.2f}, Y={other_vehicle_y:.2f}, Z={other_vehicle_z:.2f}")
            # print(f"    接收到 {len(action)} 个数据点")
            
            # ⭐ 计算两车之间的相对位置（在BEV坐标系下）
            # BEV网格定义：
            # - grid的第0维(行): 对应Y方向（左右）
            # - grid的第1维(列): 对应X方向（前后）
            # 
            # 因此：
            # - delta_grid_x (列偏移) = ΔX / voxel_size_x  （前后方向）
            # - delta_grid_y (行偏移) = ΔY / voxel_size_y  （左右方向）
            
            delta_x = other_vehicle_x - self_vehicle_x
            delta_y = other_vehicle_y - self_vehicle_y  
            delta_z = other_vehicle_z - self_vehicle_z  
            
            # print(f"    相对位置: ΔX={delta_x},  ΔY={delta_y:.2f}m, ΔZ={delta_z:.2f}m")
            
            # 转换为网格索引差异
            delta_grid_x = int(delta_x / voxel_size_x)  # 前后方向的网格差异
            delta_grid_y = int(delta_y / voxel_size_y)  # 左右方向的网格差异
            
            # print(f"    网格偏移: Δgrid_x={delta_grid_x}, Δgrid_y={delta_grid_y}")
            
            points_used = 0
            
            for p, q, score in action:
                total_points_received += 1
                
                # ⭐ 坐标转换
                # 由于配置相同，只需要加上网格偏移量
                self_p = p + delta_grid_x
                self_q = q + delta_grid_y
                
                # 判断是否在当前车辆的局部BEV范围内
                if 0 <= self_p < grid_w and 0 <= self_q < grid_h:
                    # 在范围内，更新融合地图
                    if self.cur_fused_conf_map[self_q, self_p] < score:
                        self.cur_fused_conf_map[self_q, self_p] = score
                        points_used += 1
            
            total_points_used += points_used
            usage_rate = points_used / max(len(action), 1) * 100
            # print(f"    使用了 {points_used}/{len(action)} 个数据点 ({usage_rate:.1f}%)")
        
        # print(f"\n  总计: 接收 {total_points_received} 个点, 使用 {total_points_used} 个点 ({total_points_used/max(total_points_received,1)*100:.1f}%)")
        
        # 5. 应用延迟衰减
        map_size = self.local_conf_map.shape[0] * self.local_conf_map.shape[1]
        times_delay_cop = (self.comm_comp_model.get_time_up0() + 
                        self.comm_comp_model.get_time_up1(map_size, area_cnt) + 
                        self.comm_comp_model.get_time_down() + 
                        self.comm_comp_model.get_time_proc(map_size, area_cnt))
        
        decay = np.exp(-times_delay_cop)
        
        print(f"  通信延迟: {times_delay_cop:.4f}s, 衰减系数: {decay:.4f}")
        
        # 保存当前融合地图
        self.last_fused_conf_map = self.cur_fused_conf_map.copy()
        
        # 应用衰减到当前融合地图
        # self.cur_fused_conf_map *= decay
        delay = np.exp(-self.dataset.frequency)
        self.local_conf_map_for_fuion = self.local_conf_map * delay
        
        # 应用时间衰减到上一时刻的融合地图
        max_latency = max(self.dataset.frequency, times_delay_cop)
        delay = np.exp(-max_latency)
        print(f"  时间衰减系数: {delay:.4f}")
        self.last_fused_conf_map *= delay
        self.last_fused_conf_map = np.maximum(self.last_fused_conf_map, self.local_conf_map_for_fuion)
        
        # 保存融合地图序列
        self.fused_map_seqs.append(self.last_fused_conf_map.copy())
        
        # print(f"  融合后地图统计: 最大值={self.cur_fused_conf_map.max():.4f}, 非零元素={np.count_nonzero(self.cur_fused_conf_map)}")

    def apply_control3(self, time_step, actions, area_cnt, vehicles_info):
        """
        应用控制：融合其他车辆上传的局部BEV置信度图到自己的置信度图中
        
        由于所有车辆使用相同的局部BEV配置，坐标转换大大简化
        
        参数:
            time_step: 当前时间步
            actions: dict {vehicle_id: [(p, q, score), ...]}
                    每辆车上传的局部BEV网格索引和置信度值
            area_cnt: 区域计数
            vehicles_info: dict {vehicle_id: vehicle_data}
                        所有车辆的信息，用于获取位置
        """
        # 1. 复制上一时刻的融合地图
        self.cur_fused_conf_map[...] = self.last_fused_conf_map
        
        # 2. 将自己的局部BEV置信度图融合进来
        np.maximum(self.cur_fused_conf_map, self.local_conf_map, out=self.cur_fused_conf_map)
        
        # 3. 获取当前车辆的位置信息
        self_vehicle_info = vehicles_info[self.vid]
        
        # 当前车辆的位置（原始坐标）
        self_vehicle_y_orig = self_vehicle_info['vehicle_global_position'][1]
        self_vehicle_z = self_vehicle_info['vehicle_global_position'][2]
        
        # 当前车辆的BEV坐标（修改后的Y坐标）
        self_vehicle_y_bev = self_vehicle_z + self_vehicle_y_orig
        
        # 局部BEV配置（所有车辆相同）
        local_x_min, local_x_max = self.local_bev_config['area_extents'][0]  # [-32, 32]
        local_y_min, local_y_max = self.local_bev_config['area_extents'][1]  # [-32, 32]
        voxel_size_x, voxel_size_y = self.local_bev_config['voxel_size']     # [4, 4]
        grid_h, grid_w = self.local_bev_config['grid_size']                  # [16, 16]
        
        # print(f"\n[Vehicle {self.vid}] 时间步 {time_step} - 融合其他车辆的BEV数据")
        # print(f"  自车位置: Y_bev={self_vehicle_y_bev:.2f}, Z={self_vehicle_z:.2f}")
        
        # 4. 融合其他车辆上传的数据
        total_points_received = 0
        total_points_used = 0
        
        for vid, action in actions.items():
            if vid == self.vid:
                continue  # 跳过自己
            
            if vid not in vehicles_info:
                print(f"  ⚠️ 警告: 车辆 {vid} 的信息不存在，跳过")
                continue
            
            # 获取其他车辆的位置信息
            other_vehicle_info = vehicles_info[vid]
            
            other_vehicle_y_orig = other_vehicle_info['vehicle_global_position'][1]
            other_vehicle_z = other_vehicle_info['vehicle_global_position'][2]
            
            # 其他车辆的BEV坐标
            other_vehicle_y_bev = other_vehicle_z + other_vehicle_y_orig
            
            print(f"\n  融合来自车辆 {vid} 的数据:")
            print(f"    对方位置: Y_bev={other_vehicle_y_bev:.2f}, Z={other_vehicle_z:.2f}")
            print(f"    接收到 {len(action)} 个数据点")
            
            # 计算两车之间的相对位置（在BEV坐标系下）
            # 这是关键：由于配置相同，只需要计算位置差
            delta_y_bev = other_vehicle_y_bev - self_vehicle_y_bev  # 前后方向差异
            delta_z = other_vehicle_z - self_vehicle_z              # 左右方向差异
            
            print(f"    相对位置: ΔY_bev={delta_y_bev:.2f}m, ΔZ={delta_z:.2f}m")
            
            # 转换为网格索引差异
            delta_grid_x = int(delta_y_bev / voxel_size_x)  # 前后方向的网格差异
            delta_grid_y = int(delta_z / voxel_size_y)      # 左右方向的网格差异
            
            print(f"    网格偏移: Δgrid_x={delta_grid_x}, Δgrid_y={delta_grid_y}")
            
            points_used = 0
            
            for p, q, score in action:
                total_points_received += 1
                
                # ⭐ 坐标转换（超级简单！）
                # 由于配置相同，只需要加上网格偏移量
                self_p = p + delta_grid_x
                self_q = q + delta_grid_y
                
                # 判断是否在当前车辆的局部BEV范围内
                if 0 <= self_p < grid_w and 0 <= self_q < grid_h:
                    # 在范围内，更新融合地图
                    if self.cur_fused_conf_map[self_q, self_p] < score:
                        self.cur_fused_conf_map[self_q, self_p] = score
                        points_used += 1
            
            total_points_used += points_used
            usage_rate = points_used / max(len(action), 1) * 100
            print(f"    使用了 {points_used}/{len(action)} 个数据点 ({usage_rate:.1f}%)")
        
        print(f"\n  总计: 接收 {total_points_received} 个点, 使用 {total_points_used} 个点 ({total_points_used/max(total_points_received,1)*100:.1f}%)")
        
        # 5. 应用延迟衰减
        map_size = self.local_conf_map.shape[0] * self.local_conf_map.shape[1]
        times_delay_cop = (self.comm_comp_model.get_time_up0() + 
                        self.comm_comp_model.get_time_up1(map_size, area_cnt) + 
                        self.comm_comp_model.get_time_down() + 
                        self.comm_comp_model.get_time_proc(map_size, area_cnt))
        
        decay = np.exp(-times_delay_cop)
        
        print(f"  通信延迟: {times_delay_cop:.4f}s, 衰减系数: {decay:.4f}")
        
        # 保存当前融合地图
        self.last_fused_conf_map = self.cur_fused_conf_map.copy()
        
        # 应用衰减到当前融合地图
        self.cur_fused_conf_map *= decay
        
        # 应用时间衰减到上一时刻的融合地图
        delay = np.exp(-self.dataset.frequency)
        self.last_fused_conf_map *= delay
        
        # 保存融合地图序列
        self.fused_map_seqs.append(self.last_fused_conf_map.copy())
        
        print(f"  融合后地图统计: 最大值={self.cur_fused_conf_map.max():.4f}, 非零元素={np.count_nonzero(self.cur_fused_conf_map)}")


    def apply_control2(self, time_step, actions, area_cnt):
        self.cur_fused_conf_map[...] = self.last_fused_conf_map  # 原地覆盖
        np.maximum(self.cur_fused_conf_map, self.local_conf_map, out=self.cur_fused_conf_map) # ToDO：应该不能直接覆盖，因为是local
        
        for vid, action in actions.items():
            if vid != self.vid:
                for index_x, index_y, score in action:
                    # assert self.local_conf_map[index_x, index_y] <= score, f"Score {score} is less than current value {self.local_conf_map[index_x, index_y]} at ({index_x}, {index_y}) at time step {time_step}"
                    # fuse to fused map
                    if self.cur_fused_conf_map[index_x, index_y] < score:
                        self.cur_fused_conf_map[index_x, index_y] = score
        # apply delay
        map_size = self.local_conf_map.shape[0] * self.local_conf_map.shape[1]
        times_delay_cop = self.comm_comp_model.get_time_up0() + self.comm_comp_model.get_time_up1(map_size, area_cnt) + self.comm_comp_model.get_time_down() + self.comm_comp_model.get_time_proc(map_size, area_cnt)
        decay = np.exp(-times_delay_cop)
        self.last_fused_conf_map = self.cur_fused_conf_map.copy()
        self.cur_fused_conf_map *= decay
        delay = np.exp( - self.dataset.frequency)
        self.last_fused_conf_map *= delay
        self.fused_map_seqs.append(self.last_fused_conf_map.copy())

    @torch.no_grad()
    def get_state(self, vehicle_info, time_step):
        self.local_conf_map = np.zeros((self.dataset.local_bev_config[ 'grid_size'][0], \
                                        self.dataset.local_bev_config[ 'grid_size'][1]), dtype=np.float32)
        obs = {'vid': self.vid, 'time_step': time_step}

        '''
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
        '''
        vehicle_info, projector = self.run_detection(vehicle_info, time_step, visualize = self.visualize)  # 使用点云数量检测

        self.local_conf_map = vehicle_info['bev_confidence_local']
        self.position_seqs.append(vehicle_info['vehicle_global_position'][:2])
        self.local_map_seqs.append(self.local_conf_map.copy())
        if len(self.fused_map_seqs) == 0:
            self.last_fused_conf_map = self.local_conf_map
            self.fused_map_seqs.append(self.last_fused_conf_map.copy())
        
        obs.update({'position': vehicle_info['vehicle_global_position'][:2],
                      'local_maps': self.local_conf_map,
                      'fused_maps': self.last_fused_conf_map,
                      'cur_fused_maps': self.cur_fused_conf_map,})

        LOG.info(f"Car {self.vid} at time step {time_step} has local conf_map: {self.local_conf_map.shape},  nonzero {np.count_nonzero(self.local_conf_map)}, min/max: {float(self.local_conf_map.min()), float(self.local_conf_map.max())}")
        # LOG.info(f"Car {self.vid} at time step {time_step} has state: {state}") 

        return obs, vehicle_info, projector

    
    def join_group(self, cluster_id):
        # Join a group of vehicles
        self.cluster_id = cluster_id

    def leave_group(self):
        # Leave the current group of vehicles
        self.cluster_id = None

    @torch.no_grad()
    def run_detection(self, vehicle_info, time_step, visualize = 0): 
        image_path = vehicle_info['image']
        camera_params = vehicle_info['camera_params']
        ego_params = vehicle_info['ego_params']

        detector = self.detector(device='cuda', conf_threshold=self.config.det_threshold)
        import cv2
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect(image)
        
        # 处理单个车辆的BEV
        '''
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
        '''
        vehicle_data, projector = process_single_vehicle_bev(
            agent_id=vehicle_info['agent_id'],
            image_rgb=image_path,
            detections=detections,
            camera_params=camera_params,
            ego_vehicle_params=ego_params,
            local_bev_config=self.dataset.local_bev_config,
            global_bev_config=self.dataset.global_bev_config
        )

        if visualize:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            # 原始图像
            axes[0][0].set_title('Image Detections')
            axes[0][0].imshow(plot_detections(image_rgb, detections))
            axes[0][0].set_xlim(0, 1600)
            axes[0][0].set_ylim(900, 0)
            
            # BEV置信度地图
            bev_confidence_global, bev_coverage_global, projected_positions,\
                    bev_confidence_local, bev_coverage_local, local_info = projector.project_detections_to_bev(
                        detections,
                        method='depth_estimation',
                        local_bev_config=self.dataset.local_bev_config,
                        local_center='vehicle'  # 或 'camera'
                )
            im1 = axes[0][1].imshow(bev_confidence_global, cmap='hot', origin='lower', 
                        extent=[self.dataset.global_bev_config['area_extents'][0][0], 
                                self.dataset.global_bev_config['area_extents'][0][1],
                                self.dataset.global_bev_config['area_extents'][1][0], 
                                self.dataset.global_bev_config['area_extents'][1][1]])
            axes[0][1].set_title('Global BEV Confidence Map')
            axes[0][1].set_xlabel('X (meters)')
            axes[0][1].set_ylabel('Y (meters)')
            plt.colorbar(im1, ax=axes[0][1])
            # 标记相机位置
            # cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1] # 车辆坐标系中位置
            cam_x, cam_y = projector.camera_global_position[0], projector.camera_global_position[1]  # 全局坐标系中位置
            axes[0][1].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
            axes[0][1].legend()

            # Local BEV覆盖地图
            im2 = axes[1][0].imshow(bev_confidence_local, cmap='hot', origin='lower', 
                        extent=[self.dataset.local_bev_config['area_extents'][0][0], 
                                self.dataset.local_bev_config['area_extents'][0][1],
                                self.dataset.local_bev_config['area_extents'][1][0], 
                                self.dataset.local_bev_config['area_extents'][1][1]])
            axes[1][0].set_title('Local BEV Confidence Map')
            axes[1][0].set_xlabel('X (meters)')
            axes[1][0].set_ylabel('Y (meters)')
            plt.colorbar(im2, ax=axes[1][0])

            # 标记相机位置
            cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1] # 车辆坐标系中位置
            axes[1][0].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
            axes[1][0].legend()

            # BEV置信度地图
            im1 = axes[1][1].imshow(self.cur_fused_conf_map, cmap='hot', origin='lower', 
                        extent=[self.dataset.local_bev_config['area_extents'][0][0], 
                                self.dataset.local_bev_config['area_extents'][0][1],
                                self.dataset.local_bev_config['area_extents'][1][0], 
                                self.dataset.local_bev_config['area_extents'][1][1]])
            axes[1][1].set_title('Global BEV Confidence Map')
            axes[1][1].set_xlabel('X (meters)')
            axes[1][1].set_ylabel('Y (meters)')
            plt.colorbar(im1, ax=axes[1][1])
            # 标记相机位置
            cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1] # 车辆坐标系中位置
            # cam_x, cam_y = projector.camera_global_position[0], projector.camera_global_position[1]  # 全局坐标系中位置
            axes[1][1].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
            axes[1][1].legend()

            plt.tight_layout()
            figfile = os.path.join(self.results_path, f"detection_bev_t{time_step:03d}.png")
            plt.savefig(figfile, dpi=150)
            plt.show()

        return vehicle_data, projector

    
class CarLeader(Car):
    def __init__(self, carla_vehicle):
        super().__init__(carla_vehicle)
        self.cluster_id = None
        self.vid = None

    def broadcast_data(self):
        # Broadcast the state of the leader vehicle to its members
        for member in self.members:
            member.receive_state(self.data_fusion)

    def receive2fusemap(self, data):
        pass

    def data_fusion(self):
        # Perform data fusion with the leader's state
        fusion_result = {}
        return fusion_result
    
    def control(self, action):
        # Control the leader vehicle based on the action
        pass

class Clusters():
    def __init__(self, config, dataset, cluster_id, leader_id, vehicles, pre_IoU_step, pre_IoU2K_star, pre_Iou):
        self.config = config
        self.dataset : V2XSimReader = dataset
        self.cluster_id = cluster_id
        self.members : dict[Car] = vehicles # vid -> Car
        assert leader_id in range(len(vehicles)), "Leader ID is not in the cluster"
        self.leader = vehicles[leader_id]
        self.comm_comp_model = Comm_Comp_Base(self.config)
        self.IoU = 0
        self.K_star_value = 0
        self.pre_IoU_step = pre_IoU_step
        self.pre_IoU2K_star = pre_IoU2K_star
        self.pre_Iou = pre_Iou
        self.seqs_len = config.seqs_len
        self.interest_map_seqs = []
        self.adjacency_seqs = []
        self.vehicles_data_list = []

        LOG.info(f"pre_IoU_step : {pre_IoU_step}")
        LOG.info(f"pre_IoU2K_star : {pre_IoU2K_star}")
        LOG.info(f"pre_Iou : {pre_Iou}")

        LOG.debug(f"members: f{self.members}")
    
    def get_members(self):
        return self.members
    
    def set_leader(self, leader):
        self.leader = leader
        leader.join_group(self.cluster_id)

    def get_leader(self):
        return self.leader
    
    def compute_IoU(self, gamma):
        lam = (2.0 * gamma) / (1.0 + gamma)
        lam = max(0.0, min(1.0, lam))
        return lam
    
    def overlap(self, agent_obs1, agent_obs2):
        # 判断两个Agent的local confidence map是否有重叠
        local_map1 = agent_obs1.get("local_map")
        local_map2 = agent_obs2.get("local_map")

        if local_map1 is None or local_map2 is None:
            return False, 0.0

        overlap_area = np.logical_and(local_map1 > 0, local_map2 > 0)
        a = np.sum(local_map1 > 0 )
        b = np.sum(local_map2 > 0 )
        union = a + b - np.sum(overlap_area)
        overlap_degree = np.sum(overlap_area) / union if union > 0 else 0.0

        return np.any(overlap_area), overlap_degree

    def get_state(self, time_step):
        """
        获取cluster状态并使用GODE进行通信决策
        """
        # Return the state of the cluster
        vehicles_at_time_step = self.dataset.get_vehicles_at_time(time_step)
        print(f"检测到 {len(vehicles_at_time_step)} 辆车")
        print(f"{'-'*40}")
        self.vehicles_data_list = []

        cluster_state = {}
        local_maps = []
        fused_maps = []
        cur_fused_maps = []
        positions = []
        adjacency_matrix = np.zeros((len(self.members), len(self.members)), dtype=np.float32)
        
        interest_deduct = []
        threshold = 0.8  # GODE预测得分阈值
        comm_period = getattr(self, 'comm_period', 5)  # 通信周期,默认5步
        
        # 收集所有车辆的历史数据
        all_vehicles_history_maps = []      # [N_vehicles, T, H, W]
        all_vehicles_history_positions = [] # [N_vehicles, T, 2]
        all_vehicles_current_fused = []     # [N_vehicles, H, W]
        vehicle_ids = list(self.members.keys())
        
        # ===== 第一步: 收集每辆车的状态和历史数据 =====
        for vid, member in self.members.items():
            vehicle_info = vehicles_at_time_step[vid]
            print(f"\n处理车辆 {vehicle_info['agent_id']}")
            obs, vehicle_data, projector = member.get_state(vehicle_info, time_step)
            self.vehicles_data_list.append(vehicle_data)
            
            local_maps.append(obs['local_maps'])
            fused_maps.append(obs['fused_maps'])
            cur_fused_maps.append(obs['cur_fused_maps'])
            positions.append(obs['position'])
            
            # ===== 从member的序列缓存中获取历史 =====
            # member.local_map_seqs: list of [H, W], 长度为T
            # member.position_seqs: list of [2], 长度为T
            
            if len(member.local_map_seqs) > 0 and len(member.position_seqs) > 0:
                # 转换为numpy数组
                local_map_history = np.array(member.local_map_seqs)      # [T, H, W]
                position_history = np.array(member.position_seqs)        # [T, 2]
                
                all_vehicles_history_maps.append(local_map_history)
                all_vehicles_history_positions.append(position_history)
                
                print(f"  历史序列长度: maps={len(member.local_map_seqs)}, positions={len(member.position_seqs)}")
            else:
                # 如果历史为空(刚开始),用当前数据填充
                print(f"  ⚠️  车辆{vid}历史为空,用当前数据填充")
                T = member.seqs_len
                local_map_history = np.tile(obs['local_maps'], (T, 1, 1))  # [T, H, W]
                position_history = np.tile(obs['position'], (T, 1))         # [T, 2]
                
                all_vehicles_history_maps.append(local_map_history)
                all_vehicles_history_positions.append(position_history)
            
            all_vehicles_current_fused.append(obs['cur_fused_maps'])
        
        # 添加维度检查
        print(f"\n{'='*60}")
        print(f"历史数据维度检查:")
        for i, vid in enumerate(vehicle_ids):
            print(f"  车辆{vid}: maps={all_vehicles_history_maps[i].shape}, "
                f"positions={all_vehicles_history_positions[i].shape}")
        print(f"{'='*60}")
        
        # ===== 第二步: GODE预测与评估 =====
        for i, (vid, member) in enumerate(self.members.items()):
            if member.gode_net is not None:
                try:
                    # ===== 确保模型在正确设备 =====
                    device = next(member.gode_net.parameters()).device
                    member.gode_net = member.gode_net.to(device)
                    
                    # 准备当前车辆的输入
                    self_history_maps = all_vehicles_history_maps[i]          # [T, H, W]
                    self_history_positions = all_vehicles_history_positions[i] # [T, 2]
                    
                    # 验证维度
                    assert len(self_history_maps.shape) == 3, \
                        f"self_history_maps维度错误: {self_history_maps.shape}, 期望 [T, H, W]"
                    assert len(self_history_positions.shape) == 2, \
                        f"self_history_positions维度错误: {self_history_positions.shape}, 期望 [T, 2]"
                    
                    T, H, W = self_history_maps.shape
                    
                    # ===== 构建其他车辆的数据 (考虑通信延迟) =====
                    other_indices = [j for j in range(len(vehicle_ids)) if j != i]
                    N_others = len(other_indices)
                    
                    others_maps = np.zeros((N_others, T, H, W), dtype=np.float32)
                    others_positions = np.zeros((N_others, T, 2), dtype=np.float32)
                    
                    # 简化通信延迟模型
                    for idx_other, j in enumerate(other_indices):
                        for t_idx in range(T):
                            # 计算当前历史时刻的绝对时间
                            current_absolute_time = time_step - T + t_idx + 1
                            
                            # 上次通信时间 (向下取整到comm_period的倍数)
                            last_comm_time = (current_absolute_time // comm_period) * comm_period
                            
                            # 数据延迟
                            delay = current_absolute_time - last_comm_time
                            
                            # 使用延迟后的地图索引
                            delayed_t_idx = max(0, t_idx - delay)
                            
                            # Local map: 使用延迟的数据
                            others_maps[idx_other, t_idx] = all_vehicles_history_maps[j][delayed_t_idx]
                            
                            # Position: 实时 (GPS高频广播)
                            others_positions[idx_other, t_idx] = all_vehicles_history_positions[j][t_idx]
                    
                    # 当前K值
                    K_value = self.config.TOP_K
                    
                    # 转换为torch张量并添加batch维度
                    self_history_maps_t = torch.FloatTensor(self_history_maps).unsqueeze(0).to(device)  # [1, T, H, W]
                    self_history_positions_t = torch.FloatTensor(self_history_positions).unsqueeze(0).to(device)  # [1, T, 2]
                    others_maps_t = torch.FloatTensor(others_maps).unsqueeze(0).to(device)  # [1, N-1, T, H, W]
                    others_positions_t = torch.FloatTensor(others_positions).unsqueeze(0).to(device)  # [1, N-1, T, 2]
                    K_value_t = torch.FloatTensor([[K_value]]).to(device)  # [1, 1]
                    
                    print(f"\n  车辆{vid} GODE输入:")
                    print(f"    self_history_maps_t: {self_history_maps_t.shape}")
                    print(f"    self_history_positions_t: {self_history_positions_t.shape}")
                    print(f"    others_maps_t: {others_maps_t.shape}")
                    print(f"    others_positions_t: {others_positions_t.shape}")
                    print(f"    K_value_t: {K_value_t.shape}")
                    
                    # GODE预测
                    with torch.no_grad():
                        # 确保VAE encoder也在正确设备
                        if hasattr(member.gode_net, 'vae_encoder'):
                            member.gode_net.vae_encoder = member.gode_net.vae_encoder.to(device)
                        
                        predicted_fused = member.gode_net(
                            self_history_maps_t,
                            self_history_positions_t,
                            others_maps_t,
                            others_positions_t,
                            K_value_t
                        )  # [1, H, W]
                    
                    # 转回numpy
                    predicted_fused = predicted_fused.squeeze(0).cpu().numpy()  # [H, W]
                    
                    # 实际的fused map
                    actual_fused = all_vehicles_current_fused[i]  # [H, W]
                    
                    # 计算相似度得分
                    pred_flat = predicted_fused.flatten()
                    actual_flat = actual_fused.flatten()
                    
                    # 方法1: 相关系数
                    if pred_flat.std() > 1e-6 and actual_flat.std() > 1e-6:
                        score = np.corrcoef(pred_flat, actual_flat)[0, 1]
                    else:
                        # Fallback: MSE转换
                        mse = np.mean((predicted_fused - actual_fused) ** 2)
                        score = 1.0 / (1.0 + mse)
                    
                    # 处理NaN
                    if np.isnan(score):
                        score = 0.0
                    
                    print(f"  车辆{vid} GODE预测得分: {score:.4f} (阈值: {threshold})")
                    
                    # 根据得分决定是否需要通信
                    if score < threshold:
                        interest_deduct.append(vid)
                        print(f"  ⚠️  预测偏差大 (score={score:.3f} < {threshold}), 需要通信更新")
                    else:
                        print(f"  ✓ 预测准确 (score={score:.3f} >= {threshold}), 可跳过通信")
                    
                except Exception as e:
                    print(f"  ❌ GODE预测失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # 失败时保守策略: 需要通信
                    interest_deduct.append(vid)
            else:
                # 没有GODE模型,默认需要通信
                print(f"  ⚠️  车辆{vid}无GODE模型,默认需要通信")
                interest_deduct.append(vid)
        
        # ===== 第三步: 构建邻接矩阵 =====
        comm_range = getattr(self, 'comm_range', 100.0)
        for i, vid_i in enumerate(vehicle_ids):
            for j, vid_j in enumerate(vehicle_ids):
                if i != j:
                    # 使用当前位置 (历史的最后一个)
                    pos_i = all_vehicles_history_positions[i][-1]  # [2]
                    pos_j = all_vehicles_history_positions[j][-1]  # [2]
                    distance = np.linalg.norm(pos_i - pos_j)
                    if distance <= comm_range:
                        adjacency_matrix[i, j] = 1.0
        
        # ===== 第四步: 组装cluster状态 =====
        cluster_state = {
            'local_maps': np.stack(local_maps, axis=0),
            'fused_maps': np.stack(fused_maps, axis=0) if fused_maps else None,
            'cur_fused_maps': np.stack(cur_fused_maps, axis=0),
            'positions': np.stack(positions, axis=0),
            'adjacency_matrix': adjacency_matrix,
            'interest_deduct': interest_deduct,
            'num_need_comm': len(interest_deduct),
            'comm_efficiency': 1.0 - len(interest_deduct) / len(self.members) if len(self.members) > 0 else 0.0
        }
        
        print(f"\n{'='*60}")
        print(f"GODE通信决策结果:")
        print(f"  通信周期: {comm_period}步")
        print(f"  总车辆数: {len(self.members)}")
        print(f"  需要通信: {interest_deduct} ({len(interest_deduct)}辆)")
        print(f"  可跳过: {len(self.members) - len(interest_deduct)}辆")
        print(f"  通信节省: {cluster_state['comm_efficiency']*100:.1f}%")
        print(f"{'='*60}\n")
                
        fused_bev_confidence, area_vehicle_counts = fuse_multi_vehicle_bev(
            vehicles_data=self.vehicles_data_list,
            global_bev_config=self.dataset.global_bev_config,
            local_bev_config=self.dataset.local_bev_config,
            fusion_method='max',
            visualize=self.config.visualize,
            interest_deduct = interest_deduct
        )
        interest_maps = area_vehicle_counts
        self.interest_map_seqs.append(interest_maps)
        
        # 融合检测结果
        fused_detections = fuse_multi_vehicle_detections(
            vehicles_data=self.vehicles_data_list,
            iou_threshold=0.5,
            score_threshold=0.3
        )
        
        # for vid, member in self.members.items():
        #     ego_vehicle_world_pos = self.vehicles_data_list[vid-1]['vehicle_global_position']
        #     save_path = os.path.join(member.detVis.output_dir, f"time_{time_step:03d}_vehicle_{vid}_fused_detections.png")
            
        #     member.detVis.visualize_fusion_comparison(
        #         vehicle_id = "vehicle_" + str(vid),
        #         ego_center_world = ego_vehicle_world_pos,
        #         own_projected_positions=self.vehicles_data_list[vid-1]['projected_positions'],
        #         others_projected_positions=fused_detections,
        #         local_bev_config = self.dataset.local_bev_config,
        #         save_path = save_path,
        #     )

        # 根据车辆之间距离，构建邻接矩阵
        for i, member_i in enumerate(self.members.values()):
            pos_i = np.array(positions[i])
            for j, member_j in enumerate(self.members.values()):
                pos_j = np.array(positions[j])
                distance = np.linalg.norm(pos_i - pos_j)
                adjacency_matrix[i, j] = 1/(1 + distance)  # 距离越近，权重越大
        self.adjacency_seqs.append(adjacency_matrix)
        
        # 先固定K*
        # N = self.config.num_vehicles
        # IoU = 0
        # theory_K_star = 0
        # if not self.pre_Iou:
        #     mean_overlap = 0
        #     cnt = 0
            
        #     for i in range(N):
        #         for j in range(i + 1, N):
        #             do_overlap, overlap_degree = self.overlap(cluster_state[i + 1], cluster_state[j + 1])
        #             mean_overlap += overlap_degree
        #             cnt += 1
        #     mean_overlap = mean_overlap / max(cnt, 1)
            
        #     IoU = self.compute_IoU(mean_overlap)
        #     theory_K_star, info = self.comm_comp_model.compute_k_star(self.config.num_vehicles, self.dataset.map_dims[0] * self.dataset.map_dims[1], IoU)
        # else:
        #     IoU = self.pre_IoU_step[time_step]
        #     ind = int(IoU // 0.005)
        #     theory_K_star = self.pre_IoU2K_star[ind]

        # self.IoU = IoU
        # self.K_star_value = theory_K_star
        # self.comm_comp_model._print(self.members[1].local_conf_map.shape[0] * self.members[1].local_conf_map.shape[1], theory_K_star)
        # LOG.info(f"time step {time_step}, IoU {IoU}, theory K star {theory_K_star}")
        # for i in range(N):
        #     self.members[i+1].last_slices_cnt = self.members[i+1].current_slices_limits
        #     self.members[i+1].current_slices_limits = theory_K_star
        #     cluster_state[i+1].update({
        #         'last_slices_cnt': self.members[i+1].current_slices_limits,
        #         'current_slices_limits': theory_K_star})

        states = {
            'local_maps': np.array(local_maps),
            'fused_maps': np.array(fused_maps),
            'cur_fused_maps': np.array(cur_fused_maps),
            'positions': np.array(positions),
            'fused_bev_confidence': np.array(fused_bev_confidence),
            'interest_maps': np.array(interest_maps),
            'vehicles_data_list': self.vehicles_data_list
        }

        # verify_coordinate_mapping(self.vehicles_data_list)

        return states

    def get_state_olde(self, time_step):
        # Return the state of the cluster
        vehicles_at_time_step = self.dataset.get_vehicles_at_time(time_step)
        print(f"检测到 {len(vehicles_at_time_step)} 辆车")
        print(f"{'-'*40}")
        self.vehicles_data_list = []

        cluster_state = {}
        local_maps = []
        fused_maps = []
        cur_fused_maps = []
        positions = []
        adjacency_matrix = np.zeros((len(self.members), len(self.members)), dtype=np.float32)
        
        interest_deduct = []
        thresthod = 0.2

        for vid, member in self.members.items():
            vehicle_info = vehicles_at_time_step[vid]
            print(f"\n处理车辆 {vehicle_info['agent_id']}")
            obs, vehicle_data, projector = member.get_state(vehicle_info, time_step)
            self.vehicles_data_list.append(vehicle_data)
            local_maps.append(obs['local_maps'])
            fused_maps.append(obs['fused_maps'])
            cur_fused_maps.append(obs['cur_fused_maps'])
            positions.append(obs['position'])
            # 如果gode预测到的fused map与实际的fused map 平均score小于阈值，就把这个vid记录到interest_deduct中

            
        fused_bev_confidence, area_vehicle_counts = fuse_multi_vehicle_bev(
            vehicles_data=self.vehicles_data_list,
            global_bev_config=self.dataset.global_bev_config,
            local_bev_config=self.dataset.local_bev_config,
            fusion_method='max',
            visualize=self.config.visualize,
            interest_deduct = interest_deduct
        )
        interest_maps = area_vehicle_counts
        self.interest_map_seqs.append(interest_maps)
        
        # 融合检测结果
        fused_detections = fuse_multi_vehicle_detections(
            vehicles_data=self.vehicles_data_list,
            iou_threshold=0.5,
            score_threshold=0.3
        )
        
        # for vid, member in self.members.items():
        #     ego_vehicle_world_pos = self.vehicles_data_list[vid-1]['vehicle_global_position']
        #     save_path = os.path.join(member.detVis.output_dir, f"time_{time_step:03d}_vehicle_{vid}_fused_detections.png")
            
        #     member.detVis.visualize_fusion_comparison(
        #         vehicle_id = "vehicle_" + str(vid),
        #         ego_center_world = ego_vehicle_world_pos,
        #         own_projected_positions=self.vehicles_data_list[vid-1]['projected_positions'],
        #         others_projected_positions=fused_detections,
        #         local_bev_config = self.dataset.local_bev_config,
        #         save_path = save_path,
        #     )

        # 根据车辆之间距离，构建邻接矩阵
        for i, member_i in enumerate(self.members.values()):
            pos_i = np.array(positions[i])
            for j, member_j in enumerate(self.members.values()):
                pos_j = np.array(positions[j])
                distance = np.linalg.norm(pos_i - pos_j)
                adjacency_matrix[i, j] = 1/(1 + distance)  # 距离越近，权重越大
        self.adjacency_seqs.append(adjacency_matrix)
        
        # 先固定K*
        # N = self.config.num_vehicles
        # IoU = 0
        # theory_K_star = 0
        # if not self.pre_Iou:
        #     mean_overlap = 0
        #     cnt = 0
            
        #     for i in range(N):
        #         for j in range(i + 1, N):
        #             do_overlap, overlap_degree = self.overlap(cluster_state[i + 1], cluster_state[j + 1])
        #             mean_overlap += overlap_degree
        #             cnt += 1
        #     mean_overlap = mean_overlap / max(cnt, 1)
            
        #     IoU = self.compute_IoU(mean_overlap)
        #     theory_K_star, info = self.comm_comp_model.compute_k_star(self.config.num_vehicles, self.dataset.map_dims[0] * self.dataset.map_dims[1], IoU)
        # else:
        #     IoU = self.pre_IoU_step[time_step]
        #     ind = int(IoU // 0.005)
        #     theory_K_star = self.pre_IoU2K_star[ind]

        # self.IoU = IoU
        # self.K_star_value = theory_K_star
        # self.comm_comp_model._print(self.members[1].local_conf_map.shape[0] * self.members[1].local_conf_map.shape[1], theory_K_star)
        # LOG.info(f"time step {time_step}, IoU {IoU}, theory K star {theory_K_star}")
        # for i in range(N):
        #     self.members[i+1].last_slices_cnt = self.members[i+1].current_slices_limits
        #     self.members[i+1].current_slices_limits = theory_K_star
        #     cluster_state[i+1].update({
        #         'last_slices_cnt': self.members[i+1].current_slices_limits,
        #         'current_slices_limits': theory_K_star})

        states = {
            'local_maps': np.array(local_maps),
            'fused_maps': np.array(fused_maps),
            'cur_fused_maps': np.array(cur_fused_maps),
            'positions': np.array(positions),
            'fused_bev_confidence': np.array(fused_bev_confidence),
            'interest_maps': np.array(interest_maps),
            'vehicles_data_list': self.vehicles_data_list
        }

        # verify_coordinate_mapping(self.vehicles_data_list)

        return states
    
    def get_member_state(self, member_id, time_step, action):
        # Return the states of all members in the cluster
        member_states = [member.get_state() for member in self.members]
        return member_states
    
    def step(self, time_step, actions):
        # Update the state of the cluster and its members
        # area_cnt = 0
        # for vid, action in actions.items():
        #     area_cnt += len(action)
        for vid, member in self.members.items():    # receive data to update conf map, need to consider the delay
            area_cnt = len(actions[vid]) if vid in actions.keys() else 0
            member.apply_control(time_step, actions, area_cnt, self.vehicles_data_list)
    
    def compute_reward(self, time_step):
        # 计算cluster内每个车辆fused confidence map中在RoI范围内的confidence value的和
        # reward = reward_fused - reward_local
        rewards = np.zeros(len(self.members), dtype=np.float32)
        for member in self.members.values():
            local_map = member.local_conf_map
            fused_conf_map = member.cur_fused_conf_map

            reward_local = np.sum(local_map) if np.count_nonzero(local_map) > 0 else 0.0
            reward_fused = np.sum(fused_conf_map) if np.count_nonzero(fused_conf_map) > 0 else 0.0
            # rewards.append((member.vid, reward_local, reward_fused))    
            LOG.info(f"Car {member.vid} at time step {time_step} has local RoI reward: {reward_local}, fused RoI reward: {reward_fused}, benefits: {reward_fused - reward_local}")
            rewards[member.vid - 1] = reward_fused - reward_local
        LOG.info(f"Cluster {self.cluster_id} at time step {time_step} has rewards: {rewards}")

        return np.sum(rewards)  # 返回总奖励

def verify_coordinate_mapping(vehicles_info):
    """
    验证坐标映射关系
    """
    print("\n" + "="*80)
    print("坐标映射关系验证")
    print("="*80)
    
    # 选择两个位置不同的车辆

    if len(vehicles_info) < 2:
        print("需要至少2辆车进行验证")
        return
    
    vid1, vid2 = 1, 2
    
    # 车辆1的位置
    pos1 = vehicles_info[vid1]['vehicle_global_position']
    x1, y1, z1 = pos1[0], pos1[1], pos1[2]
    
    # 车辆2的位置
    pos2 = vehicles_info[vid2]['vehicle_global_position']
    x2, y2, z2 = pos2[0], pos2[1], pos2[2]
    
    print(f"\n车辆 {vid1} 全局位置: X={x1:.2f}, Y={y1:.2f}, Z={z1:.2f}")
    print(f"车辆 {vid2} 全局位置: X={x2:.2f}, Y={y2:.2f}, Z={z2:.2f}")
    
    # 计算差异
    delta_x = x2 - x1
    delta_y = y2 - y1
    delta_z = z2 - z1
    
    print(f"\n全局坐标差异:")
    print(f"  ΔX = {delta_x:.2f}m (东西方向)")
    print(f"  ΔY = {delta_y:.2f}m (南北/前后方向)")
    print(f"  ΔZ = {delta_z:.2f}m (左右/高度方向)")
    
    # 如果Y的差异很大，说明车辆主要在前后方向移动
    # 如果Z的差异很大，说明车辆主要在左右方向移动
    
    print(f"\n根据差异判断:")
    if abs(delta_y) > abs(delta_z):
        print(f"  ✓ ΔY ({abs(delta_y):.2f}m) > ΔZ ({abs(delta_z):.2f}m)")
        print(f"  → 车辆主要在Y轴方向移动（前后）")
        print(f"  → BEV X轴应该对应全局Y轴 ✓")
    else:
        print(f"  ⚠️ ΔZ ({abs(delta_z):.2f}m) > ΔY ({abs(delta_y):.2f}m)")
        print(f"  → 车辆主要在Z轴方向移动（左右）")
        print(f"  → 需要重新检查坐标映射！")
    
    print("="*80)
    

    



    


        
 