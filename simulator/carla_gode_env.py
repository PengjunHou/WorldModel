from abc import abstractmethod
from typing import Dict, Tuple, Any, Optional

import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import random
import networkx as nx
import torch

from .carla_base_env import CarlaBaseEnv
from .carla_manager import WorldManager
from .visualize import EnvVisualBase 
from .observer import Observer
from .network import NetworkBase
from .detection import Detection
from .utils import g_camera_params, carla_rotation_to_wxyz, is_regular_sedan
from agent import (process_single_vehicle_bev, 
                   plot_detections, fuse_multi_vehicle_bev, fuse_multi_vehicle_detections, FasterRCNNDetector,
                   GODECoordinationManager, extract_vehicle_state, build_vehicle_graph
                )

class CarlaGodeEnv(CarlaBaseEnv):
    """
    集成GODE的CARLA通信环境
    
    在原有CarlaCommEnv基础上添加:
    1. 长周期通信 (T_comm): 车队内车辆周期性交换信息
    2. 短周期决策 (T_action): 基于GODE预测快速决策加速/减速
    3. GODE网络: 部署在每辆车上,预测未来confidence map
    """
    
    def __init__(self, config):
        super().__init__(config)  # 初始化父类
        
        # 原有组件
        self.communication = NetworkBase(self._world, config)
        self.obs_vehicles = []
        self.local_conf_map = {} 
        self.cur_fused_conf_map = {}
        self.last_fused_conf_map = {}
        
        self.position_seqs = {}
        self.local_map_seqs = {}
        self.fused_map_seqs = {}
        self.interest_map_seqs = []
        self.adjacency_seqs = []
        
        self.detector = FasterRCNNDetector
        
        # vehicle group
        self.v_groups = {}
        
        # ========== 新增: GODE组件 ==========
        self.use_gode = getattr(config, 'use_gode', True)
        
        if self.use_gode:
            self.gode_manager = GODECoordinationManager(
                config=config,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("[GODE] GODE manager initialized")
        
        # 训练模式标志
        self.training_mode = getattr(config, 'training_mode', False)
        
        if self.training_mode and self.use_gode:
            self.setup_gode_training()
    
    def setup_gode_training(self):
        """设置GODE训练组件"""
        # 优化器
        self.gode_optimizer = torch.optim.Adam(
            list(self.gode_manager.gode_model.parameters()) + 
            list(self.gode_manager.action_module.parameters()),
            lr=getattr(self._config.gode, 'learning_rate', 1e-4)
        )
        
        # 损失函数
        self.gode_criterion = torch.nn.MSELoss()
        
        # 训练数据缓存
        self.gode_training_buffer = []
        
        print("[GODE] Training mode enabled")
        
    def on_reset(self) -> None:
        """
        Override this method to perform additional reset operations.
        Specifically, you can spawn actors and plan routes here.
        """
        # 原有的重置逻辑
        # generate vehicles without the need of manual plan
        self._world.spawn_auto_actors(self._config.num_vehicles)
        
        # attach sensors for every vehicle
        while True:
            if len(self.obs_vehicles) >= self._config.obs_vehicles:
                break
            actor = self._world.spawn_actor()
            self.obs_vehicles.append(actor.id)
            actor_id = actor.id
            observer = Observer(self._world, self._config.observation)
            self._observers.setdefault(actor_id, observer)   
            self.communication.setvehcomm(actor)  # set bandwidth for each vehicle
            self.position_seqs.setdefault(actor_id, [])
            self.local_map_seqs.setdefault(actor_id, [])
            self.fused_map_seqs.setdefault(actor_id, [])
            self.cur_fused_conf_map[actor_id] = np.zeros(
                (self._config.conf_map.local_bev_config['grid_size'][0], 
                 self._config.conf_map.local_bev_config['grid_size'][1]), 
                dtype=np.float32
            )
        
        # 重置数据结构
        self.local_conf_map = {} 
        self.last_fused_conf_map = {}
        self.interest_map_seqs = []
        self.adjacency_seqs = []
        self.v_groups = {}
        
        # 车辆分组
        self.init_vehicle_groups(distance=200, count=5)
        self.init_obs_vehicle_path()
        
        # ========== 新增: GODE初始化 ==========
        if self.use_gode:
            # 注册GODE智能体
            for actor_id in self.obs_vehicles:
                self.gode_manager.register_vehicle(actor_id)
            
            # 重置时间计数
            self.gode_manager.last_comm_time = 0
            self.gode_manager.last_action_time = 0
            
            print(f"[GODE] Registered {len(self.obs_vehicles)} vehicles")
            print(f"[GODE] Vehicle groups: {len(self.v_groups)}")
            for group_id, vehicle_ids in self.v_groups.items():
                print(f"  Group {group_id}: {vehicle_ids}")
        
        # 物理热身
        print("Warming up simulation for physics settling...")
        for _ in range(20):
            self._world._world.tick()
            
        print("Reset complete. Vehicles should be moving now.")
        
    def init_obs_vehicle_path(self, target_num=2):
        """
        使用 WorldManager 为观测车辆分配生成点并设置分组目的地。
        """
        all_spawn_points = self._world.get_spawn_points()
        
        if len(all_spawn_points) < target_num:
            target_num = len(all_spawn_points)
        destination_points = random.sample(all_spawn_points, target_num)
        dest_locations = [p.location for p in destination_points]

        tm = self._world._vehicle_manager._tm

        for group_id, vehicle_ids in self.v_groups.items():
            group_destination = dest_locations[group_id % len(dest_locations)]
            
            for v_id in vehicle_ids:
                actor = self._world.actor_dict.get(v_id)
                if actor is None:
                    continue
                
                actor.set_autopilot(True, self._world._tm_port)
                print(f"vehicle {actor.id} in group {group_id} set auto success")
                
                tm.set_path(actor, [group_destination])
                tm.distance_to_leading_vehicle(actor, 2.0)

        print(f"Grouped vehicles are now navigating to {target_num} unique destinations.")
    
    def init_vehicle_groups(self, distance=50, count=5):
        """
        根据距离将车辆分组。
        :param distance: 组内车辆间的最大距离阈值（米）
        :param count: 每个小组的最大成员数量限制
        """
        self.v_groups = {}
        actor_ids = self.obs_vehicles
        
        assigned_vehicles = set()
        group_id = 0

        for i, actor_id in enumerate(actor_ids):
            if actor_id in assigned_vehicles:
                continue
                
            current_group = [actor_id]
            assigned_vehicles.add(actor_id)
            
            actor_i = self._world.actor_dict[actor_id]
            loc_i = actor_i.get_location()

            for j, actor_jid in enumerate(actor_ids):
                if actor_jid != actor_id and actor_jid not in assigned_vehicles:
                    if len(current_group) < count:
                        actor_j = self._world.actor_dict[actor_jid]
                        loc_j = actor_j.get_location()
                        dist = loc_i.distance(loc_j)
                        
                        if dist <= distance:
                            current_group.append(actor_j.id)
                            assigned_vehicles.add(actor_j.id)
            
            self.v_groups[group_id] = current_group
            group_id += 1

        print(f"Successfully grouped {len(assigned_vehicles)} vehicles into {len(self.v_groups)} groups.")
        
    def apply_control(self, action=None) -> None:
        """
        应用控制 - 集成GODE决策
        
        如果使用GODE: action由GODE生成,忽略输入的action
        否则: 使用原有的随机控制
        """
        if self.use_gode:
            # GODE模式: action已经在on_step中生成并应用
            # 这里不需要做任何事
            pass
        else:
            # 原有的随机控制
            action = np.random.uniform(0, 1, len(self.obs_vehicles))
            for i, v_id in enumerate(self.obs_vehicles):
                actor = self._world.actor_dict.get(v_id)
                if actor is None:
                    continue

                v = actor.get_velocity()
                current_speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

                score = action[i] if isinstance(action, np.ndarray) else action.get(v_id, 0.0)
                base_speed = 30.0
                speed_delta = score * 10.0
                target_speed = max(0.0, base_speed + speed_delta)

                self._world._vehicle_manager.set_desired_speed(actor, target_speed)
    
    def apply_control_with_gode(self, actions: Dict[int, float]) -> None:
        """
        应用GODE决策的控制
        
        Args:
            actions: {vehicle_id: acceleration} 范围[-1, 1]
        """
        for v_id, accel_action in actions.items():
            actor = self._world.actor_dict.get(v_id)
            if actor is None:
                continue
            
            # 获取当前速度
            v = actor.get_velocity()
            current_speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)  # km/h
            
            # 根据GODE action计算目标速度
            base_speed = 30.0  # km/h
            max_speed_change = 20.0  # km/h
            
            target_speed = base_speed + accel_action * max_speed_change
            target_speed = np.clip(target_speed, 0.0, 60.0)
            
            # 应用速度控制
            self._world._vehicle_manager.set_desired_speed(actor, target_speed)
            
            # 记录日志
            if self._time_step % 10 == 0:
                print(f"[GODE Action] Vehicle {v_id}: action={accel_action:.2f}, "
                      f"current_speed={current_speed:.1f}, target_speed={target_speed:.1f}")

    def visualize_topology(self):
        """可视化车队拓扑"""
        plt.clf()
        
        plt.xlim(-150, 150) 
        plt.ylim(-150, 150)

        colors = ['r', 'g', 'b', 'y', 'm', 'c']
        
        G = nx.Graph()
        pos = {}
        node_colors = []

        for group_id, vehicle_ids in self.v_groups.items():
            group_color = colors[group_id % len(colors)]
            for v_id in vehicle_ids:
                actor = self._world.actor_dict.get(v_id)
                if actor:
                    loc = actor.get_location()
                    G.add_node(v_id, group=group_id)
                    pos[v_id] = np.array([loc.x, loc.y])
                    node_colors.append(group_color)

        for group_id, vehicle_ids in self.v_groups.items():
            group_color = colors[group_id % len(colors)]
            group_actors_ids = [v for v in vehicle_ids if v in pos]
            
            for i in range(len(group_actors_ids)):
                for j in range(i + 1, len(group_actors_ids)):
                    id1, id2 = group_actors_ids[i], group_actors_ids[j]
                    dist = np.linalg.norm(pos[id1] - pos[id2])
                    
                    if dist < 200.0:
                        G.add_edge(id1, id2, color=group_color, weight=(1.0 - dist/200.0))

        edges = G.edges()
        edge_colors = [G[u][v]['color'] for u, v in edges]
        alphas = [G[u][v]['weight'] for u, v in edges]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=80, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, 
                            style='dashed', alpha=0.4)

        plt.grid(True, linestyle=':', alpha=0.5)
        plt.title(f'Step {self._time_step} - Vehicle Topology')
        
        topo_dir = os.path.join(self._config.display.results_dir, "topo")
        os.makedirs(topo_dir, exist_ok=True)
        plt.savefig(os.path.join(topo_dir, f"groups_topo_{self._time_step}.png"), dpi=180)
    
    def on_step(self) -> None:
        """
        每一步的操作
        
        集成GODE逻辑:
        1. 每T_action步: 基于GODE预测做出行动决策
        2. 每T_comm步: 执行车辆间通信
        """
        current_time = self._time_step
        
        if self.use_gode:
            # ========== GODE模式 ==========
            
            # 1. 更新每辆车的本地观测
            vehicle_positions = {}
            for actor_id in self.obs_vehicles:
                actor = self._world.actor_dict.get(actor_id)
                if actor is None:
                    continue
                
                # 提取车辆状态
                vehicle_state = extract_vehicle_state(actor, self._world)
                vehicle_positions[actor_id] = vehicle_state[:2]  # x, y
                
                # 获取本地confidence map
                local_map = self.local_conf_map.get(actor_id)
                if local_map is not None:
                    self.gode_manager.agents[actor_id].update_local_observation(
                        local_map=local_map,
                        vehicle_state=vehicle_state
                    )
            
            # 2. 周期性通信 (长周期)
            if self.gode_manager.should_communicate(current_time):
                self.gode_manager.perform_communication(
                    vehicle_groups=self.v_groups,
                    current_time=current_time
                )
            
            # 3. 预测并决策 (短周期)
            if self.gode_manager.should_decide_action(current_time):
                # 构建邻接矩阵
                if len(self.adjacency_seqs) > 0:
                    adjacency_matrix = self.adjacency_seqs[-1]
                else:
                    adjacency_matrix = None
                
                # GODE预测并决策
                actions = self.gode_manager.predict_and_decide(
                    vehicle_groups=self.v_groups,
                    vehicle_positions=vehicle_positions,
                    adjacency_matrix=adjacency_matrix,
                    current_time=current_time
                )
                
                # 应用控制
                self.apply_control_with_gode(actions)
            
            # 4. 训练模式下收集数据
            if self.training_mode:
                self.collect_gode_training_data()
        else:
            # ========== 原有模式 ==========
            for actor_id in self.obs_vehicles:
                print(f"Time Step {self._time_step} Vehicle {actor_id} pos : {self._world._get_actor_transforms()[actor_id]}")
        
        # 可视化
        self.visualize_topology()
    
    def collect_gode_training_data(self):
        """收集GODE训练数据"""
        local_maps = []
        states = []
        vehicle_ids = []
        
        for actor_id in self.obs_vehicles:
            if actor_id in self.gode_manager.agents:
                agent = self.gode_manager.agents[actor_id]
                if len(agent.local_map_history) > 0:
                    local_maps.append(agent.local_map_history[-1])
                    states.append(agent.state_history[-1])
                    vehicle_ids.append(actor_id)
        
        if len(local_maps) > 0:
            # 构建图
            positions = np.array([s[:2] for s in states])
            graph = build_vehicle_graph(positions, max_comm_range=200.0)
            
            # 存储
            self.gode_training_buffer.append({
                'local_maps': np.array(local_maps),
                'states': np.array(states),
                'graph': graph,
                'vehicle_ids': vehicle_ids,
                'time_step': self._time_step
            })
    
    def train_gode(self, num_steps: int = 100):
        """训练GODE模型"""
        if not self.use_gode or not self.training_mode:
            print("[GODE] Training not enabled")
            return
        
        if len(self.gode_training_buffer) < 2:
            print("[GODE] Not enough training data")
            return
        
        print(f"[GODE] Starting training with {len(self.gode_training_buffer)} samples")
        
        for step in range(num_steps):
            # 随机采样一对连续的时间步
            idx = np.random.randint(0, len(self.gode_training_buffer) - 1)
            
            current_data = self.gode_training_buffer[idx]
            next_data = self.gode_training_buffer[idx + 1]
            
            # 准备batch数据
            batch_data = {
                'local_maps': torch.FloatTensor(current_data['local_maps']).unsqueeze(0),
                'states': torch.FloatTensor(current_data['states']).unsqueeze(0),
                'graphs': [current_data['graph']],
                'target_maps': torch.FloatTensor(next_data['local_maps']).unsqueeze(0)
            }
            
            # 训练一步
            loss = self.gode_manager.train_step(
                batch_data=batch_data,
                optimizer=self.gode_optimizer,
                criterion=self.gode_criterion
            )
            
            if step % 10 == 0:
                print(f"[GODE Training] Step {step}/{num_steps}, Loss: {loss:.4f}")
        
        print("[GODE] Training complete")
        
    def reward(self) -> Tuple[float, Dict]:
        """
        计算奖励
        """
        total_reward = 0.0
        reward_info = {}
        
        if self.use_gode:
            # GODE奖励: 基于感知质量改进
            perception_reward = 0.0
            for actor_id in self.obs_vehicles:
                if actor_id in self.local_conf_map and actor_id in self.cur_fused_conf_map:
                    local_map = self.local_conf_map[actor_id]
                    fused_map = self.cur_fused_conf_map[actor_id]
                    
                    # 融合改进
                    improvement = np.mean(fused_map) - np.mean(local_map)
                    perception_reward += improvement * 10.0
            
            # 通信成本
            comm_penalty = 0.0
            if self.gode_manager.should_communicate(self._time_step):
                comm_penalty = -0.1
            
            # 速度保持
            speed_reward = 0.0
            for actor_id in self.obs_vehicles:
                actor = self._world.actor_dict.get(actor_id)
                if actor:
                    v = actor.get_velocity()
                    speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                    if 20 <= speed <= 40:
                        speed_reward += 0.1
            
            total_reward = perception_reward + comm_penalty + speed_reward
            
            reward_info = {
                'perception_reward': perception_reward,
                'comm_penalty': comm_penalty,
                'speed_reward': speed_reward
            }
        else:
            # 原有奖励
            total_reward = 0.0
            reward_info = {}
        
        return total_reward, reward_info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """重置环境"""
        _, obs_info = super().reset()
            
        self.communication.reset()
        
        # 清空GODE训练缓存
        if self.use_gode and self.training_mode:
            self.gode_training_buffer = []

        return self.obs, obs_info

    def step(self, action=None):
        """环境步进"""
        # 应用控制 (GODE模式下会在on_step中处理)
        if not self.use_gode:
            self.apply_control(action)
        
        # 仿真步进
        self._world.step()
        self._time_step += 1
        
        # 获取状态
        env_state = self.get_state()
        
        # 执行每步逻辑
        self.on_step()
        
        # 计算终止条件
        terminated = False
        truncated = False
        
        # 获取观测
        self.obs = {}
        obs_info = {}
        for actor_id, observer in self._observers.items():
            one_obs, one_obs_info = observer.get_observation(env_state)    
            self.obs.setdefault(actor_id, one_obs)
            obs_info.setdefault(actor_id, one_obs_info)
        
        # 计算奖励
        reward, reward_info = self.reward()
        
        info = {
            **env_state,
            **obs_info,
            **reward_info,
            "action": action if action is not None else {},
            "time_step": self._time_step
        }
        
        return self.obs, reward, terminated, truncated, info
    
    # ========== 保留原有方法 ==========
    
    def run_detection(self, actor_id: int):
        """对每辆车运行目标检测，更新感知质量map"""
        time_step = self._time_step
        state = {}
    
        camera_params = g_camera_params
        ego_params = self._world._get_actor_transforms()[actor_id]
        ego_params = {
            "translation": [ego_params.location.x, ego_params.location.y, ego_params.location.z],
            "rotation": carla_rotation_to_wxyz(ego_params.rotation)
        }

        detector = self.detector(device='cuda', conf_threshold=0.2)
        import cv2
        obs, info = self._observers[actor_id].get_observation(state)
        image = obs['camera']
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect(image)
        
        vehicle_data, projector = process_single_vehicle_bev(
            agent_id=actor_id,
            image_rgb=image_rgb,
            detections=detections,
            camera_params=camera_params,
            ego_vehicle_params=ego_params,
            local_bev_config=self._config.conf_map.local_bev_config,
            global_bev_config=self._config.conf_map.global_bev_config
        )

        if True: 
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes[0][0].set_title('Image Detections')
            axes[0][0].imshow(plot_detections(image_rgb, detections))
            axes[0][0].set_xlim(0, 1600)
            axes[0][0].set_ylim(900, 0)
            
            bev_confidence_global, bev_coverage_global, projected_positions,\
                    bev_confidence_local, bev_coverage_local, local_info = projector.project_detections_to_bev(
                        detections,
                        method='depth_estimation',
                        local_bev_config=self._config.conf_map.local_bev_config,
                        local_center='vehicle'
                )
            im1 = axes[0][1].imshow(bev_confidence_global, cmap='hot', origin='lower', 
                        extent=[self._config.conf_map.global_bev_config['area_extents'][0][0], 
                                self._config.conf_map.global_bev_config['area_extents'][0][1],
                                self._config.conf_map.global_bev_config['area_extents'][1][0], 
                                self._config.conf_map.global_bev_config['area_extents'][1][1]])
            axes[0][1].set_title('Global BEV Confidence Map')
            axes[0][1].set_xlabel('X (meters)')
            axes[0][1].set_ylabel('Y (meters)')
            plt.colorbar(im1, ax=axes[0][1])
            cam_x, cam_y = projector.camera_global_position[0], projector.camera_global_position[1]
            axes[0][1].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
            axes[0][1].legend()

            im2 = axes[1][0].imshow(bev_confidence_local, cmap='hot', origin='lower', 
                        extent=[self._config.conf_map.local_bev_config['area_extents'][0][0], 
                                self._config.conf_map.local_bev_config['area_extents'][0][1],
                                self._config.conf_map.local_bev_config['area_extents'][1][0], 
                                self._config.conf_map.local_bev_config['area_extents'][1][1]])
            axes[1][0].set_title('Local BEV Confidence Map')
            axes[1][0].set_xlabel('X (meters)')
            axes[1][0].set_ylabel('Y (meters)')
            plt.colorbar(im2, ax=axes[1][0])

            cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1]
            axes[1][0].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
            axes[1][0].legend()

            print(f"cur_fused_conf_map shape: {self.cur_fused_conf_map[actor_id].shape}")
            im1 = axes[1][1].imshow(self.cur_fused_conf_map[actor_id], cmap='hot', origin='lower', 
                        extent=[self._config.conf_map.local_bev_config['area_extents'][0][0], 
                                self._config.conf_map.local_bev_config['area_extents'][0][1],
                                self._config.conf_map.local_bev_config['area_extents'][1][0], 
                                self._config.conf_map.local_bev_config['area_extents'][1][1]])
            axes[1][1].set_title('Global BEV Confidence Map')
            axes[1][1].set_xlabel('X (meters)')
            axes[1][1].set_ylabel('Y (meters)')
            plt.colorbar(im1, ax=axes[1][1])
            cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1]
            axes[1][1].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
            axes[1][1].legend()

            plt.tight_layout()
            figdir = os.path.join(self._config.display.results_dir, "det", f"vehicle_{actor_id}")
            os.makedirs(figdir, exist_ok=True)
            figfile = os.path.join(figdir, f"detection_bev_{time_step:03d}.png")
            plt.savefig(figfile, dpi=150)

        return vehicle_data, projector 
    
    def member_state(self, actor_id: int):       
        obs = {'vid': actor_id, 'time_step': self._time_step}
        time_step = self._time_step
        
        vehicle_info, projector = self.run_detection(actor_id)

        self.local_conf_map.setdefault(actor_id, vehicle_info['bev_confidence_local'])
        self.position_seqs.setdefault(actor_id, []).append(vehicle_info['vehicle_global_position'][:2])
        self.local_map_seqs.setdefault(actor_id, []).append(vehicle_info['bev_confidence_local'].copy())
        if len(self.fused_map_seqs[actor_id]) == 0:
            self.last_fused_conf_map[actor_id] = self.local_conf_map[actor_id]
            self.fused_map_seqs[actor_id].append(self.last_fused_conf_map[actor_id].copy())
        
        obs.update({'position': vehicle_info['vehicle_global_position'][:2],
                      'local_maps': self.local_conf_map[actor_id],
                      'fused_maps': self.last_fused_conf_map[actor_id],
                      'cur_fused_maps': self.cur_fused_conf_map[actor_id],})

        return obs, vehicle_info, projector
            
    def wrapper_obs(self):
        vehicles_data_list = []
        time_step = self._time_step

        cluster_state = {}
        local_maps = []
        fused_maps = []
        cur_fused_maps = []
        positions = []
        adjacency_matrix = np.zeros((len(self.obs_vehicles), len(self.obs_vehicles)), dtype=np.float32)

        for actor_id in self.obs_vehicles:
            print(f"\n处理车辆 {actor_id} 的观测数据...")
            obs, vehicle_data, projector = self.member_state(actor_id) 
            vehicles_data_list.append(vehicle_data)
            local_maps.append(obs['local_maps'])
            fused_maps.append(obs['fused_maps'])
            cur_fused_maps.append(obs['cur_fused_maps'])
            positions.append(obs['position'])

            
        fused_bev_confidence, area_vehicle_counts = fuse_multi_vehicle_bev(
            vehicles_data=vehicles_data_list,
            global_bev_config=self._config.conf_map.global_bev_config,
            local_bev_config=self._config.conf_map.local_bev_config,
            time_step=time_step,
            fusion_method='max',
            visualize=True
        )
        interest_maps = area_vehicle_counts
        self.interest_map_seqs.append(interest_maps)
        
        fused_detections = fuse_multi_vehicle_detections(
            vehicles_data=vehicles_data_list,
            iou_threshold=0.5,
            score_threshold=0.3
        )

        for i, member_i in enumerate(self.obs_vehicles):
            pos_i = np.array(positions[i])
            for j, member_j in enumerate(self.obs_vehicles):
                pos_j = np.array(positions[j])
                distance = np.linalg.norm(pos_i - pos_j)
                adjacency_matrix[i, j] = 1/(1 + distance)
        self.adjacency_seqs.append(adjacency_matrix)

        states = {
            'local_maps': np.array(local_maps),
            'fused_maps': np.array(fused_maps),
            'cur_fused_maps': np.array(cur_fused_maps),
            'positions': np.array(positions),
            'fused_bev_confidence': np.array(fused_bev_confidence),
            'interest_maps': np.array(interest_maps),
            'vehicles_data_list': vehicles_data_list
        }

        return states
    
    # ========== GODE相关方法 ==========
    
    def save_gode_model(self, save_dir: str):
        """保存GODE模型"""
        if self.use_gode:
            self.gode_manager.save_model(save_dir, self._time_step)
    
    def load_gode_model(self, checkpoint_path: str):
        """加载GODE模型"""
        if self.use_gode:
            self.gode_manager.load_model(checkpoint_path)
