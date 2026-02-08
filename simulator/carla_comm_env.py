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
# from .toolkit import EnvMonitorOpenCV, Observer, WorldManager

from .carla_base_env import CarlaBaseEnv
from .carla_manager import WorldManager
from .visualize import EnvVisualBase 
from .observer import Observer
from .network import NetworkBase
from .detection import Detection
from .utils import g_camera_params, carla_rotation_to_wxyz, is_regular_sedan
from agent import process_single_vehicle_bev, plot_detections, fuse_multi_vehicle_bev, fuse_multi_vehicle_detections, FasterRCNNDetector

class CarlaCommEnv(CarlaBaseEnv):
    def __init__(self, config):
        super().__init__(config) # 初始化父类
        self.communication = NetworkBase(self._world, config)
        # self._det = Detection()
        self.obs_vehicles = []      # TODO: dose these vehicles need to be fixed during an episode?
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
        
        
        
    def on_reset(self) -> None:
        """
        Override this method to perform additional reset operations.
        Specifically, you can spawn actors and plan routes here.
        """
        # generate vehicles without the need of manual plan
        self._world.spawn_auto_actors(self._config.num_vehicles)
        
        # TODO: generate other objects, like bike, pedestrian
        #
        
        # attach sensors for every vehicle
        while True:
                if len(self.obs_vehicles) >= self._config.obs_vehicles:
                    break
                actor = self._world.spawn_actor()
                self.obs_vehicles.append(actor.id)
                actor_id = actor.id
                observer = Observer(self._world, self._config.observation)
                self._observers.setdefault(actor_id, observer)   
                self.communication.setvehcomm(actor) # set bandwidth for each vehicle
                self.position_seqs.setdefault(actor_id, [])
                self.local_map_seqs.setdefault(actor_id, [])
                self.fused_map_seqs.setdefault(actor_id, [])
                self.cur_fused_conf_map[actor_id] = np.zeros((self._config.conf_map.local_bev_config[ 'grid_size'][0], \
                                        self._config.conf_map.local_bev_config[ 'grid_size'][1]), dtype=np.float32)
        
        self.local_conf_map = {} 
        # self.cur_fused_conf_map = {} # TODO, apply control 中修改
        self.last_fused_conf_map = {}
        self.interest_map_seqs = []
        self.adjacency_seqs = []
        self.v_groups = {}
        
        self.init_vehicle_groups(distance = 200, count=5)
        self.init_obs_vehicle_path()
        # === 新增修复代码：物理热身 ===
        
        print("Warming up simulation for physics settling...")
        for _ in range(20):  # 运行 20 帧让车辆落地
            self._world._world.tick()  # 确保这里调用的是 world.tick()
            
        print("Reset complete. Vehicles should be moving now.")
        
    def init_obs_vehicle_path(self, target_num=2):
        """
        使用 WorldManager 为观测车辆分配生成点并设置分组目的地。
        """
        # 1. 获取所有可用的生成点 (利用 WorldManager 的封装)
        all_spawn_points = self._world.get_spawn_points()
        
        # 2. 随机选择 target_num 个点作为候选目的地
        if len(all_spawn_points) < target_num:
            target_num = len(all_spawn_points)
        destination_points = random.sample(all_spawn_points, target_num)
        dest_locations = [p.location for p in destination_points]

        # 3. 获取 Traffic Manager 的引用
        # 根据 WorldManager.py，TM 运行在 self._world._tm_port
        tm = self._world._vehicle_manager._tm

        # 4. 遍历分组并分配路径
        # 注意：假设此时 self.v_groups 已经由之前的 init_vehicle_groups 生成
        for group_id, vehicle_ids in self.v_groups.items():
            # 为当前组选择一个目的地
            group_destination = dest_locations[group_id % len(dest_locations)]
            
            for v_id in vehicle_ids:
                # 从 actor_dict 获取 actor 实例
                actor = self._world.actor_dict.get(v_id)
                if actor is None:
                    continue
                
                # 确保开启自动驾驶
                actor.set_autopilot(True, self._world._tm_port)
                print(f"vehicle {actor.id} in group {group_id} set auto success")
                
                # 使用 Traffic Manager 设置路径点
                # TM 的 set_path 接受一个 Location 列表作为航点
                tm.set_path(actor, [group_destination])
                
                # 针对协同感知场景的微调：
                # 减小跟车距离，让车队更紧凑，方便观察
                tm.distance_to_leading_vehicle(actor, 2.0)

        print(f"Grouped vehicles are now navigating to {target_num} unique destinations.")
    
    def init_vehicle_groups(self, distance=50, count=5):
        """
        根据距离将车辆分组。
        :param distance: 组内车辆间的最大距离阈值（米）
        :param count: 每个小组的最大成员数量限制
        """
        self.v_groups = {}
        # 获取所有待观察车辆的 actor 对象
        # 假设 self._world.get_actor(id) 可以获取 actor，或者你已经存储了引用
        actor_ids = self.obs_vehicles
        
        # 记录哪些车辆已经被分配了组，防止重复加入
        assigned_vehicles = set()
        group_id = 0

        for i, actor_id in enumerate(actor_ids):
            if actor_id in assigned_vehicles:
                continue
                
            # 创建一个新组，并将当前车辆作为组长/第一个成员
            current_group = [actor_id]
            assigned_vehicles.add(actor_id)
            
            actor_i = self._world.actor_dict[actor_id]
            loc_i = actor_i.get_location()

            # 寻找附近的车辆
            for j, actor_jid in enumerate(actor_ids):
                # 满足以下条件则加入组：
                # 1. 不是自己 2. 没被分配过 3. 组员没满
                if actor_jid != actor_id and actor_jid not in assigned_vehicles:
                    if len(current_group) < count:
                        actor_j = self._world.actor_dict[actor_jid]
                        loc_j = actor_j.get_location()
                        # 计算欧式距离 (L2 norm)
                        dist = loc_i.distance(loc_j)
                        
                        if dist <= distance:
                            current_group.append(actor_j.id)
                            assigned_vehicles.add(actor_j.id)
            
            # 将组存入字典
            self.v_groups[group_id] = current_group
            group_id += 1

        print(f"Successfully grouped {len(assigned_vehicles)} vehicles into {len(self.v_groups)} groups.")
        
    def apply_control(self, action) -> None:
        """
        根据感知得分(action)调节车辆速度。
        :param action: 字典 {actor_id: score}，score > 0 倾向加速，score < 0 倾向减速
        """
        # test
        action = np.random.uniform(0,1, len(self.obs_vehicles))
        # 遍历所有受控的观测车辆
        for i, v_id in enumerate(self.obs_vehicles):
            actor = self._world.actor_dict.get(v_id)
            if actor is None:
                continue

            # 获取当前车辆对应的控制量 (假设 action 是与 obs_vehicles 顺序一致的数组或以 v_id 为键的字典)
            if isinstance(action, dict):
                score = action.get(v_id, 0.0)
            else:
                score = action[i]

            # 1. 获取当前速度 (单位: m/s)
            v = actor.get_velocity()
            current_speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2) # 转换为 km/h

            # 2. 计算目标速度 (Target Speed)
            # 假设：基础速度为 30km/h，根据 score 进行动态调节
            # 你可以根据实际算法需求修改这个映射公式
            base_speed = 30.0
            speed_delta = score * 10.0  # 假设 score 在 [-1, 1] 之间
            target_speed = max(0.0, base_speed + speed_delta) 

            # 3. 通过 VehicleManager 应用控制
            # set_desired_speed 实际上是设置 Traffic Manager 的速度限制
            self._world._vehicle_manager.set_desired_speed(actor, target_speed)

        # 4. 如果需要分享感知数据，可以在此处处理通信逻辑
        # 例如：根据 score 高低决定是否触发 self.communication 广播

    def visualize_topology(self):
        plt.clf()
        
        # 获取当前所有组的所有车辆坐标
        all_pos = []
        v_info = [] # 存储 (v_id, group_id, color)
        colors = ['r', 'g', 'b', 'y', 'm', 'c']
        
        for group_id, vehicle_ids in self.v_groups.items():
            group_color = colors[group_id % len(colors)]
            for v_id in vehicle_ids:
                actor = self._world.actor_dict.get(v_id)
                if actor:
                    loc = actor.get_location()
                    all_pos.append([loc.x, loc.y])
                    v_info.append((v_id, group_id, group_color))
        
        if not all_pos: return
        
        all_pos = np.array(all_pos)
        
        # --- 核心改进：归一化处理 ---
        # 1. 计算中心点
        center = np.mean(all_pos, axis=0)
        rel_pos = all_pos - center
        
        # 2. 计算缩放比例 (将最远的车缩放到 1.0 的范围内)
        max_dist = np.max(np.linalg.norm(rel_pos, axis=1))
        if max_dist > 0:
            norm_pos = rel_pos / max_dist  # 所有点现在都在半径为 1 的圆内
        else:
            norm_pos = rel_pos
        
        # 3. 构建绘图坐标字典
        pos_dict = {v_info[i][0]: norm_pos[i] for i in range(len(v_info))}
        
        # --- 构建 NetworkX 图 ---
        G = nx.Graph()
        node_colors = []
        
        for v_id, gid, gcolor in v_info:
            G.add_node(v_id)
            node_colors.append(gcolor)
            
        # 建立边：根据归一化后的相对距离连线
        # 这里的 threshold 也要相应缩放，或者直接用 KNN
        for i in range(len(v_info)):
            for j in range(i + 1, len(v_info)):
                id1, id2 = v_info[i][0], v_info[j][0]
                dist = np.linalg.norm(norm_pos[i] - norm_pos[j])
                
                # 这里的 0.5 是归一化后的相对距离阈值，可以根据视觉效果调整
                if dist < 0.8: 
                    G.add_edge(id1, id2)

        # --- 绘制 ---
        # 固定轴范围，防止跳变
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        
        nx.draw_networkx_nodes(G, pos_dict, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(G, pos_dict, alpha=0.3, style='--')
        
        plt.title(f"Normalized Topology - Step {self._time_step}")
        
        # 保存逻辑保持不变
        topo_dir = os.path.join(self._config.display.results_dir, "topo")
        os.makedirs(topo_dir, exist_ok=True)
        plt.savefig(os.path.join(topo_dir, f"groups_topo_{self._time_step}.png"), dpi=150)
    
    def on_step(self) -> None:
        """
        Override this method to perform additional operations at each step.
        Specifically, you can update the planner and the route here.
        This method will be called after the simulator ticks.
        """
        # run object detection, get confidence map
        
        # run the RL algorithm to decide what to share based on the observation and communication state
        
        for actor_id in self.obs_vehicles:
            # update cur_fused_conf_map
            print(f"Time Step {self._time_step} Vehicle {actor_id} pos : {self._world._get_actor_transforms()[actor_id]}")
        
        if self._time_step % 10 == 0:
            self.visualize_topology()
        
    def reward(self) -> Tuple[float, Dict]:
        """
        Override this method to define the reward function.
        """
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        _, obs_info = super().reset()
            
        self.communication.reset()

        return self.obs, obs_info


    def step(self, action):
        self.apply_control(action)
        self._world.step()
        self._time_step += 1
        env_state = self.get_state()
        
        # terminated, terminal_conds = self._is_terminal()
        terminated = False
        truncated = False #self._time_step >= self._config.max_episode_steps

        # self.obs, obs_info = self._observer.get_observation(env_state)
        obs_info = {}
        self.obs = {}
        for actor_id, observer in self._observers.items():
            one_obs, one_obs_info = observer.get_observation(self.get_state())    
            self.obs.setdefault(actor_id, one_obs)
            obs_info.setdefault(actor_id, one_obs_info)
        reward, reward_info = 0, {} # self.reward()
        info = {}

        # info = {
        #     **env_state,
        #     **terminal_conds,
        #     **obs_info,
        #     **reward_info,
        #     "action": action,
        # }
        # if self._config.eval:
        #     info = {f"eval_{k}": v for k, v in info.items()}
        #     self.obs.update(info)
            
        # if self._config.display.enable:
        #     self._render(self.obs, info)
        return self.obs, reward, terminated, truncated, info
    
    def run_detection(self, actor_id: int):
        """
        对每辆车运行目标检测，更新感知质量map
        """
        time_step = self._time_step
        state = {}
    
        camera_params = g_camera_params
        ego_params = self._world._get_actor_transforms()[actor_id]
        ego_params = {
            "translation": [ego_params.location.x, ego_params.location.y, ego_params.location.z],
            "rotation": carla_rotation_to_wxyz(ego_params.rotation) # [ego_params.rotation.roll, ego_params.rotation.pitch, ego_params.rotation.yaw],
        }

        detector = self.detector(device='cuda', conf_threshold=0.2)
        import cv2
        # image = cv2.imread(image_path)
        obs, info = self._observers[actor_id].get_observation(state)
        image = obs['camera']
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
            agent_id= actor_id, #vehicle_info['agent_id'],
            image_rgb=image_rgb,
            detections=detections,
            camera_params=camera_params,
            ego_vehicle_params=ego_params,
            local_bev_config=self._config.conf_map.local_bev_config,
            global_bev_config=self._config.conf_map.global_bev_config
        )

        if True: 
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
                        local_bev_config=self._config.conf_map.local_bev_config,
                        local_center='vehicle'  # 或 'camera'
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
            # 标记相机位置
            # cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1] # 车辆坐标系中位置
            cam_x, cam_y = projector.camera_global_position[0], projector.camera_global_position[1]  # 全局坐标系中位置
            axes[0][1].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
            axes[0][1].legend()

            # Local BEV覆盖地图
            im2 = axes[1][0].imshow(bev_confidence_local, cmap='hot', origin='lower', 
                        extent=[self._config.conf_map.local_bev_config['area_extents'][0][0], 
                                self._config.conf_map.local_bev_config['area_extents'][0][1],
                                self._config.conf_map.local_bev_config['area_extents'][1][0], 
                                self._config.conf_map.local_bev_config['area_extents'][1][1]])
            axes[1][0].set_title('Local BEV Confidence Map')
            axes[1][0].set_xlabel('X (meters)')
            axes[1][0].set_ylabel('Y (meters)')
            plt.colorbar(im2, ax=axes[1][0])

            # 标记相机位置
            cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1] # 车辆坐标系中位置
            axes[1][0].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
            axes[1][0].legend()

            # BEV置信度地图
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
            # 标记相机位置
            cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1] # 车辆坐标系中位置
            # cam_x, cam_y = projector.camera_global_position[0], projector.camera_global_position[1]  # 全局坐标系中位置
            axes[1][1].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
            axes[1][1].legend()

            plt.tight_layout()
            figdir = os.path.join(self._config.display.results_dir, "det", f"vehicle_{actor_id}")
            os.makedirs(figdir, exist_ok=True)
            figfile = os.path.join(figdir, f"detection_bev_{time_step:03d}.png")
            plt.savefig(figfile, dpi=150)
            # # plt.show()

        return vehicle_data, projector 
    
    def member_state(self, actor_id: int):       
        obs = {'vid': actor_id, 'time_step': self._time_step}
        time_step = self._time_step
        
        vehicle_info, projector = self.run_detection(actor_id)  # 使用点云数量检测

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
        # Return the state of the cluster
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
        
        # 融合检测结果
        fused_detections = fuse_multi_vehicle_detections(
            vehicles_data=vehicles_data_list,
            iou_threshold=0.5,
            score_threshold=0.3
        )
        
        # for actor_id, actor in self._world.actor_dict.items():
        #     ego_vehicle_world_pos = vehicles_data_list[actor_id]['vehicle_global_position']
        #     save_path = os.path.join(member.detVis.output_dir, f"time_{time_step:03d}_vehicle_{vid}_fused_detections.png")
            
        #     member.detVis.visualize_fusion_comparison(
        #         vehicle_id = "vehicle_" + str(vid),
        #         ego_center_world = ego_vehicle_world_pos,
        #         own_projected_positions=vehicles_data_list[vid-1]['projected_positions'],
        #         others_projected_positions=fused_detections,
        #         local_bev_config = self.dataset.local_bev_config,
        #         save_path = save_path,
        #     )

        # 根据车辆之间距离，构建邻接矩阵
        for i, member_i in enumerate(self.obs_vehicles):
            pos_i = np.array(positions[i])
            for j, member_j in enumerate(self.obs_vehicles):
                pos_j = np.array(positions[j])
                distance = np.linalg.norm(pos_i - pos_j)
                adjacency_matrix[i, j] = 1/(1 + distance)  # 距离越近，权重越大
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

        # verify_coordinate_mapping(self.vehicles_data_list)

        return states        
    
    def wrapper_state(self, obs):
        """
        返回:
        local_maps: [N_v, T, H, W] (float32)
        fused_maps: [N_v, T, H, W] (float32)
        adjacency_matrix: [T, N_v, N_v] (float32)
        positions: [N_v, T, 2]   (float32)
        interest_maps: [T, H_g, W_g]  (float32)
        """
        T = 1
        N_v = len(self.obs_vehicles)
        local_maps = []
        fused_maps = []
        positions = []
        interest_maps = []
        for vid, car in self.cars.items():
            local_map_seq = car.local_map_seqs[-T:]  # 最近T个时间步
            fused_map_seq = car.fused_map_seqs[-T:]  # 最近T个时间步
            position_seq = car.position_seqs[-T:]    # 最近T个时间步
            local_maps.append(local_map_seq)
            fused_maps.append(fused_map_seq)
            positions.append(position_seq)
        interest_maps = self.clusters[0].interest_map_seqs[-T:]  # 最近T个时间步
        local_maps = np.array(local_maps)          # [N_v, T, H_l, W_l]
        fused_maps = np.array(fused_maps)          # [N_v, T, H, W]
        positions = np.array(positions)            # [N_v, T, 2]
        interest_maps = np.array(interest_maps) # [T, H_g, W_g]
        adjs = self.clusters[0].adjacency_seqs[-T:]  # 最近T个时间步

        states = {
            'local_maps': local_maps,
            'fused_maps': fused_maps,
            'positions': positions,
            'interest_maps': interest_maps,
            'adjs': adjs
        }

        LOG.info(f"wrapper_state: local_maps shape: {local_maps.shape}, fused_maps shape: {fused_maps.shape}, \
                 positions shape: {positions.shape}, interest_maps shape: {interest_maps.shape}, adjs length: {len(adjs)}")


        return states