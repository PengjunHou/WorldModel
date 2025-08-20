#import carla
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from envs.CarlaVehEnv.Car import Car, Clusters
from envs.CarlaVehEnv.Object import Object
from envs.CarlaVehEnv.CarlaDataCollector import V2XSimReader
from envs.CarlaVehEnv.RSU import RSU
from envs.CarlaVehEnv.utils import init_detection_model
from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from controller.PPO import ModelConfig, ActorCritic, PPO
from torch.utils.tensorboard import SummaryWriter

import logging
LOG = logging.getLogger(__name__)

# 1. Carla-Gymnasium Environment Wrapper
class CarlaEnv(gym.Env):
    metadata = {"render_modes": []}  # gymnasium 规范要求
    def __init__(self, config, full_episode=True, with_obs=True, load_model=True):
        super(CarlaEnv, self).__init__()
        # self.client = carla.Client(config['host'], config['port'])
        # self.world = self.client.load_world(config['town'])

        ## Configuration
        self.config = config
        self.data_path = config.data_path
        self.V2X_config = Config("train", binary=True, only_det=True)
        self.V2X_config_global = ConfigGlobal("train", binary=True, only_det=True)
        split = 'train'
        agent_data_dirs = []
        for agent_id in range(config.num_vehicles):
            agent_data_dirs.append(os.path.join(self.config.v2x_data_path, split, f"agent{agent_id+1}"))
        self.v2x_det_dataset = V2XSimDet(
            dataset_roots=agent_data_dirs, split=split, config_global=self.V2X_config_global,
            config=self.V2X_config , val=True, bound='both')
        LOG.info(f"v2x_det_dataset length: {len(self.v2x_det_dataset)}")
        LOG.info(f"v2x_det_dataset: {self.v2x_det_dataset}")

        self.dataset = V2XSimReader(self.data_path)
        self.det_model, _ = init_detection_model(self.V2X_config, num_agent=config.num_vehicles, com="lowerbound", ckpt_path=None, device=None)

        self._setup_simulation()

        ## RL
        # self.PPO_agent = PPOAgent(
        #     input_dim=obs_dim, hidden_dim=config.hidden_dim, num_areas=self.n_vehicles,
        #     lr=config.lr, gamma=config.gamma, eps_clip=config.eps_clip)

        self.RL_cfg = ModelConfig(H = self.dataset.map_dims[0], W = self.dataset.map_dims[1], num_vehicles=self.n_vehicles, patch_h = self.dataset.map_dims[0], patch_w = self.dataset.map_dims[1])
        self.RL_device = self.RL_cfg.device
        self.RL_model = ActorCritic(self.RL_cfg).to(self.RL_device)
        self.RL_agent = PPO(self.RL_model, self.RL_cfg)
        res_path = "D:/Code/WorldModel/results"
        self.sumary_writer = SummaryWriter(log_dir=os.path.join(res_path, "tensorboard"))


    def _setup_simulation(self):
        self.cur_time_step = 0
        time_step_length = self.dataset.get_time_step_length()
        config_time_step_length = self.config.time_steps
        self.time_step_length = min(config_time_step_length, time_step_length)

        self.create_maps()
        self.create_cars()
        self.create_rsu()
        self.create_objects()

        self.n_clusters = self.config.n_clusters
        self._init_cluster()

    def _init_cluster(self, strategy=None):
        self.clusters = []
        for i in range(self.n_clusters):
            cluster = Clusters(i, leader_id=1, vehicles=self.cars)
            self.clusters.append(cluster)

    def create_maps(self):
        self.global_map_height = self.dataset.map_dims[0]
        self.global_map_width = self.dataset.map_dims[1]
        self.global_conf_map = np.zeros((self.global_map_height, self.global_map_width))

    def create_cars(self):
        self.n_vehicles = self.dataset.get_vehicle_count()
        self.cars = {}
        veh_sensors = self.dataset.get_vehicle_sensors()

        assert self.n_vehicles == len(veh_sensors), "Number of vehicles and sensors do not match"

        for vid, sensor_info in veh_sensors.items():
            car = Car(vid, dataset=self.dataset, V2X_dataset = self.v2x_det_dataset, config = self.config, sensors=sensor_info, det_model = self.det_model)
            self.cars[vid] = car
        LOG.info(f"create cars finish! Number of cars : {self.n_vehicles}")

    def create_rsu(self):
        self.n_rsu = self.dataset.get_rsu_count()
        self.rsu = {}
        rsu_sensors = self.dataset.get_rsu_sensors()

        assert self.n_rsu == len(rsu_sensors), f"Number of RSUs {self.n_rsu} and sensors do not match {len(rsu_sensors)}"

        for rid, sensor_info in rsu_sensors.items():
            rsu = RSU(rid, dataset=self.dataset, sensors=sensor_info)
            self.rsu[rid] = rsu
        LOG.info(f"create rsus finish! Number of rsus : {self.n_rsu}")

    def create_objects(self):
        obj_tokens = self.dataset.get_object_tokens()
        oid = 0
        self.objects = {}
        for token in obj_tokens:
            if len(self.dataset.get_object_data(token)) < self.time_step_length:
                continue
            obj = Object(oid, dataset=self.dataset, token=token)
            self.objects[oid] = obj
            oid += 1
        self.n_objects = oid
        LOG.info(f"create objects finish! Number of objects : {self.n_objects}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cur_time_step = 0
        self._init_cluster()
        obs = self._get_observation(time_step=0)
        info = {}
        return obs, info  # gymnasium 格式

    def step(self, action):
        next_time_step = self.cur_time_step + 1

        for cluster in self.clusters: # TODO，there is only one cluster now
            cluster.step(next_time_step, action)

        if next_time_step >= self.time_step_length:
            terminated, truncated = True, False
            obs = None
        else:
            terminated, truncated = False, False
            obs = self._get_observation(next_time_step)

        reward = self._compute_reward()
        info = {}

        LOG.info(f"Time step {self.cur_time_step}, action {action}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}")
        self.cur_time_step = next_time_step
        reward = torch.tensor([reward], dtype=torch.float32, device=self.RL_device)
        terminated = torch.tensor([terminated], dtype=torch.int8, device=self.RL_device)

        LOG.info(f"shapes -  reward: {reward.shape}, terminated: {terminated.shape}")
        LOG.info(f"Step {self.cur_time_step}: obs={obs}, reward={reward.item()}, terminated={terminated.item()}, truncated={truncated}")

        return obs, reward, terminated, truncated, info

    def overlap(self, agent_obs1, agent_obs2):
        # 判断两个Agent的local confidence map是否有重叠
        local_map1 = agent_obs1.get("local_map")
        local_map2 = agent_obs2.get("local_map")

        if local_map1 is None or local_map2 is None:
            return False, 0.0

        overlap_area = np.logical_and(local_map1 > 0, local_map2 > 0)
        overlap_degree = np.sum(overlap_area) / (np.sum(local_map1 > 0) + np.sum(local_map2 > 0) - np.sum(overlap_area))

        return np.any(overlap_area), overlap_degree

    def _get_observation(self, time_step):
        # 获取当前时间步每个Agent的local confidence map
        # 将其映射到Global map上
        cluster_obs = []
        # object_states = []
        # for oid, obj in self.objects.items():
        #     object_states.append(obj.get_state(time_step))
        for i in range(self.n_clusters):
            cluster_obs.append(self.clusters[i].get_state(time_step))

        obs = cluster_obs  # 可替换为 self._combine_states(cluster_states, object_states)

        return obs
    
    def wrapper_state(self, obs, B: int = 1):
        """
        将 obs 转成带批维的状态表示以适配 PPO：
        - 若 obs 是单环境结构（原来的样子），则把该状态广播成 B 份；
        - 若 obs 是长度为 B 的列表/元组（多环境），则逐个处理后堆叠。
        返回:
        maps_b: [B, N, 1, H, W] (float32)
        adj_b : [B, N, N]       (float32)
        约定:
        单环境时 obs[0] 是 {vid: agent_obs}，并且 agent_obs["local_map"] 为 [H, W]。
        多环境时 obs 是长度为 B 的列表，其中每个元素都满足单环境结构。
        """
        def _process_single(single_obs):
            # single_obs[0] 是 {vid: agent_obs}
            N = self.n_vehicles
            LOG.info(f"single_obs[0].keys(): {single_obs}")
            sample = next(iter(single_obs.values()))
            H, W = sample["local_map"].shape

            maps = []
            adjacency_matrix = np.zeros((N, N), dtype=np.float32)

            for i in range(N):
                agent_obs = single_obs[i+1]            # vid 从 1 开始
                local_map = agent_obs["local_map"]        # [H, W]
                # 可选标准化：
                # local_map = (local_map - local_map.mean()) / (local_map.std() + 1e-6)
                maps.append(np.expand_dims(local_map.astype(np.float32), axis=0))  # [1,H,W]

            maps_np = np.stack(maps, axis=0).astype(np.float32)  # [N,1,H,W]

            # 按你的 overlap 逻辑构建邻接
            for i in range(N):
                for j in range(i+1, N):
                    do_overlap, overlap_degree = self.overlap(single_obs[i+1], single_obs[j+1])
                    if do_overlap:
                        adjacency_matrix[i, j] = adjacency_matrix[j, i] = overlap_degree

            return maps_np, adjacency_matrix  # [N,1,H,W], [N,N]

        if isinstance(obs, (list, tuple)):
            # 多环境：长度就是 B
            batch_maps = []
            batch_adj  = []
            for single_obs in obs:
                maps_np, adj_np = _process_single(single_obs)
                batch_maps.append(maps_np[None, ...])  # [1,N,1,H,W]
                batch_adj.append(adj_np[None, ...])    # [1,N,N]
            maps_b = np.concatenate(batch_maps, axis=0).astype(np.float32)  # [B,N,1,H,W]
            adj_b  = np.concatenate(batch_adj,  axis=0).astype(np.float32)  # [B,N,N]
            feature = torch.tensor(maps_b, dtype=torch.float32).to(self.RL_device)  # [B,N,1,H,W]
            adjacency_matrix = torch.tensor(adj_b, dtype=torch.float32).to(self.RL_device)  # [B,N,N]
            return feature, adjacency_matrix
        else:
            # 单环境：广播成 B 份
            maps_np, adj_np = _process_single(obs)     # [N,1,H,W], [N,N]
            maps_b = np.repeat(maps_np[None, ...], B, axis=0).astype(np.float32)  # [B,N,1,H,W]
            adj_b  = np.repeat(adj_np[None,  ...], B, axis=0).astype(np.float32)  # [B,N,N]
            feature = torch.tensor(maps_b, dtype=torch.float32).to(self.RL_device)  # [B,N,1,H,W]
            adjacency_matrix = torch.tensor(adj_b, dtype=torch.float32).to(self.RL_device)  # [B,N,N]
            return feature, adjacency_matrix
        
    def wrapper_state_old(self, obs):
        # 将obs转换为适合RL的状态表示
        # 构建图神经网络的输入，即邻接矩阵和节点特征
        feature_matrix = []
        adjacency_matrix = np.zeros((self.n_vehicles, self.n_vehicles))
        # LOG.info(f"obs[0]: {obs[0].keys()}")
        for i in range(self.n_vehicles):
            for vid, agent_obs in obs[0].items():
                if agent_obs.get("vid") == i + 1: # 车辆ID从1开始
                    feature = agent_obs.get("local_map").flatten()
                    feature_matrix.append(feature)
                    break
            
            for j in range(self.n_vehicles):
                if i != j and adjacency_matrix[i, j] == 0:
                    do_overlap, overlap_degree = self.overlap(obs[0][i+1], obs[0][j+1])
                    if do_overlap:
                        adjacency_matrix[i, j] = overlap_degree
                        adjacency_matrix[j, i] = overlap_degree
        
        feature_matrix = np.array(feature_matrix)

        LOG.info(f"feature matrix shape: {feature_matrix.shape}, adjacency matrix shape: {adjacency_matrix.shape}")

        return feature_matrix, adjacency_matrix

    def action_select_old(self, feature, adjacency_matrix):
        # 根据当前状态选择动作
        # scores_map, _, _ = self.PPO_agent.select_action(state, adjacency_matrix)       # 这里其实应该生成scores map
        # scores_map = np.random.rand(self.global_conf_map.shape[0], self.global_conf_map.shape[1])
        out = self.RL_model(feature, adjacency_matrix)  
        alpha, beta, value = out["alpha"], out["beta"], out["value"]  # [B,P,Q], [B]
        dist = Beta(alpha, beta)
        scores_map = dist.rsample()                        # [B,P,Q], reparameterized sample
        scores_map = scores_map.clamp(1e-6, 1-1e-6)           # 避免边界数值问题
        logp = dist.log_prob(scores_map).sum(dim=(1,2))   # [B]

        # 找到scores_map中最大的K个值对应的区域索引
        max_k = 3
        if max_k > 0:
            top_k_indices = np.unravel_index(np.argsort(scores_map, axis=None)[-max_k:], scores_map.shape)
            top_k_scores = scores_map[top_k_indices]
            LOG.info(f"Top {max_k} scores: {top_k_scores}, indices: {top_k_indices}")
        else:
            top_k_indices = ([], [])
            top_k_scores = []

        # 比较所有车辆中对应索引位置，谁的值最大。由最大值对应的车辆执行动作，可以将每辆车需要执行的动作存储在一个dict中，key为车辆ID，value为动作
        actions = {}
        for i in range(len(top_k_indices[0])):
            selected_vehicle = -1
            max_value = -1
            for j in range(self.n_vehicles):
                if self.cars[j+1].local_conf_map[top_k_indices[0][i], top_k_indices[1][i]] > max_value:
                    max_value = self.cars[j+1].local_conf_map[top_k_indices[0][i], top_k_indices[1][i]]
                    selected_vehicle = j + 1
            
            if selected_vehicle != -1:
                if selected_vehicle not in actions:
                    actions[selected_vehicle] = []
                actions[selected_vehicle].append((top_k_indices[0][i], top_k_indices[1][i], top_k_scores[i]))
        
        LOG.info(f"Selected actions: {actions}")
        return actions
    
    def action_select(self, state, adjacency_matrix, max_k: int = 3):
        """
        输入:
        state: np.ndarray 或 torch.Tensor, 形状 [B, N, 1, H, W]
        adjacency_matrix: np.ndarray 或 torch.Tensor, 形状 [B, N, N]
        返回:
        若 B==1: dict {vehicle_id: [(p, q, score), ...]}
        若 B>1 : list[dict], 每个 batch 一个 dict
        """
        device = self.RL_device

        # ---- 1) 转成 torch.Tensor 并放到正确设备 ----
        feature = torch.as_tensor(state, dtype=torch.float32, device=device)          # [B,N,1,H,W]
        adjacency = torch.as_tensor(adjacency_matrix, dtype=torch.float32, device=device)  # [B,N,N]

        # ---- 2) 前向网络，采样 scores_map（[B,P,Q]）----
        out = self.RL_model(feature, adjacency)
        alpha, beta, value = out["alpha"], out["beta"], out["value"]  # [B,P,Q], [B]
        dist = Beta(alpha, beta)
        scores_map = dist.rsample().clamp(1e-6, 1-1e-6)               # [B,P,Q]
        logp = dist.log_prob(scores_map).sum(dim=(1, 2))            

        B, N, C, H, W = feature.shape
        _, P, Q = scores_map.shape
        k = max(0, min(int(max_k), P * Q))   #  k 不超过网格大小
        if k == 0:
            return {} if B == 1 else [{} for _ in range(B)]

        # ---- 3) 用 torch.topk 在 patch 网格上取 Top-K（每个 batch 各取 k 个）----
        flat = scores_map.reshape(B, -1)                           # [B, P*Q]
        topk_vals, topk_idx = torch.topk(flat, k=k, dim=1, largest=True, sorted=True)  # [B,k], [B,k]
        topk_rows = topk_idx // Q                                  # [B,k]
        topk_cols = topk_idx % Q                                   # [B,k]

        # ---- 4) 把每辆车的 local map 下采样到 [P,Q]，保证索引一致 ----
        # feature: [B,N,1,H,W] -> [B*N,1,H,W] -> interpolate -> [B,N,P,Q]
        low_maps = F.interpolate(
            feature.view(B * N, 1, H, W),
            size=(P, Q), mode="bilinear", align_corners=False
        ).view(B, N, P, Q)                                         # 每辆车在 patch 网格的值

        # ---- 5) 对每个被选中的 patch (p,q)，在 N 辆车里选最大值的那辆 ----
        actions_batch = []
        for b in range(B):
            actions = {}
            for m in range(k):
                p = int(topk_rows[b, m].item())
                q = int(topk_cols[b, m].item())

                # 在 N 辆车这个 (p,q) 位置上取最大者
                per_vehicle_vals = low_maps[b, :, p, q]            # [N]
                picked_j = int(torch.argmax(per_vehicle_vals).item())
                vid = picked_j + 1                                 # 车辆ID从 1 开始

                score = float(topk_vals[b, m].item())  
                actions.setdefault(vid, []).append((p, q, score))
            actions_batch.append(actions)

        if B == 1:
            return scores_map, actions_batch[0], logp, value
        else:
            return scores_map, actions_batch, logp, value


    def _compute_reward(self):
        rewards = []
        for cluster in self.clusters:
            rewards.append(cluster.compute_reward(self.cur_time_step))
        return rewards[0]
