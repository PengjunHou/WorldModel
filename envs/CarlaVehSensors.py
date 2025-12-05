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
from envs.CarlaVehEnv.utils import init_detection_model, topk_2d
from controller.PPO import ModelConfig, ActorCritic, PPO
from controller.DPPO_agent import DPPO
from controller.model.diffusion_ppo import PPODiffusion
from controller.cfg.dppo_cfg import DPPOConfig
from torch.utils.tensorboard import SummaryWriter
import matplotlib.animation as animation
from src.det.FastCNNDet import FasterRCNNDetector

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
        self.visualize = config.visualize
        self.dataset = V2XSimReader(self.data_path)
        # statistic data
        self.statistic = {}
        self.detector = FasterRCNNDetector

        self._setup_simulation()

        ## RL
        # self.PPO_agent = PPOAgent(
        #     input_dim=obs_dim, hidden_dim=config.hidden_dim, num_areas=self.n_vehicles,
        #     lr=config.lr, gamma=config.gamma, eps_clip=config.eps_clip)
        self.RL_device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.strategy == "RL" and config.RL_model == "PPO":
            self.RL_cfg = ModelConfig(H = self.dataset.map_dims[0], W = self.dataset.map_dims[1], num_vehicles=self.n_vehicles, patch_h = self.dataset.map_dims[0], patch_w = self.dataset.map_dims[1])
            self.RL_device = self.RL_cfg.device
            self.RL_model = ActorCritic(self.RL_cfg).to(self.RL_device)
            self.RL_agent = PPO(self.RL_model, self.RL_cfg)
        elif config.strategy == "RL" and config.RL_model == "DPPO":
            obs_shape = self.dataset.map_dims
            self.RL_cfg = DPPOConfig(config, obs_dim=obs_shape, action_dim=obs_shape)
            self.RL_device = self.RL_cfg.device
            # self.RL_model = PPODiffusion(self.RL_cfg)
            self.RL_agent = DPPO(self.RL_cfg)
        res_path = config.result_path
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
        self.IoU_dist_proc()

        self.n_clusters = self.config.n_clusters
        self._init_cluster()
    
    def IoU_dist_proc(self):
        self.IoU_per_step = []
        self.tmp_IoU2K_star = [[] for _ in range(int(1/0.005))]
        self.IoU2K_star =  []
        self.Iou_pre_done = False

    def _init_cluster(self, strategy=None):
        self.clusters = []
        for i in range(self.n_clusters):
            cluster = Clusters(self.config, self.dataset, i, leader_id=1, vehicles=self.cars, pre_IoU_step = self.IoU_per_step, pre_IoU2K_star = self.IoU2K_star, pre_Iou = self.Iou_pre_done)
            self.clusters.append(cluster)

    def create_maps(self):
        # self.global_map_height = self.dataset.map_dims[0]
        # self.global_map_width = self.dataset.map_dims[1]
        # self.global_conf_map = np.zeros((self.global_map_height, self.global_map_width))
        # self.global_bev_config = {
        #     'area_extents': [[-256, 256], [-256, 256]],  # 覆盖256m x 256m区域
        #     'voxel_size': [4, 4],                       # 每个网格16m x 16m
        #     'grid_size': [128, 128]                  # 128x128网格
        # }

        # self.local_bev_config = {
        #     'area_extents': [[-32, 32], [-32, 32]],  
        #     'voxel_size': [4, 4],              
        #     'grid_size': [16, 16]                  
        # }

        pass
  

    def create_cars(self):
        self.n_vehicles = self.dataset.get_vehicle_count()
        self.cars = {}
        veh_sensors = self.dataset.get_vehicle_sensors()

        assert self.n_vehicles == len(veh_sensors), "Number of vehicles and sensors do not match"

        for vid, sensor_info in veh_sensors.items():
            car = Car(vid, dataset=self.dataset, config = self.config, detector=self.detector)
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

    def wrapper_action(self, action):
        """
        输入:
        action: np.ndarray 或 torch.Tensor, 形状 [N_v, H, W], 每辆车在自己local map上的选择，0-1矩阵
        返回:
        dict {vehicle_id: [(p, q, score), ...]}
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        N_v, H, W = action.shape
        max_k = self.config.TOP_K

        actions = {}
        for vid in range(1, N_v + 1):
            score_map = action[vid - 1]  # [H, W]
            flat = score_map.flatten()   # [H*W]
            k = max(0, min(max_k, H * W))   #  k 不超过网格大小
            if k == 0:
                actions[vid] = []
                continue
            topk_idx = np.argpartition(-flat, k)[:k]  # 前k个索引，未排序
            topk_idx_sorted = topk_idx[np.argsort(-flat[topk_idx])]  # 按值排序
            action_list = []
            for idx in topk_idx_sorted:
                row = idx // W
                col = idx % W
                score = flat[idx]   # 这里score应该是confidence value，不是score
                conf_v = self.cars[vid].local_conf_map[row, col]
                action_list.append((col, row, conf_v))
            actions[vid] = action_list

        return actions

    def step(self, action):
        '''
        输入:action： [N_v, H, W] 每辆车在自己local map上的选择，0-1矩阵
        '''
        if not self.Iou_pre_done:
            self.IoU_per_step.append(self.clusters[0].IoU)
            K_star = self.clusters[0].K_star_value
            ind = int(self.clusters[0].IoU // 0.005)
            self.tmp_IoU2K_star[ind].append(K_star)

        next_time_step = self.cur_time_step + 1

        # 这里需要封装成环境中需要的action格式
        # action： dict {vehicle_id: [(p, q, score), ...]}
        wrapped_action = self.wrapper_action(action)
        for cluster in self.clusters: # TODO，there is only one cluster now
            cluster.step(next_time_step, wrapped_action)

        if next_time_step >= self.time_step_length:
            terminated, truncated = True, False
            obs = None
            if not self.Iou_pre_done:
                LOG.info(f"tmp_IoU2K_star: {self.tmp_IoU2K_star}")
                for i in range(len(self.tmp_IoU2K_star)):
                    if len(self.tmp_IoU2K_star[i]) == 0:
                        self.IoU2K_star.append(0)
                    else:
                        self.IoU2K_star.append(np.mean(self.tmp_IoU2K_star[i]))
            self.Iou_pre_done = True
        else:
            terminated, truncated = False, False
            obs = self._get_observation(next_time_step)

        reward = self._compute_reward()
        info = {}

        LOG.info(f"Time step {self.cur_time_step}, action {wrapped_action}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}")
        self.cur_time_step = next_time_step
        # reward = torch.tensor([reward], dtype=torch.float32, device=self.RL_device)
        # terminated = torch.tensor([terminated], dtype=torch.int8, device=self.RL_device)

        # LOG.info(f"Step {self.cur_time_step}: obs={obs}, reward={reward.item()}, terminated={terminated.item()}, truncated={truncated}")

        return obs, reward, terminated, truncated, info

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

    def _get_observation(self, time_step):
        # 获取当前时间步state
        #   每个Agent，前T个时间步（包含当前时间步）的local confidence map，（single perception， N_v, T, H_l, W_l）
        #   每个Agent，前T个时间步（包含当前时间步）的last fused confidence map，（cooperative perception， actually used, N_v, T, H, W）
        #   每个Agent，前T个时间步（包含当前时间步）的position, N_v, T, 3
        #   每个区域，前T个时间步（包含当前时间步）的感兴趣的车辆数, N_a, T, H_g, W_g
        

        # for i in range(self.n_clusters):
        # OBS = {
        #     'local_maps': local_maps,
        #     'fused_maps': fused_maps,
        #     'cur_fused_maps': cur_fused_maps,
        #     'positions': positions,
        #     'fused_bev_confidence': fused_bev_confidence,
        #     'interest_maps': interest_maps
        # }
        obs = self.clusters[0].get_state(time_step)

        # obs = {
        #     'local_map': np.array(local_maps),          # [N_c, T, H_l, W_l]
        #     'fused_map': np.array(fused_maps),          # [N_c, T, H, W]
        #     'positions': np.array(positions),           # [N_c, T, 2]
        #     'interest_map': np.array(interest_maps)     # [T, H, W]
        # }  # 可替换为 self._combine_states(cluster_states, object_states)

        self.statistic.setdefault("local_map", []).append(obs["local_maps"])

        return obs
    
    def wrapper_state(self, obs, B: int = 1):
        """
        返回:
        local_maps: [N_v, T, H, W] (float32)
        fused_maps: [N_v, T, H, W] (float32)
        adjacency_matrix: [T, N_v, N_v] (float32)
        positions: [N_v, T, 2]   (float32)
        interest_maps: [T, H_g, W_g]  (float32)
        """
        T = 1
        N_v = self.n_vehicles
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
        
    
    def score_map2action(self, score_map, local_maps, max_k):
        """
        输入:
        score_map: np.ndarray 或 torch.Tensor, 形状 [P, Q]
        返回:
        list of (p, q, score)
        """
        B, P, Q = score_map.shape
        k = max(0, min(max_k, P * Q))   #  k 不超过网格大小
        if k == 0:
            return {} if B == 1 else [{} for _ in range(B)]

        # ---- 3) 用 torch.topk 在 patch 网格上取 Top-K（每个 batch 各取 k 个）----
        flat = score_map.reshape(B, -1)                           # [B, P*Q]
        topk_vals, topk_idx = torch.topk(flat, k=k, dim=1, largest=True, sorted=True)  # [B,k], [B,k]
        topk_rows = topk_idx // Q                                  # [B,k]
        topk_cols = topk_idx % Q                                   # [B,k]

        low_maps = local_maps.squeeze(2)  # [B,N,H,W]

        #print(f"score_map shape: {score_map.shape}, topk_vals shape: {topk_vals.shape}, topk_idx shape: {topk_idx.shape}")
        #print(f"topk_rows: {topk_rows}, topk_cols: {topk_cols}")
        #print(f"P: {P}, Q: {Q}, k: {k}")

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
                assert local_maps[b, picked_j, 0, p, q] == self.cars[vid].local_conf_map[p, q], \
                    f"Mismatch at batch {b}, vehicle {vid}: {local_maps[b, picked_j, 0, p, q]} vs {self.cars[vid].local_conf_map[p, q]} at time step {self.cur_time_step} "

                actions.setdefault(vid, []).append((p, q, per_vehicle_vals[picked_j].item())) 
            actions_batch.append(actions)
        

        return actions_batch
    
    def action_select(self, states, max_k, collection_policy=False):
        """
        输入:
        states: dict 包含:
            local_maps: np.ndarray 或 torch.Tensor, 形状 [N, T, H, W]
            fused_maps: np.ndarray 或 torch.Tensor, 形状 [N, T, H, W]
            positions: np.ndarray 或 torch.Tensor, 形状 [N, T, 2]
            interest_maps: np.ndarray 或 torch.Tensor, 形状 [T, H_g, W_g]
            adjs
        返回:
        若 B==1: dict {vehicle_id: [(p, q, score), ...]}
        若 B>1 : list[dict], 每个 batch 一个 dict
        """
        device = self.RL_device
        local_maps, fused_maps, position, interested_maps, adj = states_tuple
        # ---- 1) 转成 torch.Tensor 并放到正确设备 ----
        # feature = torch.as_tensor(state, dtype=torch.float32, device=device)          # [B,N,1,H,W]
        # adjacency = torch.as_tensor(adjacency_matrix, dtype=torch.float32, device=device)  # [B,N,N]

        # ---- 2) 前向网络，采样 scores_map（[B,P,Q]）----
        # out = self.RL_model(local_maps, adjs, fused_maps, prev_b, curr_b)
        # alpha, beta, value = out["alpha"], out["beta"], out["value"]  # [B,P,Q], [B]
        # dist = Beta(alpha, beta)
        # scores_map = dist.rsample().clamp(1e-6, 1-1e-6)               # [B,P,Q]
        # logp = dist.log_prob(scores_map).sum(dim=(1, 2))  
        N, T, H, W = local_maps.shape
        if self.config.strategy == 'RL' :       
            scores_map, logp, value = self.RL_agent.action_select(states_tuple)
        else:
            scores_map, logp, value = torch.zeros((N, H, W)), torch.zeros((N, H, W)), torch.zeros((N, H, W))
        _, P, Q = scores_map.shape
        randv = np.random.rand()
        if collection_policy:
            if randv < 0.7:
                collect = "Greedy"
            elif randv < 0.85:
                collect = "Random"
            else:
                collect = "RL"
        
        
        k = max_k // N # max(0, min(max_k, P * Q)) / N   #  k 不超过网格大小
        actions = {}
        # ---- 5) 对每个被选中的 patch (p,q)，在 N 辆车里选最大值的那辆 ----
        actions_batch = []
        if self.config.strategy == 'RL' or (collection_policy and collect == "RL"):
            actions_batch = self.score_map2action(scores_map, local_maps, k)
        else:
            for i in range(N):
                if self.config.strategy == "Random" or (collection_policy and collect == "Random"):
                    for m in range(k):
                        vid = i + 1
                        p = np.random.randint(0, H)
                        q = np.random.randint(0, W)
                        actions.setdefault(vid, []).append((p, q, self.cars[vid].local_conf_map[p, q]))
                        if collection_policy:
                            LOG.info(f"Random selected actions: {actions}")
                            scores_map = local_maps.mean(dim=1, keepdim=True).squeeze(2).squeeze(1)  # [B, N, 1, H, W] -> [B, 1, H, W]
                elif self.config.strategy == 'Single':
                    for m in range(N):
                        vid = m + 1
                        actions.setdefault(vid, [])
                elif self.config.strategy == "Greedy" or (collection_policy and collect == "Greedy"):
                    avg_cnt = k // N
                    for vid in range(1, N+1):
                        cnt = avg_cnt
                        if vid == N:
                            cnt = k - avg_cnt * (N-1)
                        values, p, q = topk_2d(self.cars[vid].local_conf_map, cnt)
                        for tmp in range(len(values)):
                            actions.setdefault(vid, []).append((p[tmp], q[tmp], self.cars[vid].local_conf_map[p[tmp], q[tmp]]))
                    if collection_policy:
                        LOG.info(f"Greedy selected actions: {actions}")
                        scores_map = local_maps.max(dim=1, keepdim=True).squeeze(2).squeeze(1)  # [B, N, 1, H, W] -> [B, 1, H, W]
                elif self.config.strategy == "Detection":
                    pass
                else:
                    raise ValueError(f"unimlemented strategy {self.config.strategy}")


        return scores_map, actions, logp, value


    def _compute_reward(self):
        rewards = []
        for cluster in self.clusters:
            rewards.append(cluster.compute_reward(self.cur_time_step))
        return rewards[0]
