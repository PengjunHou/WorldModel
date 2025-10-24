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
from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from controller.PPO import ModelConfig, ActorCritic, PPO
from controller.DPPO_agent import DPPO
from controller.model.diffusion_ppo import PPODiffusion
from controller.cfg.dppo_cfg import DPPOConfig
from torch.utils.tensorboard import SummaryWriter
import matplotlib.animation as animation

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
        self.visualize = config.visualize

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
        # statistic data
        self.statistic = {}

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
        self.global_map_height = self.dataset.map_dims[0]
        self.global_map_width = self.dataset.map_dims[1]
        self.global_conf_map = np.zeros((self.global_map_height, self.global_map_width))

    def create_cars(self):
        self.n_vehicles = self.dataset.get_vehicle_count()
        self.cars = {}
        veh_sensors = self.dataset.get_vehicle_sensors()

        assert self.n_vehicles == len(veh_sensors), "Number of vehicles and sensors do not match"

        for vid, sensor_info in veh_sensors.items():
            car = Car(vid, dataset=self.dataset, V2X_dataset = self.v2x_det_dataset, config = self.config,  det_model = self.det_model)
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
        if not self.Iou_pre_done:
            self.IoU_per_step.append(self.clusters[0].IoU)
            K_star = self.clusters[0].K_star_value
            ind = int(self.clusters[0].IoU // 0.005)
            self.tmp_IoU2K_star[ind].append(K_star)

        next_time_step = self.cur_time_step + 1

        for cluster in self.clusters: # TODO，there is only one cluster now
            cluster.step(next_time_step, action)

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

        LOG.info(f"Time step {self.cur_time_step}, action {action}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}")
        self.cur_time_step = next_time_step
        reward = torch.tensor([reward], dtype=torch.float32, device=self.RL_device)
        terminated = torch.tensor([terminated], dtype=torch.int8, device=self.RL_device)

        LOG.info(f"shapes -  reward: {reward.shape}, terminated: {terminated.shape}")
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
        # 获取当前时间步每个Agent的local confidence map
        # 将其映射到Global map上
        cluster_obs = []
        # object_states = []
        # for oid, obj in self.objects.items():
        #     object_states.append(obj.get_state(time_step))
        for i in range(self.n_clusters):
            cluster_obs.append(self.clusters[i].get_state(time_step))

        obs = cluster_obs  # 可替换为 self._combine_states(cluster_states, object_states)

        cur_time_local_maps = []
        for veh_i in range(len(self.clusters[0].members)):
            cur_time_local_maps.append(np.copy(self.clusters[0].members[veh_i + 1].local_conf_map))
        self.statistic.setdefault("local_map", []).append(cur_time_local_maps)

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
            N = self.n_vehicles
            sample = next(iter(single_obs.values()))
            H, W = sample["local_map"].shape

            maps = []
            fused_map = []
            last_slices_cnt = []
            cur_slice_cnt = []
            adjacency_matrix = np.zeros((N, N), dtype=np.float32)

            for i in range(N):
                agent_obs = single_obs[i + 1]  # vid 从 1 开始
                local_map = agent_obs["local_map"]              # [H, W]
                last_fused_map = agent_obs["last_fused_map"]    # [H, W]
                prev_cnt = agent_obs["last_slices_cnt"]         # 标量/array
                curr_cnt = agent_obs["current_slices_limits"]   # 标量/array

                # 可选标准化：
                # local_map = (local_map - local_map.mean()) / (local_map.std() + 1e-6)

                maps.append(np.expand_dims(local_map.astype(np.float32), axis=0))         # -> [1,H,W]
                fused_map.append(np.expand_dims(last_fused_map.astype(np.float32), axis=0))

                # 确保 budgets 变成 (1,) 再堆叠为 [N]
                prev_cnt = np.asarray(prev_cnt, dtype=np.float32).reshape(1,)
                curr_cnt = np.asarray(curr_cnt, dtype=np.float32).reshape(1,)
                last_slices_cnt.append(prev_cnt)   # (1,)
                cur_slice_cnt.append(curr_cnt)     # (1,)

            maps_np = np.stack(maps, axis=0).astype(np.float32)         # [N,1,H,W]
            fused_maps_np = np.stack(fused_map, axis=0).astype(np.float32)  # [N,1,H,W]
            prev_budget = np.concatenate(last_slices_cnt, axis=0).astype(np.float32)  # [N]
            cur_budget  = np.concatenate(cur_slice_cnt,  axis=0).astype(np.float32)   # [N]

            # 构建邻接
            for i in range(N):
                for j in range(i + 1, N):
                    do_overlap, overlap_degree = self.overlap(single_obs[i + 1], single_obs[j + 1])
                    if do_overlap:
                        adjacency_matrix[i, j] = adjacency_matrix[j, i] = overlap_degree

            LOG.info(
                f"shape: maps_np {maps_np.shape}, adjacency_matrix {adjacency_matrix.shape}, "
                f"fused_maps_np {fused_maps_np.shape}, prev_budget {prev_budget.shape}, "
                f"cur_budget {cur_budget.shape}"
            )

            # 返回:
            # [N,1,H,W], [N,N], [N,1,H,W], [N], [N]
            return maps_np, adjacency_matrix, fused_maps_np, prev_budget, cur_budget

        if isinstance(obs, (list, tuple)):
            # 多环境：长度就是 B
            batch_maps = []
            batch_adj = []
            batch_fused_maps = []
            batch_prev_budget = []
            batch_cur_budget = []

            for single_obs in obs:
                maps_np, adj_np, fused_maps_np, prev_budget, cur_budget = _process_single(single_obs)
                batch_maps.append(maps_np[None, ...])          # [1,N,1,H,W]
                batch_adj.append(adj_np[None, ...])            # [1,N,N]
                batch_fused_maps.append(fused_maps_np[None, ...])  # [1,N,1,H,W]
                batch_prev_budget.append(prev_budget[None, ...])   # [1,N]
                batch_cur_budget.append(cur_budget[None, ...])     # [1,N]  <- 修正

            maps_b = np.concatenate(batch_maps, axis=0).astype(np.float32)        # [B,N,1,H,W]
            adj_b = np.concatenate(batch_adj, axis=0).astype(np.float32)          # [B,N,N]
            fused_b = np.concatenate(batch_fused_maps, axis=0).astype(np.float32) # [B,N,1,H,W]
            prev_budget_b = np.concatenate(batch_prev_budget, axis=0).astype(np.float32)  # [B,N]
            cur_budget_b  = np.concatenate(batch_cur_budget,  axis=0).astype(np.float32)  # [B,N]

            # 转 torch
            device = self.RL_device
            maps_b_t = torch.tensor(maps_b, dtype=torch.float32, device=device)
            adj_b_t = torch.tensor(adj_b, dtype=torch.float32, device=device)
            fused_b_t = torch.tensor(fused_b, dtype=torch.float32, device=device)
            prev_budget_b_t = torch.tensor(prev_budget_b, dtype=torch.float32, device=device)
            cur_budget_b_t  = torch.tensor(cur_budget_b,  dtype=torch.float32, device=device)

            return maps_b_t, adj_b_t, fused_b_t, prev_budget_b_t, cur_budget_b_t

        else:
            # 单环境：先做一份，再广播到 B
            maps_np, adj_np, fused_maps_np, prev_budget, cur_budget = _process_single(obs)
            maps_b = np.repeat(maps_np[None, ...], B, axis=0).astype(np.float32)         # [B,N,1,H,W]
            adj_b  = np.repeat(adj_np[None,  ...], B, axis=0).astype(np.float32)         # [B,N,N]
            fused_b = np.repeat(fused_maps_np[None, ...], B, axis=0).astype(np.float32)  # [B,N,1,H,W]
            prev_budget_b = np.repeat(prev_budget[None, ...], B, axis=0).astype(np.float32)  # [B,N]
            cur_budget_b  = np.repeat(cur_budget[None,  ...], B, axis=0).astype(np.float32)  # [B,N]

            device = self.RL_device
            maps_b_t = torch.tensor(maps_b, dtype=torch.float32, device=device)
            adj_b_t = torch.tensor(adj_b, dtype=torch.float32, device=device)
            fused_b_t = torch.tensor(fused_b, dtype=torch.float32, device=device)
            prev_budget_b_t = torch.tensor(prev_budget_b, dtype=torch.float32, device=device)
            cur_budget_b_t  = torch.tensor(cur_budget_b,  dtype=torch.float32, device=device)

            return maps_b_t, adj_b_t, fused_b_t, prev_budget_b_t, cur_budget_b_t
        
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
    
    def action_select(self, states_tuple, max_k, collection_policy=False):
        """
        输入:
        state: np.ndarray 或 torch.Tensor, 形状 [B, N, 1, H, W]
        adjacency_matrix: np.ndarray 或 torch.Tensor, 形状 [B, N, N]
        返回:
        若 B==1: dict {vehicle_id: [(p, q, score), ...]}
        若 B>1 : list[dict], 每个 batch 一个 dict
        """
        device = self.RL_device
        local_maps, adjs, fused_maps, prev_b, curr_b = states_tuple
        # ---- 1) 转成 torch.Tensor 并放到正确设备 ----
        # feature = torch.as_tensor(state, dtype=torch.float32, device=device)          # [B,N,1,H,W]
        # adjacency = torch.as_tensor(adjacency_matrix, dtype=torch.float32, device=device)  # [B,N,N]

        # ---- 2) 前向网络，采样 scores_map（[B,P,Q]）----
        # out = self.RL_model(local_maps, adjs, fused_maps, prev_b, curr_b)
        # alpha, beta, value = out["alpha"], out["beta"], out["value"]  # [B,P,Q], [B]
        # dist = Beta(alpha, beta)
        # scores_map = dist.rsample().clamp(1e-6, 1-1e-6)               # [B,P,Q]
        # logp = dist.log_prob(scores_map).sum(dim=(1, 2))  
        B, N, C, H, W = local_maps.shape
        if self.config.strategy == 'RL' :       
            scores_map, logp, value = self.RL_agent.action_select(states_tuple)
        else:
            scores_map, logp, value = torch.zeros((1, H, W)), torch.zeros((1, H, W)), torch.zeros((1, H, W))
        _, P, Q = scores_map.shape
        randv = np.random.rand()
        if collection_policy:
            if randv < 0.7:
                collect = "Greedy"
            elif randv < 0.85:
                collect = "Random"
            else:
                collect = "RL"
        
        
        k = max(0, min(max_k, P * Q))   #  k 不超过网格大小
        if k == 0:
            return {} if B == 1 else [{} for _ in range(B)]
        LOG.info(f"Batch size: {B}, Vehicles: {N}, Channels: {C}, Height: {H}, Width: {W}, Patches: {P}, QPatches: {Q}, k: {k}")

        # ---- 5) 对每个被选中的 patch (p,q)，在 N 辆车里选最大值的那辆 ----
        actions_batch = []
        if self.config.strategy == 'RL' or (collection_policy and collect == "RL"):
            actions_batch = self.score_map2action(scores_map, local_maps, k)
        else:
            for b in range(B):
                actions = {}

                if self.config.strategy == "Random" or (collection_policy and collect == "Random"):
                    for m in range(k):
                        vid = np.random.randint(1, N+1)
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
