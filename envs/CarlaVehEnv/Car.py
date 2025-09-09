from envs.CarlaVehEnv.Sensor import Lidar, Camera
from envs.CarlaVehEnv.CarlaDataCollector import V2XSimReader
from envs.CarlaVehEnv.Visualize import visualize_step
from envs.CarlaVehEnv.utils import *
from envs.CarlaVehEnv.Comm_Comp_settings import Comm_Comp_Base
from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from nuscenes.utils.geometry_utils import transform_matrix
from v2x_sim_visualizer import render_sample_data, render_scene_lidar
from pyquaternion import Quaternion
from functools import reduce
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import logging
LOG = logging.getLogger(__name__)

class Car():
    def __init__(self, vid, dataset, V2X_dataset, config, det_model, carla_vehicle = None):
        self.carla_vehicle = carla_vehicle
        self.config = config
        self.visualize = config.visualize          # 可视化的数据
        self.dataset : V2XSimReader = dataset
        self.v2x_det_dataset : V2XSimDet = V2X_dataset  # V2XSimDet dataset for detection
        self.cluster_id = -1
        self.vid = vid
        self.results_path = os.path.join(self.config.result_path, f"vehicle_{self.vid:02d}")
        self.setup()
        self.sensor_channel = "LIDAR_TOP"
        self.strategy = config.strategy

        self.V2X_config = Config("train", binary=True, only_det=True)
        self.V2X_config_global = ConfigGlobal("train", binary=True, only_det=True)
        self.det_model = det_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_slices_cnt = 0
        self.current_slices_limits = 0
        
        # define the map
        self.local_conf_map = np.zeros((self.dataset.map_dims[0], self.dataset.map_dims[1]), dtype=np.float32)
        self.cur_fused_conf_map = np.zeros((self.dataset.map_dims[0], self.dataset.map_dims[1]), dtype=np.float32)
        self.last_fused_conf_map = np.zeros((self.dataset.map_dims[0], self.dataset.map_dims[1]), dtype=np.float32)
        self.RoI_area_extents = self.V2X_config.area_extents

        self.comm_comp_model = Comm_Comp_Base(self.config)


    def setup(self):
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)


    def apply_control(self, time_step, actions, area_cnt):
        self.cur_fused_conf_map[...] = self.last_fused_conf_map  # 原地覆盖
        np.maximum(self.cur_fused_conf_map, self.local_conf_map, out=self.cur_fused_conf_map)

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
        
    def update_metadata(self, time_step):
        # Update the metadata of the vehicle
        metadata = self.dataset.get_vehicle_metadata(self.vid)
        if time_step == 0:
            time_step += 1
        self.speed = metadata[time_step]['velocity']
        self.position = metadata[time_step]['position']
        self.rotation = metadata[time_step]['rotation']

        return metadata[time_step]

    @torch.no_grad()
    def get_state(self, time_step):
        """
        返回该车在给定time_step的观测与检测结果：
        - objects_local: (N,4,2) 本地BEV四角点
        - objects_world: (N,1,4,2) 世界系四角点（若可获得位姿/变换矩阵）
        - scores: (N,) 置信度
        """
        self.local_conf_map = np.zeros((self.dataset.map_dims[0], self.dataset.map_dims[1]), dtype=np.float32)
        state = {'vid': self.vid, 'time_step': time_step}

        # self.last_slices_cnt = self.current_slices_limits
        # self.current_slices_limits = theory_K_star #self.comm_comp_model.compute_k_star(self.config.num_vehicles, self.dataset.map_dims[0] * self.dataset.map_dims[1], lamba=IoU)

        objects_local, scores, trans_matrix = self.run_detection(time_step, det_method="points count", visualize = self.visualize)  # 使用点云数量检测

        device = self.device
        if isinstance(trans_matrix, torch.Tensor):
            trans_mats = trans_matrix[None, None].to(device) if trans_matrix.ndim == 2 else trans_matrix.to(device)
        else:
            T = np.asarray(trans_matrix, dtype=np.float32)
            T = torch.from_numpy(T).to(device)
            trans_mats = T[None, None] if T.ndim == 2 else T

        # 尝试变换到世界系（两条路：优先NuScenes位姿，退路用trans_matrix）
        objects_world = None

        # # 4.1 优先：NuScenes位姿（如果你类里有 self.dataset / self.nusc 可获得 sample_data 的 tokens）
        # try:
        #     sensors_data = self.dataset.get_agent_sensor_files(self.vid, time_step)
        #     lidar_sd = None
        #     for _, sd in sensors_data.items():
        #         if sd['channel'].split('_id_')[0] == self.sensor_channel:
        #             lidar_sd = sd
        #             break
        #     if lidar_sd and 'ego_pose_token' in lidar_sd and 'calibrated_sensor_token' in lidar_sd:
        #         pose = self.v2x_sim.get("ego_pose", lidar_sd['ego_pose_token'])
        #         cs   = self.v2x_sim.get("calibrated_sensor", lidar_sd['calibrated_sensor_token'])
        #         T_world_ego  = transform_matrix(pose['translation'], Quaternion(pose['rotation']))
        #         T_ego_sensor = transform_matrix(cs['translation'],   Quaternion(cs['rotation']))
        #         T_world_sensor = T_world_ego @ T_ego_sensor
        #         objects_world = bev_local_to_world_corners(T_world_sensor, objects_local)
        # except Exception:
        #     pass

        # 4.2 退路：用 trans_matrix（很多版本就是本地->世界的4x4）
        if objects_world is None:
            T = trans_mats.detach().cpu().numpy()
            objects_world = bev_local_to_world_corners(T, objects_local)

        # 5) 打包返回
        # 统一成 numpy
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        state.update({
            'objects_local': objects_local,         # (N,4,2) 本地BEV
            'objects_world': objects_world,         # (N,1,4,2) 世界系
            'scores': scores,                       # (N,)
        })

        self.dataset.boxes_to_conf_map(self.local_conf_map, objects_world, scores)
        state.update({'local_map': self.local_conf_map,
                      'cur_fused_map': self.cur_fused_conf_map, # for evaluation
                      'last_fused_map': self.last_fused_conf_map,
                      'last_slices_cnt': self.last_slices_cnt,
                      'current_slices_limits': self.current_slices_limits})

        LOG.info(f"Car {self.vid} at time step {time_step} has local conf_map: {self.local_conf_map.shape},  nonzero {np.count_nonzero(self.local_conf_map)}, min/max: {float(self.local_conf_map.min()), float(self.local_conf_map.max())}")
        # LOG.info(f"Car {self.vid} at time step {time_step} has state: {state}") 

        return state

    
    def join_group(self, cluster_id):
        # Join a group of vehicles
        self.cluster_id = cluster_id

    def leave_group(self):
        # Leave the current group of vehicles
        self.cluster_id = None

    @torch.no_grad()
    def run_detection(self, time_step, det_method="lowerbound", visualize = 0):
        """
        对单个 agent 的点云 BEV 输入进行目标检测，输出 scores & boxes
        参考 test_codet.py 的推理流程
        """    
        (padded_voxel_points, padded_voxel_points_teacher_det, label_one_hot, reg_target,
        reg_loss_mask, anchors_map, vis_maps, gt_max_iou, filename,
        target_agent_id, num_sensor, trans_matrix) = self.v2x_det_dataset[time_step][self.vid-1]
        
        if visualize:
            # 可视化点云（单车）
            sample  = self.dataset.get_sample(time_step)
            channel = self.sensor_channel + f"_id_{self.vid}"
            sample_data_token = sample['data'][channel]
            render_sample_data(self.dataset.v2x_sim, sample_data_token, with_anns = True, underlay_map = False,  pointsensor_channel=channel, axes_limit=32, \
                               out_path=os.path.join(self.results_path, f"lidar_t{time_step:03d}.png"))

            plt.imshow(np.max(padded_voxel_points.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
            img_file = os.path.join(self.results_path, f"voxel_map_t{time_step:03d}.png")
            plt.savefig(img_file)

        # 构造 data，与 test_codet.py 保持一致
        # data = {
        #     "bev_seq": padded_voxel_points.unsqueeze(0).to(device),
        #     "labels": torch.zeros_like(reg_target).unsqueeze(0).to(device),  # dummy
        #     "reg_targets": reg_target.unsqueeze(0).to(device),
        #     "anchors": anchors_map.unsqueeze(0).to(device),
        #     "vis_maps": vis_maps.unsqueeze(0).to(device),
        #     "reg_loss_mask": reg_loss_mask.unsqueeze(0).to(device).type(dtype=torch.bool),
        #     "target_agent_ids": torch.tensor([[target_agent_id]]).to(device),
        #     "num_agent": torch.tensor([[num_agent]]).to(device),
        #     "trans_matrices": trans_matrix.unsqueeze(0).to(device),
        # }

        # # 调用推理
        # if flag == "lowerbound_box_com":
        #     loss, cls_loss, loc_loss, result = fafmodule.predict_all_with_box_com(
        #         data, data["trans_matrices"], validation=False
        #     )
        # else:
        #     result = fafmodule.predict_all(
        #         data, 1, validation=False, num_agent = 1
        #     )

        # LOG.info(f"Detection result for agent {target_agent_id}: {result}")
        # # 解析结果
        # scores, boxes = [], []

        # if isinstance(result, list) and len(result) > 0:
        #     # result[0] 是一个 tuple (detections, some_tensor)
        #     detections, _ = result[0]

        #     if isinstance(detections, list) and len(detections) > 0:
        #         agent_detections = detections[0]  # [[{...}]]
        #         if len(agent_detections) > 0:
        #             pred_dict = agent_detections[0]  # {'pred': ..., 'score': ..., ...}

        #             if "score" in pred_dict and "pred" in pred_dict:
        #                 scores = pred_dict["score"]
        #                 if torch.is_tensor(scores):
        #                     scores = scores.detach().cpu().numpy()
        #                 boxes = pred_dict["pred"]
        #                 if torch.is_tensor(boxes):
        #                     boxes = boxes.detach().cpu().numpy()

        if det_method == "random":
            objects_local = []  # 模拟数据
            for i in range(100):
                objects_local.append(np.random.rand(4, 2) * i)  # 100个检测框的四个角点，范围在0-30米之间
            objects_local = np.array(objects_local, dtype=np.float32)
            scores = np.random.rand(100)  # 模拟数据
        elif det_method == "points count":
            # 基于点云数量的简单检测
            conf_map = voxel_to_confidence_map(
                padded_voxel_points,
                take_last_t=True,      # 你的shape是(1,256,256,13)，取最后一帧
                z_reduce="sum",        # Z上求和，等价“点数作为confidence”
                smooth_sigma=1.0,      # 轻微平滑（可调 0/1/1.5）
                norm_mode="percentile",
                p_low=1.0, p_high=99.0
            )  # (H,W), float32 in [0,1]

            # 2) 从 heatmap 里提取候选框（也可以只用 heatmap，不出框）
            objects_local, scores = heatmap_to_boxes(
                conf_map,
                thresh="percentile",   # 用分位数阈值
                thr_percentile=97.0,   # 越高越少框，可调
                min_pixels=12
            )

            # 3) 可视化（保存热力图和阈值mask）
            if visualize and getattr(self, "results_path", None):
                # 置信度热力图
                plt.figure()
                plt.imshow(conf_map, origin="upper")
                plt.title("Confidence (Point Count → Z-sum)")
                img_file = os.path.join(self.results_path, f"conf_heatmap_t{time_step:03d}.png")
                plt.savefig(img_file, bbox_inches="tight")
                plt.close()

                # 可选：叠加矩形框
                if objects_local.shape[0] > 0:
                    plt.figure()
                    plt.imshow(conf_map, origin="upper")
                    for k in range(objects_local.shape[0]):
                        xs = [objects_local[k,i,0] for i in range(4)] + [objects_local[k,0,0]]
                        ys = [objects_local[k,i,1] for i in range(4)] + [objects_local[k,0,1]]
                        plt.plot(xs, ys, linewidth=1.5)
                    plt.title(f"Detections (N={objects_local.shape[0]})")
                    img_file = os.path.join(self.results_path, f"conf_boxes_t{time_step:03d}.png")
                    plt.savefig(img_file, bbox_inches="tight")
                    plt.close()

            if len(objects_local) == 0:
                objects_local = np.zeros((0, 4, 2), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
            else:
                objects_local = np.stack(objects_local, axis=0)  # (N, 4, 2)
                scores = np.array(scores, dtype=np.float32)

        elif det_method == "coperception":
            # 2) 组装 predict_all 所需的 data 字典（单车）
            device = self.device
            if isinstance(trans_matrix, torch.Tensor):
                trans_mats = trans_matrix[None, None].to(device) if trans_matrix.ndim == 2 else trans_matrix.to(device)
            else:
                T = np.asarray(trans_matrix, dtype=np.float32)
                T = torch.from_numpy(T).to(device)
                trans_mats = T[None, None] if T.ndim == 2 else T

            self.det_model.model.eval()
            data = {
                "bev_seq": torch.tensor(padded_voxel_points[None, ...], dtype=float).to(device),# (1,T,C,H,W)
                "labels": torch.tensor(label_one_hot[None, ...], dtype=float).to(device),
                "reg_targets": torch.tensor(reg_target[None, ...], dtype=float).to(device),
                "reg_loss_mask": torch.tensor(reg_loss_mask[None, ...]).to(device).bool(),
                "anchors": torch.tensor(anchors_map[None, ...], dtype=float).to(device),
                "vis_maps": torch.empty(0, device=device),
                "target_agent_ids": torch.tensor([[0]], device=device),
                "num_agent": torch.tensor([[1]], device=device),   # 单车
                "trans_matrices": trans_mats,
            }

            # 3) 推理（FaFModule）
            seq_results = self.det_model.predict_all(data, batch_size=1, validation=False, num_agent=1)
            # # 返回是 list（按agent），单车取第0个
            # if not seq_results or len(seq_results[0]) == 0:
            #     state.update({
            #         'objects_local': np.zeros((0,4,2), dtype=np.float32),
            #         'objects_world': np.zeros((0,1,4,2), dtype=np.float32),
            #         'scores': np.zeros((0,), dtype=np.float32),
            #         'filename': filename
            #     })
            #     return state
            
            LOG.info(f"seq_results: {seq_results}")

            class_selected = seq_results[0][0][0][0]     # dict: {'pred':(N,4,2), 'score':(N,), ...}
            # objects_local  = class_selected['pred']    # (N,4,2) numpy
            # scores         = class_selected['score']   # (N,)   torch或numpy
        else:
            raise ValueError(f"Unsupported detection method: {det_method}")



        return objects_local, scores, trans_matrix

    
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
        LOG.info(f"pre_IoU_step : {pre_IoU_step}")
        LOG.info(f"pre_IoU2K_star : {pre_IoU2K_star}")
        LOG.info(f"pre_Iou : {pre_Iou}")

        LOG.debug(f"members: f{self.members}")

    # def add_member(self, member):
    #     self.members.append(member)

    # def remove_member(self, member):
    #     if member in self.members:
    #         self.members.remove(member)
    #     else:
    #         raise ValueError("Member not found in the cluster")
    
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
        # Return the state of the cluster
        cluster_state = {}
        local_maps = []

        for vid, member in self.members.items():
            member_state = member.get_state(time_step)
            cluster_state[member.vid] = member_state
            local_maps.append(member_state["local_map"])

        N = self.config.num_vehicles
        IoU = 0
        theory_K_star = 0
        if not self.pre_Iou:
            mean_overlap = 0
            cnt = 0
            
            for i in range(N):
                for j in range(i + 1, N):
                    do_overlap, overlap_degree = self.overlap(cluster_state[i + 1], cluster_state[j + 1])
                    mean_overlap += overlap_degree
                    cnt += 1
            mean_overlap = mean_overlap / max(cnt, 1)
            
            IoU = self.compute_IoU(mean_overlap)
            theory_K_star, info = self.comm_comp_model.compute_k_star(self.config.num_vehicles, self.dataset.map_dims[0] * self.dataset.map_dims[1], IoU)
        else:
            IoU = self.pre_IoU_step[time_step]
            ind = int(IoU // 0.005)
            theory_K_star = self.pre_IoU2K_star[ind]

        self.IoU = IoU
        self.K_star_value = theory_K_star
        self.comm_comp_model._print(self.members[1].local_conf_map.shape[0] * self.members[1].local_conf_map.shape[1], theory_K_star)
        LOG.info(f"time step {time_step}, IoU {IoU}, theory K star {theory_K_star}")
        for i in range(N):
            self.members[i+1].last_slices_cnt = self.members[i+1].current_slices_limits
            self.members[i+1].current_slices_limits = theory_K_star
            cluster_state[i+1].update({
                'last_slices_cnt': self.members[i+1].current_slices_limits,
                'current_slices_limits': theory_K_star})

        if self.config.visualize:
            token_scene_no = 'scene_5'
            my_scene_token = self.dataset.v2x_sim.field2token('scene', 'name', token_scene_no)[0]
            render_scene_lidar(self.dataset.v2x_sim, my_scene_token, axes_limit=96, single_frame_idx=time_step, out_path=os.path.join(self.config.result_path, f"fused_scene"))
            overlay_confidence_maps(local_maps, out_path= os.path.join(self.config.result_path, "fused_scene", f"conf_map_{time_step}.png"))


        states = cluster_state
        # states = GNN(cluster_state)     # TODO
        # LOG.info(f"Cluster {self.cluster_id} at time step {time_step} has states: {states}")
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
            member.apply_control(time_step, actions, area_cnt)
    
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


    

    



    


        
 