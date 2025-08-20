from envs.CarlaVehEnv.Sensor import Lidar, Camera
from envs.CarlaVehEnv.CarlaDataCollector import V2XSimReader
from envs.CarlaVehEnv.Visualize import visualize_step
from envs.CarlaVehEnv.utils import *
from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from nuscenes.utils.geometry_utils import transform_matrix
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
    def __init__(self, vid, dataset, V2X_dataset, config, det_model, sensors = None, carla_vehicle = None, visualize = False):
        self.carla_vehicle = carla_vehicle
        self.config = config
        self.visualize = visualize          # 可视化的数据
        self.dataset : V2XSimReader = dataset
        self.v2x_det_dataset : V2XSimDet = V2X_dataset  # V2XSimDet dataset for detection
        self.cluster_id = -1
        self.vid = vid
        self.sensor_types = sensors 
        self.sensors = {}               # a dict of sensor objects (sensor_id: sensor)
        self.setup()
        self.sensor_channel = "LIDAR_TOP"
        self.upload_data = None         # 根据action决定上传的数据，可能是metadata + sensor data 后的数据
        self.rev_data = None            # 接收的数据

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

        self.time_slice_unit = self.config.time_slice_unit
        self.summary_slices = self.config.summary_slices
        self.feature_slices = self.config.feature_slices
        self.process_slices = self.config.process_slices
        self.step_total_slices = self.dataset.frequency / self.time_slice_unit



    
    def setup(self):
        sensor_id = 0
        for type, sensors in self.sensor_types.items():
            for type_id in range(sensors['count']):
                self.sensors[sensor_id] = {"type": type, "type_id": type_id, "token":sensors['sensors'][type_id]}
                sensor_id += 1
        if self.visualize:
            import matplotlib.pyplot as plt

            # 创建并复用窗口
            self.fig, self.axs = plt.subplots(4, 6, figsize=(15, 10))

    def apply_control(self, time_step, actions, area_cnt):
        times_slice = 2 * self.summary_slices + self.feature_slices * area_cnt + self.process_slices
        self.cur_fused_conf_map = deepcopy(self.last_fused_conf_map)

        for vid, action in actions.items():
            if vid != self.vid:
                for index_x, index_y, score in action:
                    # assert self.local_conf_map[index_x, index_y] <= score, f"Score {score} is less than current value {self.local_conf_map[index_x, index_y]} at ({index_x}, {index_y})"
                    # fuse to fused map
                    if self.cur_fused_conf_map[index_x, index_y] < score:
                        self.cur_fused_conf_map[index_x, index_y] = score
        
        # apply delay
        self.cur_fused_conf_map = self.cur_fused_conf_map * np.exp(0 - times_slice * self.time_slice_unit)
        self.last_fused_conf_map = deepcopy(self.cur_fused_conf_map)
        self.last_fused_conf_map = self.last_fused_conf_map * np.exp( - self.dataset.frequency)
        

        # if self.visualize:
        #     imgs = []
        #     for sensor_id, sensor_data in self.sensors_data.items():
        #         if sensor_data is not None:
        #             img, format = sensor_data['filename'], sensor_data['fileformat']
        #             imgs.append(img)
        #         else:
        #             imgs.append(None)

        #         visualize_step(time_step, imgs, self.dataset.root, self.fig, self.axs)

    def update_metadata(self, time_step):
        # Update the metadata of the vehicle
        metadata = self.dataset.get_vehicle_metadata(self.vid)
        if time_step == 0:
            time_step += 1
        self.speed = metadata[time_step]['velocity']
        self.position = metadata[time_step]['position']
        self.rotation = metadata[time_step]['rotation']

        return metadata[time_step]
    
    def get_sensor_data(self, sensor_id, time_step):
        # Return the data from a specific sensor
        sensor_token = self.sensors[sensor_id]['token']
        data = self.dataset.get_agent_sensor_files(self.vid, time_step)
        assert sensor_token in data, f"Sensor {sensor_token} not found in data"

        return data[sensor_token]

    @torch.no_grad()
    def get_state(self, time_step: int):
        """
        返回该车在给定time_step的观测与检测结果：
        - objects_local: (N,4,2) 本地BEV四角点
        - objects_world: (N,1,4,2) 世界系四角点（若可获得位姿/变换矩阵）
        - scores: (N,) 置信度
        """
        self.local_conf_map = np.zeros((self.dataset.map_dims[0], self.dataset.map_dims[1]), dtype=np.float32)
        state = {'vid': self.vid, 'time_step': time_step}

        # 1) 取该车该时刻的数据样本（与您现有代码一致）
        (padded_voxel_points, padded_voxel_points_teacher_det, label_one_hot, reg_target,
        reg_loss_mask, anchors_map, vis_maps, gt_max_iou, filename,
        target_agent_id, num_sensor, trans_matrix) = self.v2x_det_dataset[time_step][self.vid-1]

        plt.imshow(np.max(padded_voxel_points.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
        plt.show() 

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
        # 返回是 list（按agent），单车取第0个
        if not seq_results or len(seq_results[0]) == 0:
            state.update({
                'objects_local': np.zeros((0,4,2), dtype=np.float32),
                'objects_world': np.zeros((0,1,4,2), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'filename': filename
            })
            return state
        
        LOG.info(f"seq_results: {seq_results}")

        class_selected = seq_results[0][0][0][0]     # dict: {'pred':(N,4,2), 'score':(N,), ...}
        # objects_local  = class_selected['pred']    # (N,4,2) numpy
        # scores         = class_selected['score']   # (N,)   torch或numpy
        objects_local = []  # 模拟数据
        for i in range(100):
            objects_local.append(np.random.rand(4, 2) * i)  # 100个检测框的四个角点，范围在0-30米之间
        objects_local = np.array(objects_local, dtype=np.float32)
        scores = np.random.rand(100)  # 模拟数据

        # 4) 尝试变换到世界系（两条路：优先NuScenes位姿，退路用trans_matrix）
        objects_world = None

        # 4.1 优先：NuScenes位姿（如果你类里有 self.dataset / self.nusc 可获得 sample_data 的 tokens）
        try:
            sensors_data = self.dataset.get_agent_sensor_files(self.vid, time_step)
            lidar_sd = None
            for _, sd in sensors_data.items():
                if sd['channel'].split('_id_')[0] == self.sensor_channel:
                    lidar_sd = sd
                    break
            if lidar_sd and 'ego_pose_token' in lidar_sd and 'calibrated_sensor_token' in lidar_sd:
                pose = self.v2x_sim.get("ego_pose", lidar_sd['ego_pose_token'])
                cs   = self.v2x_sim.get("calibrated_sensor", lidar_sd['calibrated_sensor_token'])
                T_world_ego  = transform_matrix(pose['translation'], Quaternion(pose['rotation']))
                T_ego_sensor = transform_matrix(cs['translation'],   Quaternion(cs['rotation']))
                T_world_sensor = T_world_ego @ T_ego_sensor
                objects_world = bev_local_to_world_corners(T_world_sensor, objects_local)
        except Exception:
            pass

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
            'filename': filename
        })

        self.dataset.boxes_to_conf_map(self.local_conf_map, objects_world, scores)
        state.update({'local_map': self.local_conf_map,
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
    def __init__(self, cluster_id, leader_id, vehicles):
        self.cluster_id = cluster_id
        self.members : dict[Car] = vehicles # vid -> Car
        assert leader_id in range(len(vehicles)), "Leader ID is not in the cluster"
        self.leader = vehicles[leader_id]
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
    
    def get_state(self, time_step):
        # Return the state of the cluster
        cluster_state = {}

        for vid, member in self.members.items():
            member_state = member.get_state(time_step)
            cluster_state[member.vid] = member_state
        
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
        area_cnt = 0
        for vid, action in actions.items():
            area_cnt += len(action)
        for vid, member in self.members.items():    # receive data to update conf map, need to consider the delay
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


    

    



    


        
 