import os
import json
import math
from collections import defaultdict
from nuscenes.utils.geometry_utils import transform_matrix
from typing import List, Dict, Any, Optional
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes as V2XSimDataset

import logging
LOG = logging.getLogger(__name__)

class V2XSimReader:
    def __init__(self, root_dir: str = r"V2X-Sim-2.0-mini"):
        """
        目录结构示例:
        root_dir/
        ├── v2.0-mini/               # json 文件夹
        │   ├── sample_data.json
        │   ├── sample.json
        │   ├── sample_annotation.json
        │   ├── sensor.json
        │   └── ego_pose.json
        ├── sweeps/                  # 传感器原始数据
        │   ├── CAM_FRONT_id_1/
        │   │   └── scene_5_000006.png
        │   └── ...
        ├── lidarseg/                # 原始点云数据
        │   └── v2.0-mini/
        │       └── scene_5_000006.pcd.bin
        └── maps/                    # 地图文件
            └── <map_token>.bin
        """
        self.v2x_sim = V2XSimDataset(version='v2.0-mini', dataroot=root_dir, verbose=True)
        self.root        = root_dir
        self.json_dir    = os.path.join(root_dir, 'v2.0-mini')
        self.sweeps_dir  = os.path.join(root_dir, 'sweeps')
        self.lidar_dir   = os.path.join(root_dir, 'lidarseg', 'v2.0-mini')
        self.maps_dir    = os.path.join(root_dir, 'maps')

        # load JSONs
        def _load(fn):
            path = os.path.join(self.json_dir, fn)
            return json.load(open(path, 'r'))
        
        self.scene = self.v2x_sim.scene[0]
        self.first_sample_token = self.scene.get("first_sample_token")
        self.last_sample_token = self.scene.get("last_sample_token")
        self.channels = ['LIDAR_TOP']
                        # ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'DEP_FRONT']
                        #  'DEP_FRONT_LEFT', 'DEP_FRONT_RIGHT', 'DEP_BACK', 'DEP_BACK_LEFT', 'DEP_BACK_RIGHT', 'BEV_TOP', 'SEG_FRONT', \
                        #  'SEG_FRONT_RIGHT', 'SEG_FRONT_LEFT', 'SEG_BACK', 'SEG_BACK_LEFT', 'SEG_BACK_RIGHT', 'LIDAR_TOP', 'SEMLIDAR_TOP', \
                        #  'IMU_TOP', 'GNSS_TOP']
        # self.samples       = _load('sample.json')
        LOG.info(f"Loading V2X-Sim scene: {self.scene} \n \
                 first sample token {self.first_sample_token} \n \
                 last sample token {self.last_sample_token}")

        self.sample_data   = _load('sample_data.json')
        self.annotations   = _load('sample_annotation.json')
        self.sensors       = _load('sensor.json')
        self.ego_poses     = _load('ego_pose.json')
        self.instances     = _load('instance.json')
        self.calibrated_sensors = _load('calibrated_sensor.json')
        self.categories    = _load('category.json')

        self.start_timestamp = 6
        self.timestamp_length = 100
        self.time_length = 20 # 20s的数据，采样频率为5Hz
        self.frequency = 0.2

        self._preprocess()

        # build lookup tables
        self._build_indices()

        # build global map
        self.voxel_size = (10, 10, 0.4)  # 体素大小
        self.area_extents, self.map_dims = self.init_global_map(self.voxel_size)

    def _preprocess(self):
        """
        预处理数据集，提取车辆数、RSU数、传感器数等信息。
        """
        self.vehicle_ids = set()
        self.rsu_ids     = set()
        self.sensor_to_calibrated_sensor = {}
        self.calibrated_sensor_to_sensor = {}
        self.rsu_sensors, self.veh_sensors = self.sensors_info()
        self.n_vehicles = len(self.vehicle_ids)
        self.n_rsu      = len(self.rsu_ids)
        self.objects    = self.get_instance_tokens()
        self.n_objects   = len(self.objects)

        self.veh_trajectory = {}

        self.obj_data = {}
        self.obj_trajectory  = {}

        LOG.debug(f"Number of vehicles in dataset: {self.n_vehicles}")
        LOG.debug(f"Number of RSUs in dataset: {self.n_rsu}")
        LOG.debug(f"Number of objects in dataset: {self.n_objects}")

    def _build_indices(self):
        # instance lookup
        self._instances = {}
        for instance in self.instances:
            key = instance['token']
            self._instances[key] = instance

        # annotation lookup
        self._idx_ann = {}
        for ann in self.annotations:    # 凭借这个可以查看某个instance的lifecycle
            key = ann['token']
            self._idx_ann[key]= ann
        
        # category lookup
        self._category = {}
        for type in self.categories:
            key = type['token']
            self._category[key] = type


        self._idx_sample = {}
        self._idx_sd = {}
        sample_token = self.first_sample_token
        while sample_token != '' and sample_token is not None:
            # sample lookup: timestamp -> sample
            sample = self.v2x_sim.get('sample', sample_token)
            key = sample['timestamp']
            self._idx_sample[key] = sample

            # sample_data lookup: (calibrated_sensor_token, timestamp) -> sample_data
            for vehicle_id in self.vehicle_ids:
                for channel in self.channels:
                    if f'{channel}_id_{vehicle_id}' in sample['data']:
                        # print(f"Found {channel} for vehicle {vehicle_id} in sample {sample_token}")
                        sample_data = self.v2x_sim.get('sample_data', sample['data'][f'{channel}_id_{vehicle_id}'])
                        key_sensor_timestamp = (sample_data['calibrated_sensor_token'], sample_data['timestamp'])
                        self._idx_sd[key_sensor_timestamp] = sample_data
                        # print(f"Sample data: {sample_data}")

            sample_token = sample['next']

        # sensor channel lookup : sensor_token -> channel
        self._idx_channel = {}
        for s in self.sensors:
            key_sensor = s['token']
            self._idx_channel[key_sensor] = s['channel']

        # ego_pose lookup
        self._idx_ego_pose = {}
        for ego_pose in self.ego_poses:
            key = ego_pose['token']
            self._idx_ego_pose[key] = ego_pose   

    def get_instance_tokens(self) -> int:
        """
        返回数据集中实例的数量。
        """
        tokens = [instance['token'] for instance in self.instances]

        return tokens
    
    def get_sample(self, time_step):
        """
        返回指定agent在指定时间步的样本数据。
        agent_id: 0 for RSU, 1 for vehicle
        time_step: 时间步
        """
        timestamp = self.start_timestamp + time_step
        sample = self._idx_sample.get(timestamp)
        assert sample is not None, f"Sample not found for timestamp {timestamp}"

        return sample
    
    def sensors_info(self):
        """
        返回一个 dict:
        vehicle_id -> {
            modality -> {
            'count': int,
            'sensors': [sensor_token, ...],
            },
            ...
        }
        """
        veh_info = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sensors': []}))
        rsu_info = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sensors': []}))

        for item in self.calibrated_sensors:
            calibrated_sensor_token = item['token']
            sensor_token = item['sensor_token']
            if sensor_token not in self.sensor_to_calibrated_sensor:
                self.sensor_to_calibrated_sensor[sensor_token] = calibrated_sensor_token
            else:
                raise ValueError(f"Duplicate sensor token: {sensor_token}")
            if calibrated_sensor_token not in self.calibrated_sensor_to_sensor:
                self.calibrated_sensor_to_sensor[calibrated_sensor_token] = sensor_token
            else:
                raise ValueError(f"Duplicate calibrated sensor token: {calibrated_sensor_token}")
        
        for s in self.sensors:
            channel = s['channel']
            parts = channel.split('_id_')
            if len(parts) > 1 and parts[0] in self.channels:
                modality = s.get('modality', 'unknown')
                token = s['token']
                id = int(parts[1].split('_')[0])
                if id == 0:
                    self.rsu_ids.add(id)
                    rsu_info[id][modality]['count'] += 1
                    rsu_info[id][modality]['sensors'].append(token)
                else:
                    self.vehicle_ids.add(id)
                    veh_info[id][modality]['count'] += 1
                    veh_info[id][modality]['sensors'].append(token)

        # for rid, modalities in rsu_info.items():
        #     print(f"RSU {rid}:")
        #     for modality, data in modalities.items():
        #         print(f"  {modality}: {data['count']} sensors")
        #         print(f"    Tokens: {data['sensors']}")

        # for vid, modalities in veh_info.items():
        #     print(f"Vehicle {vid}:")
        #     for modality, data in modalities.items():
        #         print(f"  {modality}: {data['count']} sensors")
        #         print(f"    Tokens: {data['sensors']}")

        return rsu_info, veh_info
        
    def get_agent_sensor_files(self, agent_id, time_step):
        """
        返回agent在指定时间步的传感器数据文件路径。
        agent_id: 0 for RSU, 1 for vehicle
        time_step: 时间步
        {
            modality : [
                sample_data, ...
            ]
        }

        """
        sensors = None
        if agent_id == 0:
            sensors = self.rsu_sensors[agent_id]
        else:
            sensors = self.veh_sensors[agent_id]

        data = {}
        for modality, info in sensors.items():
            # print(f"Modality: {modality}")
            # print(f"Sensor count: {info['count']}")
            # print(f"Sensor tokens: {info['sensors']}")
            for sensor_token in info['sensors']:
                calibrated_sensor_token = self.sensor_to_calibrated_sensor.get(sensor_token)
                timestamp = self.start_timestamp + time_step
                sample_data = self._idx_sd.get((calibrated_sensor_token, timestamp))
                assert sample_data is not None, f"Sample data not found for {sensor_token} at {time_step}, calibrated_sensor_token: {calibrated_sensor_token}, timestamp: {timestamp}"

                data[sensor_token] = sample_data
        
        # print(f"Agent {agent_id} sensor data at time step {time_step}:")
        # for sensor_token, sample_data in data.items():
        #     print(f"  Sensor token: {sensor_token}")
        #     print(f"    Sample data: {sample_data}")
        
        return data

    def get_vehicle_metadata(self, vid):
        """
        返回车辆所有时间步的元数据，包括位置、速度等信息。
        """
        # 通过车上某个传感器的token，从sample_data中找到ego_pose_token，然后在ego_pose中找到位置和速度
        trajectory = []
        prev_pos = None
        prev_t   = None
        for modality, info in self.veh_sensors[vid].items():
            temp_sensor_token = info['sensors'][0]
            temp_calibrated_sensor_token = self.sensor_to_calibrated_sensor.get(temp_sensor_token)

            all_egpose_data = []
            for time_step in range(0, self.timestamp_length):
                timestamp = self.start_timestamp + time_step
                sample_data = self._idx_sd.get((temp_calibrated_sensor_token, timestamp))
                assert sample_data is not None, f"Sample data not found for {temp_sensor_token} at {time_step}"
                ego_pose_token = sample_data['ego_pose_token']
                # print(f"ego_pose_token: {ego_pose_token}")
                ego_pose = self._idx_ego_pose.get(ego_pose_token)
                # print(f"ego_pose: {ego_pose}")
                assert ego_pose is not None, f"Ego pose not found for {ego_pose_token}"
                all_egpose_data.append(ego_pose)

            # temp_sensor_token = info['sensors'][3]
            # temp_calibrated_sensor_token = self.sensor_to_calibrated_sensor.get(temp_sensor_token)

            # all_egpose_data2 = []
            # for time_step in range(0, self.timestamp_length):
            #     timestamp = self.start_timestamp + time_step
            #     sample_data = self._idx_sd.get((temp_calibrated_sensor_token, timestamp))
            #     assert sample_data is not None, f"Sample data not found for {temp_sensor_token} at {time_step}"
            #     ego_pose_token = sample_data['ego_pose_token']
            #     # print(f"ego_pose_token: {ego_pose_token}")
            #     ego_pose = self._idx_ego_pose.get(ego_pose_token)
            #     # print(f"ego_pose: {ego_pose}")
            #     assert ego_pose is not None, f"Ego pose not found for {ego_pose_token}"
            #     all_egpose_data2.append(ego_pose)
            
            # # 验证同一辆车的ego_pose的rotation是否一致
            # for i in range(len(all_egpose_data)):
            #     print(f"ego_pose1: {all_egpose_data[i]}")
            #     print(f"ego_pose2: {all_egpose_data2[i]}")
            #     assert all_egpose_data[i]['rotation'] == all_egpose_data2[i]['rotation'], f"Rotation mismatch at index {i} for vehicle {vid}"


            for rec in all_egpose_data:
                # print(f"ego_pose: {rec}")
                t = rec['timestamp']
                pos = np.array(rec['translation'], dtype=float)
                rotatoion = np.array(rec['rotation'], dtype=float)
                
                if prev_pos is None:
                    vel = np.zeros(3)
                else:
                    dt = (t - prev_t) * self.frequency
                    if dt <= 0:
                        vel = np.zeros(3)
                    else:
                        vel = (pos - prev_pos) / dt
                prev_pos = pos
                prev_t = t
                trajectory.append({
                    'timestamp': t,
                    'position': pos.tolist(),
                    'velocity': vel.tolist(),
                    'rotation': rotatoion.tolist()
                })

            break

        # print(f"Vehicle {vid} trajectory:")
        # for rec in trajectory:
        #     print(f"  Timestamp: {rec['timestamp']}, Position: {rec['position']}, Velocity: {rec['velocity']}, Rotation: {rec['rotation']}")    
        
        return trajectory

    def get_sensor_metadata(self, sensor_token: str) -> Dict[str, Any]:
        """
        返回传感器的元数据，包括传感器类型、位置、旋转等信息。
        """
        sensor = self.sensors.get(sensor_token)
        if sensor is None:
            raise ValueError(f"Sensor {sensor_token} not found")
        return sensor
    
    def get_object_metadata(self, object_token: str) -> Dict[str, Any]:
        """
        返回对象的元数据，包括对象类型、位置、旋转等信息。
        """
        if self.obj_data.get(object_token) is None:
            self.obj_data[object_token] = self.get_object_data(object_token)

        trajectory = []
        timestamp = self.start_timestamp
        prev_pos = None

        for rec in self.obj_data[object_token]:
            pos = np.array(rec['translation'], dtype=float)
            rotation = np.array(rec['rotation'], dtype=float)

            if prev_pos is None:
                vel = np.zeros(3)
            else:
                dt = self.frequency
                if dt <= 0:
                    vel = np.zeros(3)
                else:
                    vel = (pos - prev_pos) / dt
            
            trajectory.append({
                'timestamp': timestamp,
                'position': pos.tolist(),
                'velocity': vel.tolist(),
                'rotation': rotation.tolist()
            })
            prev_pos = pos
            timestamp += 1

        # print(f"Object {object_token} trajectory:")
        # for rec in trajectory:
        #     print(f"  Timestamp: {rec['timestamp']}, Position: {rec['position']}, Velocity: {rec['velocity']}, Rotation: {rec['rotation']}")
        return trajectory
         
    def get_object_data(self, object_token: str):
        """
        返回对象(instance)的包括对象的传感器数据。
        """
        instance_obj = self._instances.get(object_token)
        first_annotation_token = instance_obj['first_annotation_token']
        last_annotation_token = instance_obj['last_annotation_token']
        data = []

        while first_annotation_token != "":
            annotation = self._idx_ann.get(first_annotation_token)
            # assert annotation is not None, f"Annotation {first_annotation_token} not found"
            if annotation == None:
                print(f"token {object_token}, annotation {annotation} not found")
                return data
            data.append(annotation)
            first_annotation_token = annotation['next']

        # print(f"Last annotation: {annotation}")
        # print(f"Data: {data}")

        return data
    
    def get_type(self, token):
        instance = self._instances.get(token)
        assert instance is not None, f"instance {token} not found"
        category_token = instance['category_token']
        type = self._category[category_token]
        if type is not None:
            return type['name']
        return 'Uknown'

    def get_vehicle_count(self) -> int:
        """
        返回数据集中车辆的数量。
        """
        return self.n_vehicles

    def get_vehicle_sensors(self) -> Dict[str, Any]:
        """
        返回数据集中车辆的传感器信息。
        """
        return self.veh_sensors
    
    
    def get_rsu_count(self) -> int:
        """
        返回数据集中RSU的数量。
        """
        return self.n_rsu
    
    def get_rsu_sensors(self) -> Dict[str, Any]:
        """
        返回数据集中RSU的传感器信息。
        """
        return self.rsu_sensors
    
    def get_object_count(self) -> int:
        """
        返回数据集中对象的数量。
        """
        return self.n_objects
    
    def get_object_tokens(self) -> List[str]:
        """
        返回数据集中对象的传感器信息。
        """
        return self.objects
    
    def get_time_step_length(self) -> int:
        """
        返回数据集中时间步的长度。
        """
        return self.timestamp_length


    def get_cord_range(self, translation, rotation, size):
        length, width, height = size    # 这里有可能是width, length, height = size  # nuScenes: size = [w, l, h]
        corners_local = np.array([
            [ length/2,  width/2,  height/2],
            [ length/2, -width/2,  height/2],
            [-length/2, -width/2,  height/2],
            [-length/2,  width/2,  height/2],
            [ length/2,  width/2, -height/2],
            [ length/2, -width/2, -height/2],
            [-length/2, -width/2, -height/2],
            [-length/2,  width/2, -height/2],
        ])
        R = Quaternion(rotation).rotation_matrix
        center = np.array(translation)
        world_corners = (R @ corners_local.T).T + center
        min_xyz = world_corners.min(axis=0)
        max_xyz = world_corners.max(axis=0)

        return min_xyz, max_xyz

    def init_global_map(self, voxel_size=(0.25, 0.25, 0.4)):
        """
        初始化全局地图，设置地图的尺寸和分辨率。
        returns: area_extents, voxel_size
        self.voxel_size = (0.25, 0.25, 0.4)
        self.area_extents = np.array([[-96.0, 96.0], [-96.0, 96.0], [-4.0, 4.0]])
        """
        area_extents = np.array([[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]])
        sample_token = self.first_sample_token
        while sample_token != '' and sample_token is not None:
            sample = self.v2x_sim.get("sample", sample_token)
            next_sample_token = sample.get('next')
            # print(f"Processing sample: {sample.keys()}")
            if sample is None:
                continue
            for ann_token in sample['anns']:
                ann = self.v2x_sim.get("sample_annotation", ann_token)
                if ann is None:
                    continue
                translation = ann['translation']
                rotation = ann['rotation']
                size = ann['size']
                min_xyz, max_xyz = self.get_cord_range(translation, rotation, size)
                # print(f"Annotation {ann_token} min_xyz: {min_xyz}, max_xyz: {max_xyz}")

                area_extents[:, 0] = np.minimum(area_extents[:, 0], min_xyz)
                area_extents[:, 1] = np.maximum(area_extents[:, 1], max_xyz)
            
            for sample_data_token in sample['data'].values():
                sample_data = self.v2x_sim.get("sample_data", sample_data_token)
                if sample_data is None:
                    continue

                ego_pose_token = sample_data['ego_pose_token']
                ego_pose = self.v2x_sim.get("ego_pose", ego_pose_token)
                ego_rotation = ego_pose['rotation']
                ego_translation = ego_pose['translation']
                min_xyz, max_xyz = self.get_cord_range(ego_translation, ego_rotation, [0.1, 0.1, 0.1])
                # print(f"Calibrated sensor {calibrated_sensor_token} min_xyz: {min_xyz}, max_xyz: {max_xyz}")

                area_extents[:, 0] = np.minimum(area_extents[:, 0], min_xyz)
                area_extents[:, 1] = np.maximum(area_extents[:, 1], max_xyz)

            # print(f"Current area extents: {area_extents}")
                
            sample_token = next_sample_token

        print(f"Global map area extents: {area_extents}")
        print(f"Global map voxel size: {voxel_size}")
        area_extents[:,0] = (np.floor(area_extents[:,0] / voxel_size)) * voxel_size
        area_extents[:,1] = (np.ceil(area_extents[:,1] / voxel_size)) * voxel_size
        print(f"Adjusted area extents: {area_extents}")

        map_dims = np.ceil((area_extents[:, 1] - area_extents[:, 0]) / voxel_size)
        map_dims = map_dims.astype(int)
        print(f"Global map dimensions: {map_dims}")
        
        
        return area_extents, map_dims
    
    def boxes_to_conf_map(self, conf_map, boxes_world, scores, agg="max"):
        """
        将世界坐标系下的边界框转换为置信度地图。
        boxes_world: (N, 1, 4, 2) 或 (N, 4, 2) 世界系四角点（xy）
        scores: (N,) 每个框的分数
        area_extents: [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        voxel_size: (vx, vy, vz)
        agg: 'max' 或 'sum'
        returns: conf_map (Ny, Nx)
        """
        map_xmin, map_xmax = self.area_extents[0]
        map_ymin, map_ymax = self.area_extents[1]
        # LOG.info(f"conf map area: [{map_xmin}, {map_xmax}], [{map_ymin}, {map_ymax}]")
        # LOG.info(f"conf map size: {conf_map.shape}, {conf_map}")
        for k in range(boxes_world.shape[0]):
            corners = boxes_world[k][0] 
            x_min = np.maximum(np.min(corners[:, 0]), map_xmin)
            x_max = np.minimum(np.max(corners[:, 0]), map_xmax)
            y_min = np.maximum(np.min(corners[:, 1]), map_ymin)
            y_max = np.minimum(np.max(corners[:, 1]), map_ymax)

            x_index_min = int(np.floor((x_min - map_xmin) / self.voxel_size[0]))
            x_index_max = int(np.floor((x_max - map_xmin) / self.voxel_size[0]))
            y_index_min = int(np.floor((y_min - map_ymin) / self.voxel_size[1]))
            y_index_max = int(np.floor((y_max - map_ymin) / self.voxel_size[1]))

            # LOG.info(f"Object corners: {corners}, x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
            # LOG.info(f"Indices: x_index_min: {x_index_min}, x_index_max: {x_index_max}, y_index_min: {y_index_min}, y_index_max: {y_index_max}")
            # LOG.info(f"Scores: {scores[k]}, conf_map shape: {conf_map.shape}, conf_map: {conf_map}")
            # 遍历这个索引，然后将scores加到conf_map上
            for i in range(x_index_min, x_index_max):
                for j in range(y_index_min, y_index_max):
                    if agg == "max":
                        LOG.debug(f"Updating conf_map[{i}, {j}] with score {scores[k]}")
                        conf_map[i, j] = max(conf_map[i, j], scores[k])
                    elif agg == "sum":
                        conf_map[i, j] += scores[k]
                    else:
                        raise ValueError(f"Unknown aggregation method: {agg}")
            # LOG.info(f"dtype: {conf_map.dtype}")
            # LOG.info(f"nonzero: {np.count_nonzero(conf_map)}")
            # LOG.info(f"min/max: {float(conf_map.min())}, {float(conf_map.max())}")
            
        return conf_map

# ----------------------------
# 示例用法
# ----------------------------
if __name__ == "__main__":
    root = "D:\\Code\\coperception\\data\\v2x_sim_dataset"
    reader = V2XSimReader(root)

    sample_token = "q68g8v6474j101675mphs2r8w97575ca"
    agent_token  = "qd2333299cth40oy7mx16ejzcn02c345"

    # reader.get_agent_sensor_files(0, 0)
    # reader.get_agent_sensor_files(0, 99)

    # reader.get_agent_sensor_files(1, 0)
    # reader.get_agent_sensor_files(1, 99)

    # reader.get_agent_sensor_files(5, 0)
    # reader.get_agent_sensor_files(5, 99)

    # reader.get_vehicle_metadata(4)



    # reader.get_object_data('rzu6onez5wun9318opx587yi8q8di1gt')

    # instance_token = '00pzy71d3g8a3vl3d4bay484xe7gjayk'
    # reader.get_object_metadata(instance_token)
