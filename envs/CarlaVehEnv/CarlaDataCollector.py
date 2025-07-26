import os
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional

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
        self.root        = root_dir
        self.json_dir    = os.path.join(root_dir, 'v2.0-mini')
        self.sweeps_dir  = os.path.join(root_dir, 'sweeps')
        self.lidar_dir   = os.path.join(root_dir, 'lidarseg', 'v2.0-mini')
        self.maps_dir    = os.path.join(root_dir, 'maps')

        # load JSONs
        def _load(fn):
            path = os.path.join(self.json_dir, fn)
            return json.load(open(path, 'r'))
        self.sample_data   = _load('sample_data.json')
        self.samples       = _load('sample.json')
        self.annotations   = _load('sample_annotation.json')
        self.sensors       = _load('sensor.json')
        self.ego_poses     = _load('ego_pose.json')
        self.instances     = _load('instance.json')
        self.calibrated_sensors = _load('calibrated_sensor.json')
        self.categories    = _load('category.json')

        self.start_timestamp = 6
        self.timestamp_length = 100
        self.time_length = 20 # 20s的数据，采样频率为5Hz

        self._preprocess()

        # build lookup tables
        self._build_indices()

        

    def _preprocess(self):
        """
        预处理数据集，提取车辆数、RSU数、传感器数等信息。
        """
        self.vehicle_ids = set()
        self.rsu_ids     = set()
        self.rsu_sensors, self.veh_sensors = self.sensors_info()
        self.n_vehicles = len(self.vehicle_ids)
        self.n_rsu      = len(self.sensors)
        self.objects    = self.get_instance_tokens()
        self.n_objects   = len(self.objects)

        
    def get_instance_tokens(self) -> int:
        """
        返回数据集中实例的数量。
        """
        tokens = {instance['token'] for instance in self.instances}

        return tokens
    
    def sensors_info(self) -> List[dict]:
        """
        返回一个 dict:
        vehicle_id -> {
            modality -> {
            'count': int,
            'tokens': [sensor_token, ...]
            },
            ...
        }
        """
        veh_info = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'tokens': []}))
        rsu_info = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'tokens': []}))
        
        for s in self.sensors:
            channel = s['channel']
            parts = channel.split('_id_')
            if len(parts) > 1:
                modality = s.get('modality', 'unknown')
                token = s['token']
                id = int(parts[1].split('_')[0])
                if id == 0:
                    self.rsu_ids.add(id)
                    rsu_info[id][modality]['count'] += 1
                    rsu_info[id][modality]['tokens'].append(token)
                else:
                    self.vehicle_ids.add(id)
                    veh_info[id][modality]['count'] += 1
                    veh_info[id][modality]['tokens'].append(token)

        print(f"RSU count: {len(self.rsu_ids)}")
        print(f"Vehicle count: {len(self.vehicle_ids)}")

        for rid, modalities in rsu_info.items():
            print(f"RSU {rid}:")
            for modality, data in modalities.items():
                print(f"  {modality}: {data['count']} sensors")
                print(f"    Tokens: {data['tokens']}")

        for vid, modalities in veh_info.items():
            print(f"Vehicle {vid}:")
            for modality, data in modalities.items():
                print(f"  {modality}: {data['count']} sensors")
                print(f"    Tokens: {data['tokens']}")

        return rsu_info, veh_info
        






    def _build_indices(self):
        # annotation lookup
        self._idx_ann = {}
        for ann in self.annotations:
            key = (ann['instance_token'], ann['sample_token'])
            self._idx_ann.setdefault(key, []).append(ann)

        # sample_data lookup
        self._idx_sd = {}
        for sd in self.sample_data:
            key = (sd['calibrated_sensor_token'], sd['sample_token'])
            self._idx_sd[key] = sd

        # channel -> sensor_token
        self._token_of_channel = { s['channel']: s['token'] for s in self.sensors }

        # sample_token -> frame info
        self._frame_of = { f['token']: f for f in self.samples }

    def get_vehicle_metadata(self, vid, time_step) -> Dict[str, Any]:
        """
        返回车辆在指定时间步的元数据，包括位置、速度等信息。
        """
        # 这里可以根据需要实现获取车辆元数据的逻辑
        pass

    def get_vehicle_sensors_data(self, vid, time_step):
        """
        返回车辆在指定时间步的传感器数据。
        """
        sensors = self.veh_sensors.get(vid, {})
        data = {}
        for modality, info in sensors.items():
            if str.upper(modality) == 'CAMERA':
                # 处理相机数据
                data[modality] = self.get_sensor_frame_file(info['tokens'][0], time_step)
            elif str.upper(modality) == 'LIDAR':
                # 处理激光雷达数据
                data[modality] = self.get_pointcloud_file(time_step)
            else:
                # 其他模态
                data[modality] = None
        return data
    
    def get_rsu_sensors_data(self, rid, time_step):
        """
        返回RSU在指定时间步的传感器数据。
        """
        sensors = self.rsu_sensors.get(rid, {})
        data = {}
        for modality, info in sensors.items():
            if modality == 'CAMERA':
                # 处理相机数据
                data[modality] = self.get_sensor_frame_file(info['tokens'][0], time_step)
            elif modality == 'LIDAR':
                # 处理激光雷达数据
                data[modality] = self.get_pointcloud_file(time_step)
            else:
                # 其他模态
                data[modality] = None
        return data

    def get_agent_perceptions(self,
                              agent_token: str,
                              sample_token: str) -> List[Dict[str, Any]]:
        """返回该 agent 在该帧的所有标注。"""
        return self._idx_ann.get((agent_token, sample_token), [])

    def get_sensor_frame_info(self,
                              sensor_channel: str,
                              sample_token: str) -> Optional[Dict[str, Any]]:
        """
        返回该通道在该帧的 metadata（含 filename、width、height …），
        但不加载文件本身。
        """
        calib_token = self._token_of_channel.get(sensor_channel)
        if calib_token is None:
            return None
        return self._idx_sd.get((calib_token, sample_token))

    def get_sensor_frame_file(self,
                              sensor_channel: str,
                              sample_token: str) -> Optional[str]:
        """
        根据 metadata 中的 filename，在 sweeps/ 目录下返回原始文件绝对路径。
        """
        info = self.get_sensor_frame_info(sensor_channel, sample_token)
        if not info or 'filename' not in info:
            return None
        rel = info['filename']  # e.g. "CAM_FRONT_id_1/scene_5_000006.png"
        path = os.path.join(self.sweeps_dir, rel)
        return path if os.path.exists(path) else None

    def get_pointcloud_file(self,
                            sample_token: str) -> Optional[str]:
        """
        返回该帧的原始点云文件路径 (pcd.bin)，位于 lidarseg/v2.0-mini/
        通过 sample_data 中任意一条记录的 filename 提取 basename:
        'scene_5_000006' -> 'scene_5_000006.pcd.bin'
        """
        # 找到任何一条 sample_data 记录来提取 basename
        rec = next((r for r in self.sample_data if r['sample_token']==sample_token), None)
        if rec is None or 'filename' not in rec:
            return None
        base = os.path.splitext(os.path.basename(rec['filename']))[0]
        fn   = f"{base}.pcd.bin"
        path = os.path.join(self.lidar_dir, fn)
        return path if os.path.exists(path) else None

    def get_map_file(self, map_token: str) -> Optional[str]:
        """
        返回指定 map_token 对应的地图文件路径，如 maps/<map_token>.bin
        """
        fn = f"{map_token}.bin"
        path = os.path.join(self.maps_dir, fn)
        return path if os.path.exists(path) else None

    def get_agent_state(self,
                        agent_token: str,
                        sample_token: str) -> Dict[str, Any]:
        """
        返回该 agent 在该帧的位置和速度估计：
          - position: [x,y,z]
          - velocity: [vx,vy,vz] (若无 prev 则全 0)
        """
        anns = self.get_agent_perceptions(agent_token, sample_token)
        if not anns:
            raise KeyError(f"No annotation for {agent_token} @ {sample_token}")
        pos = anns[0]['translation']

        frame = self._frame_of[sample_token]
        prev = frame.get('prev')
        if prev:
            prev_anns = self.get_agent_perceptions(agent_token, prev)
            if prev_anns:
                ppos = prev_anns[0]['translation']
                t0 = frame['timestamp']
                t1 = self._frame_of[prev]['timestamp']
                dt = (t0 - t1) or 1.0
                vel = [(pos[i]-ppos[i])/dt for i in range(3)]
            else:
                vel = [0.0,0.0,0.0]
        else:
            vel = [0.0,0.0,0.0]

        return {'position': pos, 'velocity': vel}


# ----------------------------
# 示例用法
# ----------------------------
if __name__ == "__main__":
    root = "/home/peh324/Codes/V2X-Sim-2.0-mini/V2X_sim_2_mini"
    reader = V2XSimReader(root)

    sample_token = "q68g8v6474j101675mphs2r8w97575ca"
    agent_token  = "qd2333299cth40oy7mx16ejzcn02c345"



    # # 1) agent 的所有感知标注
    # anns = reader.get_agent_perceptions(agent_token, sample_token)
    # print("Annotations:", anns)

    # # 2) 某通道原始数据文件 (image / npz)
    # img_path = reader.get_sensor_frame_file("CAM_FRONT_id_1", sample_token)
    # print("Image file:", img_path)

    # # 3) 原始点云数据 (.pcd.bin)
    # lidar_path = reader.get_pointcloud_file(sample_token)
    # print("Pointcloud file:", lidar_path)

    # # 4) 地图文件
    # # 首先从 sample_data.json 或 scene.json 中获知 map_token，这里假设已知：
    # map_path = reader.get_map_file("map_xyz123")
    # print("Map file:", map_path)

    # # 5) agent 的位置与速度
    # state = reader.get_agent_state(agent_token, sample_token)
    # print("Agent state:", state)
