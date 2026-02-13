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
from agent import FasterRCNNDetector
from .Logger import CarlaDataLogger

class CarlaCommEnv(CarlaBaseEnv):
    def __init__(self, config):
        super().__init__(config) # 初始化父类
        self.obs_vehicles = []      # TODO: dose these vehicles need to be fixed during an episode?
        self.detector = FasterRCNNDetector
        # vehicle group
        self.v_groups = {}
        
        root = self._config.dataset_root
        run_name = "GODE"
        self._logger = CarlaDataLogger(root_dir=root, run_name=run_name, flush_every=10)
        self._vehicle_to_group = {}  # 
        self._time_step = 0
        
    def on_reset(self) -> None:
        """
        Override this method to perform additional reset operations.
        Specifically, you can spawn actors and plan routes here.
        """
        
        traffic_lights = self._world.carla_actors(actor_type = 'traffic_light')

        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(9999)
            tl.set_red_time(0)
            tl.set_yellow_time(0)
            
        self._time_step = 0
        self.obs_vehicles = []
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

        self.v_groups = {}
        self.init_vehicle_groups(distance = 5000, count=10)
        self._vehicle_to_group = {}
        for gid, vids in self.v_groups.items():
            for vid in vids:
                self._vehicle_to_group[int(vid)] = int(gid)

        self.init_obs_vehicle_path()
        # === 新增修复代码：物理热身 ===
        
        print("Warming up simulation for physics settling...")
        for _ in range(20):  # 运行 20 帧让车辆落地
            self._world._world.tick()  # 确保这里调用的是 world.tick()

        if self._time_step != 0:
            print(f"Flushing logger at time step {self._time_step}")
            self._logger.flush()  
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
    
    def init_vehicle_groups(self, distance=5000, count=10):
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

    def visualize_topology(self):
        """
        - Node color = group id
        - For each node, connect to its 2 nearest neighbors within the SAME group (k=2)
        - Save figure to disk every call
        - Use world (x,y) coordinates but enforce a square view (equal scale) for nicer visuals
        """

        # ---------- 1) collect positions ----------
        if not hasattr(self, "v_groups") or not self.v_groups:
            return

        node_pos = {}   # actor_id -> (x,y)
        node_gid = {}   # actor_id -> group_id

        for gid, vids in self.v_groups.items():
            for vid in vids:
                actor = self._world.actor_dict.get(int(vid), None)
                if actor is None:
                    continue
                loc = actor.get_transform().location
                node_pos[int(vid)] = (float(loc.x), float(loc.y))
                node_gid[int(vid)] = int(gid)

        if len(node_pos) < 2:
            return

        # ---------- 2) figure setup (reuse single figure) ----------
        if not hasattr(self, "_topo_fig") or self._topo_fig is None:
            self._topo_fig = plt.figure(figsize=(7, 7))
            self._topo_ax = self._topo_fig.add_subplot(111)
            plt.ion()

        ax = self._topo_ax
        ax.clear()

        # ---------- 3) color map by group ----------
        group_ids = sorted(set(node_gid.values()))
        cmap = plt.get_cmap("tab20")
        gid_to_color = {g: cmap(i % 20) for i, g in enumerate(group_ids)}

        # ---------- 4) build edges: kNN (k=2) within each group ----------
        # We'll create an undirected edge set to avoid drawing duplicates.
        edges = set()  # store (min(u,v), max(u,v))

        k = 3
        for gid, vids in self.v_groups.items():
            vids = [int(v) for v in vids if int(v) in node_pos]
            if len(vids) < 2:
                continue

            pts = np.array([node_pos[v] for v in vids], dtype=np.float32)  # (M,2)

            # pairwise distances (M,M)
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)  # (M,M)
            np.fill_diagonal(dist, np.inf)

            for i, u in enumerate(vids):
                # nearest k indices
                nn_idx = np.argsort(dist[i])[: min(k, len(vids)-1)]
                for j in nn_idx:
                    v = vids[int(j)]
                    a, b = (u, v) if u < v else (v, u)
                    edges.add((a, b))

        # ---------- 5) draw edges (length reflects relative distance) ----------
        # Make near edges thicker / less transparent, far edges thinner / more transparent.
        # We compute distances in world coords.
        if edges:
            dlist = []
            for u, v in edges:
                pu = np.array(node_pos[u], dtype=np.float32)
                pv = np.array(node_pos[v], dtype=np.float32)
                dlist.append(float(np.linalg.norm(pu - pv)))
            dmin, dmax = (min(dlist), max(dlist)) if dlist else (0.0, 1.0)
            span = max(dmax - dmin, 1e-6)

            for (u, v) in edges:
                gid = node_gid.get(u, node_gid.get(v, -1))
                color = gid_to_color.get(gid, (0.5, 0.5, 0.5))

                pu = np.array(node_pos[u], dtype=np.float32)
                pv = np.array(node_pos[v], dtype=np.float32)
                d = float(np.linalg.norm(pu - pv))

                # normalized distance in [0,1]
                dn = (d - dmin) / span
                # near => thicker & higher alpha; far => thinner & lower alpha
                lw = float(np.clip(2.5 - 1.8 * dn, 0.6, 2.5))
                alpha = float(np.clip(0.65 - 0.45 * dn, 0.12, 0.65))

                ax.plot([pu[0], pv[0]], [pu[1], pv[1]],
                        linewidth=lw, alpha=alpha, color=color)

        # ---------- 6) draw nodes ----------
        for gid in group_ids:
            vids = [vid for vid in node_pos.keys() if node_gid.get(vid) == gid]
            pts = np.array([node_pos[v] for v in vids], dtype=np.float32)

            ax.scatter(pts[:, 0], pts[:, 1],
                    s=70, color=gid_to_color[gid],
                    label=f"group {gid}",
                    edgecolors="k", linewidths=0.5)

            # optional: label actor id
            for vid in vids:
                x, y = node_pos[vid]
                ax.text(x + 0.5, y + 0.5, str(vid), fontsize=8)

        # ---------- 7) make it square & nice (no normalization, but square view) ----------
        all_pts = np.array(list(node_pos.values()), dtype=np.float32)
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)

        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        rx, ry = (xmax - xmin), (ymax - ymin)
        r = max(rx, ry) * 0.55  # half-range (slightly padded)
        r = max(r, 5.0)         # avoid too tiny view (meters)

        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

        ts = getattr(self, "_time_step", None)
        ax.set_title(f"Vehicle-group topology" + (f" | t={ts}" if ts is not None else ""))
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.legend(loc="best", fontsize=8)

        self._topo_fig.tight_layout()
        self._topo_fig.canvas.draw_idle()
        plt.pause(0.001)

        # ---------- 8) save to disk ----------
        # Prefer CARLA snapshot frame if available, else use timestep
        frame_id = None
        try:
            frame_id = int(self._world.carla_world.get_snapshot().frame)
        except Exception:
            frame_id = None

        if frame_id is None:
            frame_id = int(ts) if ts is not None else 0

        # default save dir: dataset_root/run_name/topology (if you have config), else ./data/topology
        root = self._config.dataset_root
        run_name = "GODE"
        if root is None:
            save_dir = os.path.join("data", "topology", run_name)
        else:
            save_dir = os.path.join(root, run_name, "topology")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"topology_{frame_id:08d}.png")
        self._topo_fig.savefig(save_path, dpi=180)


    
    def on_step(self) -> None:
        """
        Override this method to perform additional operations at each step.
        Specifically, you can update the planner and the route here.
        This method will be called after the simulator ticks.
        """
        # run object detection, get confidence map
        
        # run the RL algorithm to decide what to share based on the observation and communication state
        
        for actor_id in self.obs_vehicles:
            pass
        
        if self._time_step % 5 == 0:
            self.visualize_topology()
        
    def step(self, action):
        self.apply_control(action)
        self._world.step()
        self._time_step += 1
        env_state = {} #self.get_state()
        
        snapshot = self._world.carla_world.get_snapshot()
        frame_id = int(snapshot.frame)
        truncated = self._time_step >= self._config.max_episode_steps

        # self.obs, obs_info = self._observer.get_observation(env_state)
        obs_info = {}
        self.obs = {}
        for actor_id, observer in self._observers.items():
            one_obs, one_obs_info = observer.get_observation(self.get_state())    
            self.obs.setdefault(actor_id, one_obs)
            obs_info.setdefault(actor_id, one_obs_info)

        for actor_id in self.obs_vehicles:
            actor = self._world.actor_dict.get(actor_id)
            if actor is None:
                continue
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            yaw = actor.get_transform().rotation.yaw

            group_id = self._vehicle_to_group.get(int(actor_id), -1)
            one_obs = self.obs.get(actor_id, None)
            
            self._logger.log_step(
                frame=frame_id,
                timestep=int(self._time_step),
                actor_id=int(actor_id),
                group_id=int(group_id),
                transform=transform,
                velocity=velocity,
                yaw=float(yaw),
                obs=one_obs
            )
        
        reward, reward_info = 0, {} # self.reward()
        info = {}
        # terminated, terminal_conds = self._is_terminal()
        terminated = truncated

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
    