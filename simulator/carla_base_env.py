from abc import abstractmethod
from typing import Dict, Tuple, Any, Optional

import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# from .toolkit import EnvMonitorOpenCV, Observer, WorldManager

from .carla_manager import WorldManager
from .visualize import EnvVisualBase 
from .observer import Observer
from .network import NetworkBase


class CarlaBaseEnv(gym.Env):
    # Gymnasium 推荐定义 metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config):
        super().__init__() # 初始化父类
        self._config = config

        self._monitor = EnvVisualBase(self._config)
        self._world = WorldManager(self._config)
        self._world.on_reset(self.on_reset)
        self._world.on_step(self.on_step)
        # self._observer = Observer(self._world, self._config.observation)
        self._observers = {}   # TODO
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        
        
    @abstractmethod
    def on_reset(self) -> None:
        """
        Override this method to perform additional reset operations.
        Specifically, you can spawn actors and plan routes here.
        """
        pass

    @abstractmethod
    def apply_control(self, action) -> None:
        """
        Override this method to apply control to actors.
        This method will be called before the simulator ticks.
        """
        pass

    @abstractmethod
    def on_step(self) -> None:
        """
        Override this method to perform additional operations at each step.
        Specifically, you can update the planner and the route here.
        This method will be called after the simulator ticks.
        """
        pass

    @abstractmethod
    def reward(self) -> Tuple[float, Dict]:
        """
        Override this method to define the reward function.
        """
        pass

    @abstractmethod
    def get_terminal_conditions(self) -> Dict[str, bool]:
        """
        Override this method to define the terminal condition.
        If one of the keys in the returned dictionary gives True, the episode will be terminated.
        """
        pass

    def get_ego_vehicle(self) -> carla.Actor:
        """
        Override this method to return the ego vehicle.
        The default behavior is to return self.ego
        """
        return self.ego

    def get_state(self) -> Dict:
        """Return the environment state. Implement this method to define the env state."""
        # obtain state, different algorithm corresponding different state
        # set state in the policy
        # return self._state
        return {}

    def _get_action_space(self):
        action_config = self._config.action
        if action_config.discrete:
            self.n_steer = len(action_config.discrete_steer)
            self.n_acc = len(action_config.discrete_acc)
            return spaces.Discrete(self.n_steer * self.n_acc)
        else:
            return spaces.Box(
                low=action_config.continuous_whittle_index[0],
                high=action_config.continuous_whittle_index[1],
                shape = action_config.shape,
                dtype=np.float32,
            )

    def _get_observation_space(self):
        # assert len(self._observers) > 0, f"No observer, check the logic!"
        # for k, v in self._observers.items():
        #     return self._observers[k].get_observation_space()
        return spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)

    # 修改: reset 签名必须包含 seed 和 options
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # must call super().reset to set seeding
        print("[CARLA] Environment reset")
        super().reset(seed=seed)

        for observer in self._observers.values():
            observer.destroy()
            self._observers = {}
        self._world.reset()
        
        for actor_id, observer in self._observers.items():
            actor = self._world.actor_dict[actor_id]
            if actor is None:
                print(f"[CARLA] Warning: Actor id {actor_id} not found in world during reset.")
                continue
            observer.reset(actor)
        
        self._time_step = 0
        
        self.obs, obs_info = {} , {}
        for actor_id, observer in self._observers.items():
            one_obs, one_obs_info = observer.get_observation(self.get_state())    
            self.obs.setdefault(actor_id, one_obs)
            obs_info.setdefault(actor_id, one_obs_info)

        return (self.obs, obs_info)

    def get_vehicle_control(self, action):
        """
        Convert actions in the action space to vehicle control in CARLA
        """
        action_config = self._config.action
        # Calculate acceleration and steering
        if action_config.discrete:
            acc = action_config.discrete_acc[action // self.n_steer]
            steer = action_config.discrete_steer[action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]
        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    def _is_terminal(self):
        terminal_conds = self.get_terminal_conditions()
        terminal = False
        for k, v in terminal_conds.items():
            if v:
                print(f"[CARLA] Terminal condition triggered: {k}")
                terminal = True
            # 确保转换为 bool 的 numpy array 或 scalar
            terminal_conds[k] = np.array([v], dtype=np.bool_)
            
        if terminal:
            terminal_conds["episode_timesteps"] = self._time_step
        
        # 注意：这里我们只返回 "terminated" (任务结束) 状态
        # "truncated" (超时) 需要在 step 中额外判断
        terminal_conds["terminal"] = terminal
        return terminal, terminal_conds

    def step(self, action):
        self.apply_control(action)
        self._world.step()
        self._time_step += 1
        self.obs, reward, terminated, truncated, info = None, None, None, None, None
        # env_state = self.get_state()
        
        # # 修改: 获取 terminated
        # terminated, terminal_conds = self._is_terminal()
        
        # 修改: 定义 truncated (截断)
        # 通常这里用于处理 TimeLimit。如果没有最大步数限制，设为 False。
        # 如果 self._config 中有 max_episode_steps，可以在这里判断：
        # truncated = self._time_step >= self._config.max_episode_steps
        truncated = False 

        # self.obs, obs_info = self._observer.get_observation(env_state)
        self.obs, obs_info = {} , {}
        for actor_id, observer in self._observers.items():
            one_obs, one_obs_info = observer.get_observation(self.get_state())    
            self.obs.setdefault(actor_id, one_obs)
            obs_info.setdefault(actor_id, one_obs_info)
        # reward, reward_info = self.reward()
        info = {"test": "value"}  # 占位符，请根据实际信息实现
        # info = {
        #     **env_state,
        #     **terminal_conds,
        #     **obs_info,
        #     **reward_info,
        #     "action": action,
        # }
        # if self._config.eval:
        #     info = {f"eval_{k}": v for k, v in info.items()}
        #     # 注意: Gymnasium 不建议修改 obs 为 dict 混合体，
        #     # 但如果你在这里做了特殊处理，请确保 observation_space 匹配。
        #     # 这里暂时保留你原本的逻辑，但要注意 info 才是放元数据的地方。
        #     if isinstance(self.obs, dict):
        #          self.obs.update(info)
            
        # if self._config.display.enable:
        #     self._render(self.obs, info)

        # 修改: 返回 5 个值 (obs, reward, terminated, truncated, info)
        return self.obs, reward, terminated, truncated, info

    def is_collision(self):
        """
        Check if the ego vehicle is in collision.
        You must include 'collsion' in observation.names to use this method.
        """
        return self.obs["collision"][0] > 0

    def _render(self, obs, info):
        self._monitor.render(obs, info)