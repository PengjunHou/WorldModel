
from abc import abstractmethod
from typing import Dict, Tuple, Any, Optional

class BaseAgent:
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step

    @abstractmethod
    def policy(self, obs, state=None, mode="train") -> Tuple[Dict, Any]:
        pass

    @abstractmethod
    def learn(self, experience):
        pass