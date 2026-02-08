import random
from .BaseAgent import BaseAgent

import numpy as np


class RandomAgent(BaseAgent):
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step

    def policy(self, obs, state=None, mode="train"):
        batch_size = len(next(iter(obs.values())))

        if self.actor_dist_disc != "twohot":
            act = {k: np.stack([v.sample() for _ in range(batch_size)]) for k, v in self.act_space.items() if k != "reset"}
            return act, state
