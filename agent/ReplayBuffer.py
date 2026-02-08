from collections import deque
import random

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, q_maps, actions, reward, next_state, done):
        self.buffer.append({
            'state': {k: v.clone().detach() for k, v in state.items()},
            'q_maps': q_maps.clone().detach(),
            'actions': actions.clone().detach(),
            'reward': float(reward),
            'next_state': {k: v.clone().detach() for k, v in next_state.items()},
            'done': done
        })
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)