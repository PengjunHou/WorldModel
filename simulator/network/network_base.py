
from ..carla_manager import WorldManager

class NetworkBase:
    def __init__(self, world: WorldManager, config):
        self._world = world
        self._comm_config = config
        
    
    def setvehcomm(self, actor):
        pass
    
    def reset(self):
        pass
    
    