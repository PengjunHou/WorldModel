from envs.CarlaVehEnv.Sensor import Lidar, Camera
from envs.CarlaVehEnv.CarlaDataCollector import V2XSimReader
from envs.CarlaVehEnv.Visualize import visualize_step
import numpy as np
import matplotlib.pyplot as plt
import logging
LOG = logging.getLogger(__name__)

class Object():
    def __init__(self, oid, dataset, token, visualize = False):
        self.oid = oid
        self.token = token
        self.dataset : V2XSimReader = dataset
        self.visualize = visualize
        self.setup()

    def setup(self):
        self.trajectory = self.dataset.get_object_metadata(self.token)
        self.sensor_data = self.dataset.get_object_data(self.token)
        self.type = self.get_type()
        if self.visualize:
            self.fig, self.axs = plt.subplots(4, 6, figsize=(15, 10))

    def get_importance(self, time_step):
        return self.importance
    
    def set_importance(self, importance):
        self.importance = importance

    def get_location(self, time_step):
        location = self.trajectory[time_step]['position']
        return location
    

    def get_rotation(self, time_step):
        rotation = self.trajectory[time_step]['rotation']
        return rotation
    
    def get_speed(self, time_step):
        speed = self.trajectory[time_step]['velocity']
        return speed
    

    def get_type(self):
        name = self.dataset.get_type(self.token)
        return name

    
    def get_state(self, time_step):
        LOG.info(f"time step {time_step}, trajectory {self.trajectory}")
        metadata = self.trajectory[time_step]
        sensor_data = self.sensor_data[time_step]
        state = None        # TODO:这里需要结合
        return state 
    
    def step(self, time_step):
        # update the object state
    
        sensor_data = self.sensor_data[time_step]
        if self.visualize:
            pass    
