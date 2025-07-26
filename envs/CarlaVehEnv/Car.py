from envs.CarlaVehEnv.Sensor import Lidar, Camera
from envs.CarlaVehEnv.CarlaDataCollector import V2XSimReader
from envs.CarlaVehEnv.Visualize import visualize_step
import numpy as np
import logging
LOG = logging.getLogger(__name__)

class Car():
    def __init__(self, vid, dataset, sensors = None, carla_vehicle = None, visualize = True):
        self.carla_vehicle = carla_vehicle
        self.visualize = visualize          # 可视化的数据
        self.dataset : V2XSimReader = dataset
        self.cluster_id = -1
        self.vid = vid
        self.sensor_types = sensors 
        self.sensors = {}               # a dict of sensor objects (sensor_id: sensor)
        self.setup()
        self.upload_data = None         # 根据action决定上传的数据，可能是metadata + sensor data 后的数据
        self.rev_data = None            # 接收的数据
        
    
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



    def apply_control_upload(self, time_step, action):
        # Apply control to the vehicle based on the action,action is a numpy array composed of 0 or 1
        # indicate whether to upload the data or not
        LOG.debug(f"time step {time_step}, vid {self.vid}, cluster {self.cluster_id}, actions {action}")
        metadata = self.update_metadata(time_step)

        self.sensors_data = {}
        for sensor_id, sensor in self.sensors.items():
            if action[sensor_id] > 0:
                # Upload the data from the sensor
                data = self.get_sensor_data(sensor_id, time_step)   
                self.sensors_data[sensor_id] = data
            else:
                # 用之前的数据后者收到的
                self.sensors_data[sensor_id] = None

        if self.visualize:
            imgs = []
            for sensor_id, sensor_data in self.sensors_data.items():
                if sensor_data is not None:
                    img, format = sensor_data['filename'], sensor_data['fileformat']
                    imgs.append(img)
                else:
                    imgs.append(None)

                visualize_step(time_step, imgs, self.dataset.root, self.fig, self.axs)

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

    def get_state(self, time_step):
        '''
        TODO: What dose the state of the vehicle comprise of?
        Get the state of the vehicle
        '''
        # state = self.combine_states()    # TODO: apply RNN to the state
        state = None
        return state

    
    
    def join_group(self, cluster_id):
        # Join a group of vehicles
        self.cluster_id = cluster_id

    def leave_group(self):
        # Leave the current group of vehicles
        self.cluster_id = None


    def receive_data(self, data, time_step):
        pass
    
class CarLeader(Car):
    def __init__(self, carla_vehicle):
        super().__init__(carla_vehicle)
        self.cluster_id = None
        self.vid = None

    def broadcast_data(self):
        # Broadcast the state of the leader vehicle to its members
        for member in self.members:
            member.receive_state(self.data_fusion)

    def receive_data(self, data):
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
        return states
    
    def get_member_state(self, member_id, time_step, action):
        # Return the states of all members in the cluster
        member_states = [member.get_state() for member in self.members]
        return member_states
    
    def step(self, time_step, action):
        # Update the state of the cluster and its members
        for vid, member in self.members.items():
            member.apply_control_upload(time_step, action[member.vid])
            
    

    



    


        
 