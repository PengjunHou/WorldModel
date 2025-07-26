from WorldModel.envs.CarlaVehEnv.Sensor import Lidar, Camera

class Car():
    def __init__(self, vid, sensors = None, carla_vehicle = None):
        self.carla_vehicle = carla_vehicle
        self.cluster_id = -1
        self.vid = vid
        self.time_step = -1
        self.speed = 0
        self.position = (0, 0, 0)
        self.rotation = (0, 0, 0)
        self.sensors = {}
        self.sensors_data = {}
        
        self.setup(sensors)
    
    def setup(self, sensors):
        '''
        Setup the sensors for the vehicle
        sensors: a dict of sensor objects (sensor_type: n_sensors)
        default: Camera: 6, Lidar: 2
        '''
        assert isinstance(sensors, dict), "sensors should be a dictionary"
        sensor_id = 0
        for sensor_type, n_sensors in sensors.items():
            assert sensor_type in ['Lidar', 'Camera'], "Unknown sensor type"
            if sensor_type == 'Lidar':
                for i in range(n_sensors):
                    sensor_id += 1
                    sensor = Lidar(sensor_id = sensor_id, type_id = i)
                    sensor.set_position((0, 0, 0))
                    sensor.set_angle((0, 0, 0))
                    sensor.data_path = None
                    self.sensors[sensor.sensor_id] = sensor
            elif sensor_type == 'Camera':
                for i in range(n_sensors):
                    sensor_id += 1
                    sensor = Camera(sensor_id = sensor_id, type_id = i)
                    self.sensors[sensor.sensor_id] = sensor
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")

    # 从数据集中提取数据
    def get_speed(self, time_step):
        '''
        TODO: get speed from the carla vehicle or sensor or  file
        Get the speed of the vehicle
        '''
        return self.speed

    def get_position(self, time_step):
        '''
        TODO: get position from the carla vehicle or sensor or  file
        Get the position of the vehicle
        '''
        return self.position
    

    def get_rotation(self, time_step):
        '''
        TODO: get rotation from the carla vehicle or sensor or  file
        Get the rotation of the vehicle
        '''
        return self.rotation
    

    def get_state(self):
        '''
        TODO: What dose the state of the vehicle comprise of?
        Get the state of the vehicle
        '''
        metadata = {
            'vid': self.vid,
            'cluster_id': self.cluster_id,
            'position': self.position,
            'rotation': self.rotation,
            'speed': self.speed
        }

        # Get the state of the sensors
        sensor_states = {}
        for sensor_id, sensor in self.sensors.items():
            sensor_states[sensor_id] = sensor.get_state()
        
        # Combine the metadata and sensor states
        state = self.combine_states(metadata, sensor_states)    # TODO: apply RNN to the state
        return state

    
    def get_sensor_data(self, sensor_type, type_id):
        # Return the data from a specific sensor
        sensor = self.get_sensor(sensor_type, type_id)
        if sensor:
            return sensor.get_data()
        else:
            raise ValueError(f"Sensor {type_id} of type {sensor_type} not found")
    
    def join_group(self, cluster_id):
        # Join a group of vehicles
        self.cluster_id = cluster_id

    def leave_group(self):
        # Leave the current group of vehicles
        self.cluster_id = None

    
    def get_sensor(self, sensor_type, sensor_id):
        # Return a dictionary of the car's sensors
        # This could include cameras, LIDAR, etc.
        pass

    def update_metadata(self, time_step):
        # Update the metadata of the vehicle
        assert time_step > self.time_step, "time_step should be greater than the previous time_step"
        self.time_step = time_step
        self.speed = self.get_speed(time_step)
        self.position = self.get_position(time_step)
        self.rotation = self.get_rotation(time_step)


    def apply_control_upload(self, time_step, action):
        # Apply control to the vehicle based on the action,action is a numpy array composed of 0 or 1
        # indicate whether to upload the data or not
        self.update_metadata(time_step)

        for sensor_id, sensor in self.sensors.items():
            if action[sensor_id] == 1:
                # Upload the data from the sensor
                data = sensor.get_data(time_step)   # TODO: also get the feature of the data
                self.sensors_data[sensor_id] = data


    def receive_data(self, state):
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
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.members : list[Car] = []
        self.leader = None

    def add_member(self, member):
        self.members.append(member)

    def remove_member(self, member):
        if member in self.members:
            self.members.remove(member)
        else:
            raise ValueError("Member not found in the cluster")
    
    def get_members(self):
        return self.members
    
    def set_leader(self, leader):
        self.leader = leader
        leader.join_group(self.cluster_id)

    def get_leader(self):
        return self.leader
    
    def get_state(self, time_step, action):
        # Return the state of the cluster
        cluster_state = {}
        for member in self.members:
            member_state = member.get_state()
            cluster_state[member.vid] = member_state
        
        states = None
        # states = GNN(cluster_state)     # TODO
        return states
    
    def get_member_state(self, member_id, time_step, action):
        # Return the states of all members in the cluster
        member_states = [member.get_state() for member in self.members]
        return member_states
    
    def step(self, time_step, action):
        # Update the state of the cluster and its members
        for member in self.members:
            member.apply_control_upload(time_step, action[member.vid])
            
    

    



    


        
 