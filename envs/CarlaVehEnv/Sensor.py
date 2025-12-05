

class Sensor():
    def __init__(self, sensor_type, sensor_id, type_id):
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        self.type_id = type_id
        self.position = (0, 0, 0)
        self.angle = (0, 0, 0)
        self.data = None
        self.data_size = 0
        self.agent_id = -1
        self.data_path = None

    # 从数据集中提取数据
    def get_data(self, time_step):
        # Return the data from the sensor
        return self.data

    def get_state(self):
        '''
        Get the state of the sensor as part of the vehicle's state
        '''
        pass

    def get_feature(self):
        '''
        Excute the feature extraction process to get the feature of the sensor data
        return self.data
        '''
        return self.data
    
    def get_data_size(self):
        '''
        Get the size of the data
        return self.data_size or the size of the feature
        '''
        return self.data_size

class Lidar(Sensor):
    def __init__(self, sensor_id, type_id):
        super().__init__('Lidar', sensor_id, type_id)
        self.data_path = None   # TODO: add the data path
    
    def set_position(self, type_id, position = None):
        # Set the data for the sensor, not usually called
        if position is not None:
            self.position = position
        else:
            default_position = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
            # Set the position based on the type_id
            if type_id < len(default_position):
                self.position = default_position[type_id]
            else:
                raise ValueError(f"Unknown type_id: {type_id}")
        return self.position

    def set_angle(self, type_id, angle = None):
        # Set the data for the sensor, not usually called
        if angle is not None:
            self.angle = angle
        else:
            default_angle = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
            # Set the angle based on the type_id
            if type_id < len(default_angle):
                self.angle = default_angle[type_id]
            else:
                raise ValueError(f"Unknown type_id: {type_id}")
        return self.angle
        

    def get_feature(self):
        '''
        Excute the feature extraction process to get the feature of the sensor data
        return self.data
        '''
        return self.data
    
    def get_data_size(self):
        '''
        Get the size of the data
        return self.data_size or the size of the feature
        '''
        return self.data_size
    
class Camera(Sensor):
    def __init__(self, sensor_id, type_id):
        super().__init__('Camera', sensor_id, type_id)
        self.data_path = None   # TODO: add the data path
    
    def set_position(self, type_id, position = None):
        # Set the data for the sensor, not usually called
        if position is not None:
            self.position = position
        else:
            default_position = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
            # Set the position based on the type_id
            if type_id < len(default_position):
                self.position = default_position[type_id]
            else:
                raise ValueError(f"Unknown type_id: {type_id}")
        return self.position
    
    def set_angle(self, type_id, angle = None):
        # Set the data for the sensor, not usually called
        if angle is not None:
            self.angle = angle
        else:
            default_angle = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
            # Set the angle based on the type_id
            if type_id < len(default_angle):
                self.angle = default_angle[type_id]
            else:
                raise ValueError(f"Unknown type_id: {type_id}")
        return self.angle

    def get_feature(self):
        '''
        Excute the feature extraction process to get the feature of the sensor data
        return self.data
        '''
        return self.data
    
    def get_data_size(self):
        '''
        Get the size of the data
        return self.data_size or the size of the feature
        '''
        return self.data_size
    
class GPS(Sensor):
    def __init__(self, sensor_id, type_id):
        super().__init__('GPS', sensor_id, type_id)
        self.data_path = None   # TODO: add the data path

    def set_position(self, type_id, position = None):
        # Set the data for the sensor, not usually called
        if position is not None:
            self.position = position
        else:
            default_position = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
            # Set the position based on the type_id
            if type_id < len(default_position):
                self.position = default_position[type_id]
            else:
                raise ValueError(f"Unknown type_id: {type_id}")
        return self.position
    
    def set_angle(self, type_id, angle = None):
        # Set the data for the sensor, not usually called
        if angle is not None:
            self.angle = angle
        else:
            default_angle = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
            # Set the angle based on the type_id
            if type_id < len(default_angle):
                self.angle = default_angle[type_id]
            else:
                raise ValueError(f"Unknown type_id: {type_id}")
        return self.angle

    def get_feature(self):
        '''
        Excute the feature extraction process to get the feature of the sensor data
        return self.data
        '''
        return self.data
    
    def get_data_size(self):
        '''
        Get the size of the data
        return self.data_size or the size of the feature
        '''
        return self.data_size
    
    
class GNSS(Sensor):
    def __init__(self, sensor_id, type_id):
        super().__init__('GNSS', sensor_id, type_id)
        self.data_path = None
    
    def set_position(self, type_id, position = None):
        # Set the data for the sensor, not usually called
        if position is not None:
            self.position = position
        else:
            default_position = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
            # Set the position based on the type_id
            if type_id < len(default_position):
                self.position = default_position[type_id]
            else:
                raise ValueError(f"Unknown type_id: {type_id}")
        return self.position
    
    def set_angle(self, type_id, angle = None):
        # Set the data for the sensor, not usually called
        if angle is not None:
            self.angle = angle
        else:
            default_angle = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
            # Set the angle based on the type_id
            if type_id < len(default_angle):
                self.angle = default_angle[type_id]
            else:
                raise ValueError(f"Unknown type_id: {type_id}")
        return self.angle

    def get_feature(self):
        '''
        Excute the feature extraction process to get the feature of the sensor data
        return self.data
        '''
        return self.data
    
    def get_data_size(self):
        '''
        Get the size of the data
        return self.data_size or the size of the feature
        '''
        return self.data_size
    
