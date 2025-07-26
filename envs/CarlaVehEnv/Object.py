

class Objects():
    def __init__(self, name):
        self.name = name
        self.obj_type = obj_type
        self.location = location
        self.rotation = rotation
        self.importance = importance

    def get_importance(self):
        return self.importance
    
    def set_importance(self, importance):
        self.importance = importance

    def get_location(self):
        return self.location

    def get_rotation(self):
        return self.rotation

    def get_name(self):
        return self.name

    def get_type(self):
        return self.obj_type
    
    def step(self, time_step, action):
        # update the object state
        pass

    
    
