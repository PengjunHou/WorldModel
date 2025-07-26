#import carla
import gym
from envs.CarlaVehEnv.Car import Car, Clusters
from envs.CarlaVehEnv.Object import Object
from envs.CarlaVehEnv.CarlaDataCollector import V2XSimReader
from envs.CarlaVehEnv.RSU import RSU
from envs.CarlaVehEnv.Object import Object


import logging
LOG = logging.getLogger(__name__)


# 1. Carla-Gym Environment Wrapper
class CarlaEnv(gym.Env):
    def __init__(self, config, full_episode=True, with_obs=True, load_model=True):
        super(CarlaEnv, self).__init__()
        # Connect to Carla
        # self.client = carla.Client(config['host'], config['port'])
        # self.world = self.client.load_world(config['town'])

        ## Configuration
        self.config = config
        self.data_path = config.data_path
        # Initialize sensors, vehicles, and other components
        self._setup_simulation()       

        ## RL
        self.action_space = gym.spaces.MultiDiscrete([self.n_vehicles] * 24)  # 0 or 1 for each vehicle
        obs_dim = config.obs_dim
        self.observation_space = gym.spaces.Box(-1e6, 1e6, shape=(obs_dim,), dtype=float)

    def _setup_simulation(self):
        # Load the dataset
        self.dataset = V2XSimReader(self.data_path)
        self.cur_time_step = 0
        time_step_length = self.dataset.get_time_step_length()
        config_time_step_length = self.config.time_steps
        self.time_step_length = config_time_step_length if config_time_step_length < time_step_length else time_step_length

        self.create_cars()
        self.create_rsu()
        self.create_objects()
    
        self.n_clusters = self.config.n_clusters
        self._init_cluster()
        
    
    def _init_cluster(self, stratergy = None):
        # Initialize clusters based on the strategy
        # spawn vehicles, attach sensors
        self.clusters = []
        for i in range(self.n_clusters):
            cluster = Clusters(i, leader_id=1, vehicles=self.cars)
            self.clusters.append(cluster)

    def create_cars(self):
        self.n_vehicles = self.dataset.get_vehicle_count()
        self.cars = {}
        veh_sensors = self.dataset.get_vehicle_sensors()

        assert self.n_vehicles == len(veh_sensors), "Number of vehicles and sensors do not match"

        for vid, sensor_info in veh_sensors.items():
            # Create a vehicle object
            car = Car(vid, dataset=self.dataset, sensors=sensor_info)
            self.cars[vid] = car
        
        LOG.info(f"create cars finish! Number of cars : {self.n_vehicles}")
        
    def create_rsu(self):
        self.n_rsu = self.dataset.get_rsu_count()
        self.rsu = {}
        rsu_sensors = self.dataset.get_rsu_sensors()

        assert self.n_rsu == len(rsu_sensors), f"Number of RSUs {self.n_rsu} and sensors do not match {len(rsu_sensors)}"

        for rid, sensor_info in rsu_sensors.items():
            # Create a RSU object
            rsu = RSU(rid, dataset=self.dataset, sensors=sensor_info)
            self.rsu[rid] = rsu
        
        LOG.info(f"create rsus finish! Number of rus : {self.n_rsu}")

    def create_objects(self):
        obj_tokens = self.dataset.get_object_tokens()
        oid = 0
        self.objects = {}

        for token in obj_tokens:
            if len(self.dataset.get_object_data(token)) < self.time_step_length:
                continue
            obj = Object(oid, dataset=self.dataset, token=token)
            self.objects[oid] = obj
            oid += 1

        self.n_objects = oid

        LOG.info(f"create objects finish! Number of objects : {self.n_objects}")


    def reset(self):
        # reset world and return initial observation
        self.cur_time_step = 0
        self._init_cluster()
        return self._get_observation(time_step=0)

    def step(self, action):
        # apply vehicle controls or data-upload decisions
        next_time_step = self.cur_time_step + 1

        if next_time_step >= self.time_step_length:
            done = True 
            obs = None
        else:
            done = False
            obs = self._get_observation(next_time_step)

        # for rid, rsu in self.rsu.items():
        #     rsu.step(next_time_step, action)
        for cluster in self.clusters:
            cluster.step(next_time_step, action)
        # for oid, obj in self.objects.items():
        #     obj.step(next_time_step, action)

        reward = self._compute_reward()
        info = {}

        LOG.debug(f"Time step {self.cur_time_step}, action {action}, reward {reward}, done {done}, info {info}")

        self.cur_time_step = next_time_step

        return obs, reward, done, info

    def _get_observation(self, time_step):
        # gather per-vehicle states
        cluster_states = []
        object_states = []
        for oid, obj in self.objects.items():
            object_states.append(obj.get_state(time_step))
        for cluster in self.clusters:
            cluster_states.append(cluster.get_state(time_step))
        
        states = None #self._combine_states(cluster_states, object_states)
        return states

    def _compute_reward(self):
        # define reward for RL
        return 0.0
    