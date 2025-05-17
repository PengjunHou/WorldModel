#import carla
import gym
from CarlaVehEnv.Car import Car, Clusters
from CarlaVehEnv.Object import Objects
from CarlaVehEnv.CarlaDataCollector import V2XSimReader
#from CarlaVehEnv.RSU import RSU

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

        self.preprocess_dataset(self.data_path)       # TODO: 处理数据集，得到车辆数/Object数/RSU数/Sensors数

        self.n_clusters = config.n_clusters
        self.n_vehicles = config.n_vehicles
        self.n_rsu = config.n_rsu
        self.n_clusters = config.n_clusters
        self.time_steps = config.time_steps
        self.cur_time_step = -1
        self.time_step_length = config.time_step_length
        self.clusters = []
        self.objects = []

        ## RL
        self.action_space = gym.spaces.Discrete(config.action_dim)
        obs_dim = config.obs_dim
        self.observation_space = gym.spaces.Box(-1e6, 1e6, shape=(obs_dim,), dtype=float)

        # Initialize sensors, vehicles, and other components
        self._setup_simulation()

    def preprocess_dataset(self, data_path):
        # Preprocess the dataset
        # self.dataset = Dataset(self.data_path)
        # self.dataset.preprocess()
        self.dataset = V2XSimReader(self.data_path)
        self.n_vehicles = self.dataset.get_vehicle_count()
        self.n_rsu = self.dataset.get_rsu_count()
        self.time_step_length = self.dataset.get_time_step_length()
        self.sensors = self.dataset.get_sensors()
        self.n_sensors = len(self.sensors)
        self.objects = self.dataset.get_objects()

    def _setup_simulation(self):
        # spawn vehicles, attach sensors
        for i in range(self.n_clusters):
            cluster = Clusters(i, self.n_vehicles, self.sensors)
            self.clusters.append(cluster)
        
        # RSU
        # self.rsu = RSU(self.n_rsu, self.sensors)

        # Object
        self.objects.append[Objects(self.n_clusters, self.n_vehicles, self.sensors)]
        

    def reset(self):
        # reset world and return initial observation
        return self._get_observation()

    def step(self, action):
        # apply vehicle controls or data-upload decisions
        self.cur_time_step += 1
        if self.cur_time_step >= self.time_steps:
            done = True
        else:
            done = False

        # for rsu in self.rsu:
        #     rsu.step(self.cur_time_step, action)
        for cluster in self.clusters:
            cluster.step(self.cur_time_step, action)
        for obj in self.objects:
            obj.step(self.cur_time_step, action)

        obs = self._get_observation(self.cur_time_step)
        reward = self._compute_reward()
        info = {}
        return obs, reward, done, info

    def _get_observation(self, time_step, action):
        # gather per-vehicle states
        cluster_states = []
        object_states = []
        for obj in self.objects:
            object_states.append(obj.get_state(time_step, action))
        for cluster in self.clusters:
            cluster_states.append(cluster.get_state(time_step, action))
        
        states = self._combine_states(cluster_states, object_states)
        return states

    def _compute_reward(self):
        # define reward for RL
        return 0.0