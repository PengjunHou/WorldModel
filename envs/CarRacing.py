

import numpy as np
import random
from PIL import Image
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

import utils


class CarRacingWrapper(CarRacing):
    '''
    Wrapper for gym CarRacing environment that returns expected outputs
    '''
    def __init__(self, full_episode=False):
        super(CarRacingWrapper, self).__init__()
        self.full_episode = full_episode

    def _process_frame(self, frame):
        obs = frame[0:84, :, :]
        obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
        obs = np.array(obs)
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = super(CarRacingWrapper, self).step(action)
        obs = self._process_frame(obs)
        if self.full_episode:
            done = False
        return self._process_frame(obs), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super(CarRacingWrapper, self).reset(seed=seed, options=options)
        obs = self._process_frame(obs)
        return obs

class CarRacingEnv(CarRacingWrapper):
    def __init__(self, args, load_model=True, full_episode=False, with_obs=False):
        super(CarRacingEnv, self).__init__(full_episode=full_episode)
        self.with_obs = with_obs # whether or not to return the frame with the encodings
        self.embedding = utils.load_em_model()(args)
        self.pred = utils.load_pred_model()(args)
        
        if load_model:
            self.embedding.set_weights()
            self.pred.set_weights()
        self.pred_states = self.pred.init_state()
        
        self.full_episode = full_episode 
        # self.observation_space = Box(low=np.NINF, high=np.Inf, shape=(args.z_size+args.rnn_size*args.state_space))
        self.observation_space = Box(low=np.NINF, high=np.Inf, shape=(3,), dtype=np.float32)

    def encode_obs(self, obs):
        z = self.embedding.encode(obs)
        return z
  
    def reset(self, seed=None, options=None):
        obs = super(CarRacingEnv, self).reset(seed=seed, options=options) # calls step
        z_state = obs# self.encode_obs(obs)
        # self.pred_states = self.pred.init_state(self.pred)
        if self.with_obs:
            return [z_state, obs]
        else:
            return z_state
    
    def step(self, action):
        obs, reward, done, truncated, info = super(CarRacingEnv, self).step(action)
        z_state = obs# self.encode_obs(obs)
        # self.pred_states = self.pred.next_state(self.pred, z_state, action, self.pred_states)
        if self.with_obs:
            return [z_state, obs], reward, done, truncated, info
        else:
            return z_state, reward, done, truncated, info
    
    def close(self):
        super(CarRacingEnv, self).close()
        pass

    def seed(self, seed=None):
        pass

'''
# code from https://github.com/zacwellmer/WorldModels
class CarRacingWrapper(CarRacing):
    def __init__(self, full_episode=False):
        super(CarRacingWrapper, self).__init__()
        self.full_episode = full_episode
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

    def _process_frame(self, frame):
        obs = frame[0:84, :, :]
        obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
        obs = np.array(obs)
        return obs

    def step(self, action):
        obs, reward, done, _ = super(CarRacingWrapper, self).step(action)
        if self.full_episode:
            return self._process_frame(obs), reward, False, {}
        return self._process_frame(obs), reward, done, {}


class CarRacingEnv(CarRacingWrapper):
    def __init__(self, args, load_model=True, full_episode=False, with_obs=False):
        super(CarRacingEnv, self).__init__(full_episode=full_episode)
        self.with_obs = with_obs # whether or not to return the frame with the encodings
        self.embedding = CVAE(args)
        self.pred = MDNRNN(args)
        
        if load_model:
            self.embedding.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
            self.pred.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])
        self.pred_states = pred_init_state(self.pred)
        
        self.full_episode = False 
        self.observation_space = Box(low=np.NINF, high=np.Inf, shape=(args.z_size+args.rnn_size*args.state_space))

    def encode_obs(self, obs):
        # convert raw obs to z, mu, logvar
        result = np.copy(obs).astype(np.float)/255.0
        result = result.reshape(1, 64, 64, 3)
        z = self.embedding.encode(result)[0]
        return z
  
    def reset(self):
        self.pred_states = pred_init_state(self.pred)
        if self.with_obs:
            [z_state, obs] = super(CarRacingEnv, self).reset() # calls step
            self.N_tiles = len(self.track)
            return [z_state, obs]
        else:
            z_state = super(CarRacingEnv, self).reset() # calls step
            self.N_tiles = len(self.track)
            return z_state
    
    def step(self, action):
        obs, reward, done, _ = super(CarRacingEnv, self).step(action)
        z = tf.squeeze(self.encode_obs(obs))
        h = tf.squeeze(self.pred_states[0])
        c = tf.squeeze(self.pred_states[1])
        if self.pred.args.state_space == 2:
            z_state = tf.concat([z, c, h], axis=-1)
        else:
            z_state = tf.concat([z, h], axis=-1)
        if action is not None: # don't compute state on reset
            self.pred_states = rnn_next_state(self.pred, z, action, self.pred_states)
        if self.with_obs:
            return [z_state, obs], reward, done, {}
        else:
            return z_state, reward, done, {}
    
    def close(self):
        super(CarRacingEnv, self).close()
        tf.keras.backend.clear_session()
        gc.collect()
'''