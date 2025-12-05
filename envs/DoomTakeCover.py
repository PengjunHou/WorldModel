
import numpy as np
import random
from PIL import Image
import utils

from gym.spaces.box import Box
from ppaquette_gym_doom.doom_take_cover import DoomTakeCoverEnv
from gym.utils import seeding


class DoomTakeCoverWrapper(DoomTakeCoverEnv):
    '''
    Wrapper for gym CarRacing environment that returns expected outputs
    '''
    def __init__(self, full_episode=False):
        super(DoomTakeCoverWrapper, self).__init__()

    def _process_frame(self, frame):
        pass

    def _step(self, action):
        obs, reward, done, _ = super(DoomTakeCoverWrapper, self)._step(action)
        return obs, reward, done, {}


class DoomTakeCoverEnv(DoomTakeCoverWrapper):
    def __init__(self, args, render_mode=False, load_model=True, with_obs=False):
        super(DoomTakeCoverEnv, self).__init__()

        self.with_obs = with_obs

        self.no_render = True
        if render_mode:
            self.no_render = False
        self.current_obs = None

        self.embedding = utils.load_em_model(args.em_model)
        self.pred = utils.load_pred_model(args.pred_model)
        
        if load_model:
            self.embedding.set_weights()
            self.pred.set_weights()



    def close(self):
        super(DoomTakeCoverEnv, self).close()


    def _step(self, action):
        # update states of pred
        self.frame_count += 1
    
        self.pred_states = self.pred.next_state(self.pred, self.z, action, self.pred_states) 

        obs, reward, done, _ = super(DoomTakeCoverEnv, self)._step(action)

        if done:
            self.restart = 1
        else:
            self.restart = 0

        if self.with_obs:
            return [self._current_state(), self.current_obs], reward, done, {}
        else:
            return self._current_state(), reward, done, {}

    def _encode(self, img):
        z = self.embedding.encode(img)
        return z

    def _reset(self):
        obs = super(DoomTakeCoverEnv, self)._reset()
        small_obs = self._process_frame(obs)
        self.current_obs = small_obs
        self.pred_states = self.pred.init_state(self.pred)
        self.z = self._encode(small_obs)
        self.restart = 1
        self.frame_count = 0

        if self.with_obs:
            return [self._current_state(), self.current_obs]
        else:
            return self._current_state()

    def _process_frame(self, frame):
        obs = frame[0:400, :, :]
        obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
        obs = np.array(obs)
        return obs

    def _current_state(self):
        pass

    def _seed(self, seed=None):
        # set seed
        return [seed]


'''
# code from https://github.com/zacwellmer/WorldModels

class DoomTakeCoverMDNRNN(DoomTakeCoverWrapper):
  def __init__(self, args, render_mode=False, load_model=True, with_obs=False):
    super(DoomTakeCoverMDNRNN, self).__init__()

    self.with_obs = with_obs

    self.no_render = True
    if render_mode:
      self.no_render = False
    self.current_obs = None

    self.embedding = CVAE(args)
    self.pred = MDNRNN(args)

    if load_model:
      self.embedding.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
      self.pred.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])

    self.action_space = Box(low=-1.0, high=1.0, shape=())
    self.obs_size = self.pred.args.z_size + self.pred.args.rnn_size * self.pred.args.state_space

    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3))
    self.actual_observation_space = Box(low=-50., high=50., shape=(self.obs_size))

    self._seed()

    self.pred_states = None
    self.z = None
    self.restart = None
    self.frame_count = None
    self.viewer = None
    self._reset()

  def close(self):
    super(DoomTakeCoverMDNRNN, self).close()
    tf.keras.backend.clear_session()
    gc.collect()

  def _step(self, action):

    # update states of pred
    self.frame_count += 1
   
    self.pred_states = rnn_next_state(self.pred, self.z, action, self.pred_states) 

    # actual action in wrapped env:

    threshold = 0.3333
    full_action = [0] * 43

    if action < -threshold:
      full_action[11] =1

    if action > threshold:
      full_action[10] = 1

    obs, reward, done, _ = super(DoomTakeCoverMDNRNN, self)._step(full_action)
    small_obs = self._process_frame(obs)
    self.current_obs = small_obs
    self.z = self._encode(small_obs)

    if done:
      self.restart = 1
    else:
      self.restart = 0

    if self.with_obs:
      return [self._current_state(), self.current_obs], reward, done, {}
    else:
      return self._current_state(), reward, done, {}

  def _encode(self, img):
    simple_obs = np.copy(img).astype(np.float)/255.0
    simple_obs = simple_obs.reshape(1, 64, 64, 3)
    z = self.embedding.encode(simple_obs)[0]
    return z

  def _reset(self):
    obs = super(DoomTakeCoverMDNRNN, self)._reset()
    small_obs = self._process_frame(obs)
    self.current_obs = small_obs
    self.pred_states = rnn_init_state(self.pred)
    self.z = self._encode(small_obs)
    self.restart = 1
    self.frame_count = 0

    if self.with_obs:
      return [self._current_state(), self.current_obs]
    else:
      return self._current_state()

  def _process_frame(self, frame):
    obs = frame[0:400, :, :]
    obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
    obs = np.array(obs)
    return obs

  def _current_state(self):
    if self.pred.args.state_space == 2:
      return np.concatenate([self.z, tf.keras.backend.flatten(self.pred_states[1]), tf.keras.backend.flatten(self.pred_states[0])], axis=0) # cell then hidden fro some reason
    return np.concatenate([self.z, tf.keras.backend.flatten(self.pred_states[0])], axis=0) # only the hidden state

  def _seed(self, seed=None):
    if seed:
      tf.random.set_seed(seed)
    self.np_random, seed = seeding.np_random(seed)
    return [seed]'
    
'''