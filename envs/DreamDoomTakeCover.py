import numpy as np
import random
from PIL import Image
import utils
import json
import os

from gym.spaces.box import Box
from ppaquette_gym_doom.doom_take_cover import DoomTakeCoverEnv
from gym.utils import seeding



class DreamDoomTakeCoverEnv:
    def __init__(self, args, render_mode=False, load_model=True):
        self.render_mode = render_mode
        model_path_name = 'results/{}/{}'.format(args.exp_name, args.env_name)
        with open(os.path.join(model_path_name, 'tf_initial_z/initial_z.json'), 'r') as f:
            [initial_mu, initial_logvar] = json.load(f)

        self.initial_mu_logvar = np.array([list(elem) for elem in zip(initial_mu, initial_logvar)])

        self.embedding = utils.load_em_model(args.em_model)
        self.pred = utils.load_pred_model(args.pred_model)
        
        if load_model:
            self.embedding.set_weights()
            self.pred.set_weights()

        # future versions of OpenAI gym needs a dtype=np.float32 in the next line:
        self.action_space = Box(low=-1.0, high=1.0, shape=())
        obs_size = self.pred.args.z_size + self.pred.args.rnn_size * self.pred.args.state_space
        # future versions of OpenAI gym needs a dtype=np.float32 in the next line:
        self.observation_space = Box(low=-50., high=50., shape=(obs_size,))

        self.pred_states = None
        self.o = None

        self._training=True

        self.seed() 
        self.reset()

    def _sample_init_z(self):
        idx = self.np_random.randint(low=0, high=self.initial_mu_logvar.shape[0])
        init_mu, init_logvar = self.initial_mu_logvar[idx]
        init_mu = init_mu / 10000.0
        init_logvar = init_logvar / 10000.0
        init_z = init_mu + np.exp(init_logvar/2.0) * self.np_random.randn(*init_logvar.shape)
        return init_z

    def reset(self):
        self.pred_states = self.pred.init_state(self.pred)
        z = np.expand_dims(self._sample_init_z(), axis=0)
        self.o = z
        z_ch = tf.concat([z, self.pred_states[1], self.pred_states[0]], axis=-1)
        return tf.squeeze(z_ch)

    def seed(self, seed=None):
        pass

    def step(self, action):
        rnn_states_p1, z_tp1, r_tp1, d_tp1 = self.pred.run_sim(self.pred, self.o, self.pred_states, action, training=self._training)
        self.pred_states = rnn_states_p1
        self.o = z_tp1

        z_ch = tf.squeeze(tf.concat([z_tp1, self.pred_states[1], self.pred_states[0]], axis=-1))    
        return z_ch.numpy(), tf.squeeze(r_tp1), d_tp1.numpy(), {}

    def close(self):
        pass

    def render(self, mode):
        pass