import numpy as np
import random
# from envs.DoomTakeCover import DoomTakeCoverEnv
from envs.CarRacing import CarRacingEnv
# from envs.DreamDoomTakeCover import DreamDoomTakeCoverEnv




def make_env(args, dream_env=False, seed=-1, render_mode=False, full_episode=False, with_obs=False, load_model=False):
    if args.env_name == 'DoomTakeCover':
        if dream_env:
            print('making rnn doom environment')
            env = DreamDoomTakeCoverEnv(args=args, render_mode=render_mode, load_model=load_model)
        else:
            print('making real doom environment')
            env = DoomTakeCoverEnv(args=args, render_mode=render_mode, load_model=load_model, with_obs=with_obs)
    if args.env_name == 'CarRacing-v2':
        if dream_env:
            raise ValueError('training in dreams for carracing is not yet supported')
        else:
            print('makeing real CarRacing environment')
        env = CarRacingEnv(args=args, full_episode=full_episode, with_obs=with_obs, load_model=load_model)
    if (seed >= 0):
        env.seed(seed)
    return env