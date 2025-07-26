from utils import PARSER, init_log
import env
from env import make_env

import matplotlib.pyplot as plt
import numpy as np
import os, sys

os.environ['OMP_NUM_THREADS'] = '1'
BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEPATH)  


def initialize_settings(args):
    pass


def main(args):
    global config_args
    config_args = args

    initialize_settings(args)

    master()
    # else:
    #     slave()

def master():
    plt.close('all')
    env = make_env(args=config_args, dream_env=False, render_mode=False)
    obs = env.reset()  
    step = 0
    done = False
    while not done:
        action = np.ones((6, 24), dtype = "float32") # env.action_space.sample()
        obs_img, reward, terminated, _ = env.step(action)
        done = terminated  # 判断是否结束

        # 显示图像
        # plt.imshow(obs_img.astype(np.uint8))
        # plt.title(f"Step {step}, Reward: {reward:.2f}")
        # plt.axis('off')
        # plt.pause(0.01)
        step += 1

if __name__ == '__main__':
    
    args = PARSER.parse_args()
    level_str = "debug"
    init_log(level_str)
    print(args)
    print(args.env_name)
    print(args.seed)
    print(args.em_model)
    print(args.pred_model)
    print(args.ctrl_model)

    # multi-processing
    # from mpi4py import MPI
    # if "parent" == mpi_fork(args.controller_num_worker+1): os.exit()

    main(args)