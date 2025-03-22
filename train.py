from utils import PARSER
from env import make_env

import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['OMP_NUM_THREADS'] = '1'


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
    env = make_env(args=config_args, dream_env=False, render_mode=False)
    obs = env.reset()  
    step = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs_img, reward, terminated, truncated, _ = env.step(action)  # 适配新版本的 step() 返回值
        done = terminated or truncated  # 判断是否结束

        # 显示图像
        plt.imshow(obs_img.astype(np.uint8))
        plt.title(f"Step {step}, Reward: {reward:.2f}")
        plt.axis('off')
        plt.pause(0.01)
        step += 1

if __name__ == '__main__':
    
    args = PARSER.parse_args()
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