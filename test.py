from utils import PARSER, init_log
from env import make_env
from controller.PPO import PPO, RolloutBuffer, save_checkpoint, load_checkpoint, ModelConfig

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, sys
matplotlib.use('Agg')  

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
    iters = config_args.epoch
    plt.close('all')
    env = make_env(args=config_args, dream_env=False, render_mode=False)

    assert config_args.load_model == 1, f"load model setting false"
    state = load_checkpoint(env.RL_agent.model, env.RL_agent.opt,
                    ckpt_path=config_args.model_checkpt,   # 目录即可，会取 latest.pt
                    resume_rng=True,
                    dataclass_type=ModelConfig)  # 返回 dataclass
    loaded_cfg = state["cfg"]  # 可能是 ModelConfig 或 dict
    env.RL_agent.model.eval()

    obs, info = env.reset()  
    step = 0
    done = False
    while not done:
        local_maps, adj, fused_maps, prev_b, curr_b = env.wrapper_state(obs)
        states_tuple = (local_maps, adj, fused_maps, prev_b, curr_b)
        score_map, action, logp, value = env.action_select(states_tuple)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated  # 判断是否结束
        step += 1

    # TODO: add some baselines
    


    plt.close('all')


if __name__ == '__main__':
    
    args = PARSER.parse_args()
    level_str = "info"
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