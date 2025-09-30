from utils import PARSER, init_log
from env import make_env
from controller.PPO import PPO, RolloutBuffer, save_checkpoint, load_checkpoint, ModelConfig
from evaluation import *

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, sys, csv
import copy
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

    evaluation_maps = master()
    return evaluation_maps
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

    evaluation_maps = []
    obs, info = env.reset()  
    step = 0
    done = False
    while not done:
        local_maps, adj, fused_maps, prev_b, curr_b = env.wrapper_state(obs)
        states_tuple = (local_maps, adj, fused_maps, prev_b, curr_b)
        max_k = curr_b[0][0]
        #print(f"current max time slices cnt {max_k.item()}, {type(max_k)}")
        score_map, action, logp, value = env.action_select(states_tuple, max_k=int(max_k.item()))
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated  # 判断是否结束
        step += 1

        # for evaluation
        if obs is not None:
            # #print(f"obs len {len(obs)}, keys {obs[0].keys()}")
            evaluation_maps.append(copy.deepcopy(obs))

    plt.close('all')
    return evaluation_maps


if __name__ == '__main__':
    
    args = PARSER.parse_args()
    level_str = "info"
    init_log(level_str)
    #print(args)
    #print(args.env_name)
    #print(args.seed)

    # multi-processing
    # from mpi4py import MPI
    # if "parent" == mpi_fork(args.controller_num_worker+1): os.exit()

    # main(args)
    strategies = ['Single', 'Random', 'RL', 'Greedy']
    # for strategy in strategies:
    #     args.strategy = strategy
    #     evaluation_maps = main(args)
    #     get_perf_strategy(config_args, evaluation_maps, -1, config_args.strategy, out_path=None)

    root_path = os.path.join(args.result_path, "evaluation")
    out_path = os.path.join(args.result_path, "evaluation", f"eval_benchmark_at_time")
    eval_banchmark_at_time(args, root_path=root_path, strategies=strategies, K = -1, out_path=out_path)