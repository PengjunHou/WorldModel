from utils import PARSER, init_log
from env import make_env
from controller.PPO import PPO, RolloutBuffer, save_checkpoint, load_checkpoint, ModelConfig

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

    master()
    # else:
    #     slave()

def evaluation(evaluation_maps, max_k, strategy):
    file = f"{config_args.strategy}_{max_k}_{strategy}.csv"
    os.makedirs(config_args.result_path, exist_ok=True)
    result_file = os.path.join(config_args.result_path, file)

    lengths = len(evaluation_maps)
    results = []
    for time_step in range(lengths):
        cur_confidence_value = 0
        for agent_i in range(config_args.num_vehicles):
            maps = None
            if strategy == 'Single':
                maps = evaluation_maps[time_step][0][agent_i+1]['local_map']
            else:
                maps = evaluation_maps[time_step][0][agent_i+1]['cur_fused_map']

            print(f"car {agent_i + 1} cur fused map sum 7: {np.sum(maps)}")
            cur_confidence_value += np.sum(maps)
        results.append(cur_confidence_value)
    
    # 将result写到文件中
    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_step", "sum_confidence"])
        for t, val in enumerate(results):
            writer.writerow([t, f"{val:.6f}"])

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
        score_map, action, logp, value = env.action_select(states_tuple, max_k=config_args.max_bandwidth_slices)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated  # 判断是否结束
        step += 1

        # for evaluation
        if obs is not None:
            # print(f"obs len {len(obs)}, keys {obs[0].keys()}")
            evaluation_maps.append(copy.deepcopy(obs))

    # TODO: add some baselines
    evaluation(evaluation_maps, config_args.max_bandwidth_slices, config_args.strategy)

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