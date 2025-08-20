from utils import PARSER, init_log
from env import make_env
from controller.PPO import PPO, RolloutBuffer

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
    iters = 100
    plt.close('all')
    env = make_env(args=config_args, dream_env=False, render_mode=False)
    for it in range(iters):
        obs, info = env.reset()  
        buffer = RolloutBuffer(env.time_step_length, (env.RL_cfg.patch_h, env.RL_cfg.patch_w), env.RL_device)
        step = 0
        done = False
        while not done:
            feature, adj = env.wrapper_state(obs)   # 获取状态特征和邻接矩阵
            score_map, action, logp, value = env.action_select(feature, adj)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated  # 判断是否结束
            buffer.add((feature, adj), score_map, logp, value, reward, done)

            # 显示图像
            # plt.imshow(obs_img.astype(np.uint8))
            # plt.title(f"Step {step}, Reward: {reward:.2f}")
            # plt.axis('off')
            # plt.pause(0.01)
            step += 1
        batch = buffer.stack()
        info = env.RL_agent.update(batch, get_new_logp_fn=None)
        env.sumary_writer.add_scalar('loss', info['loss'], it)
        env.sumary_writer.add_scalar('policy_loss', info['policy_loss'], it)
        env.sumary_writer.add_scalar('value_loss', info['value_loss'], it)
        env.sumary_writer.add_scalar('entropy', info['entropy'], it)

        print(f"Iter {it}: loss={info['loss']:.4f} policy={info['policy_loss']:.4f} value={info['value_loss']:.4f} ent={info['entropy']:.2f} KL={info['approx_kl']:.4f}")

    print("Training loop finished (toy demo). Replace DummyFleetEnv.step with your evaluator.")
    env.sumary_writer.close()

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