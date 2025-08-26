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

    if config_args.load_model:
        state = load_checkpoint(env.RL_agent.model, env.RL_agent.opt,
                        ckpt_path=config_args.model_checkpt,   # 目录即可，会取 latest.pt
                        resume_rng=True,
                        dataclass_type=ModelConfig)  # 返回 dataclass
        loaded_cfg = state["cfg"]  # 可能是 ModelConfig 或 dict

    is_best_model = False
    best_model_reward = -np.inf
    for it in range(iters):
        is_best_model = False
        obs, info = env.reset()  
        buffer = RolloutBuffer(env.time_step_length, (env.RL_cfg.patch_h, env.RL_cfg.patch_w), env.RL_device)
        step = 0
        done = False
        while not done:
            local_maps, adj, fused_maps, prev_b, curr_b = env.wrapper_state(obs)   # 获取状态特征和邻接矩阵
            states_tuple = (local_maps, adj, fused_maps, prev_b, curr_b)
            score_map, action, logp, value = env.action_select(states_tuple)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated  # 判断是否结束
            buffer.add(states_tuple, score_map, logp, value, reward, done)

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
        env.sumary_writer.add_scalar('gae_mean', info['gae_mean'], it)

        print(f"Iter {it}: loss={info['loss']:.4f} policy={info['policy_loss']:.4f} value={info['value_loss']:.4f} ent={info['entropy']:.2f} KL={info['approx_kl']:.4f} gae_mean={info['gae_mean']:.4f} gae_std={info['gae_std']:.4f} gae_nonzero={info['gae_nonzero']:.4f} ratio_mean={info['ratio_mean']:.4f} ratio_std={info['ratio_std']:.4f} ratio_min={info['ratio_min']:.4f} ratio_max={info['ratio_max']:.4f} ratio_close1={info['ratio_close1']:.4f} alpha_beta_sum_mean={info['alpha_beta_sum_mean']:.4f}")

        if config_args.save_model:
            if info['gae_mean'] > best_model_reward:
                is_best_model = True
                best_model_reward = info['gae_mean']
            ckpt = save_checkpoint(env.RL_agent.model, env.RL_agent.opt, env.RL_agent.cfg, out_dir=config_args.model_checkpt,
                       step=it, is_best=is_best_model, extra={"train_info": info})
            print("model saved: ", ckpt)


    print("Training loop finished.")
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