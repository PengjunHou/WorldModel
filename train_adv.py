from utils import PARSER, init_log
from env import make_env
from controller.PPO import PPO, RolloutBuffer, save_checkpoint, load_checkpoint, ModelConfig
from envs.CarlaVehSensors import CarlaEnv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, sys
import torch
matplotlib.use('Agg')  

os.environ['OMP_NUM_THREADS'] = '1'
BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEPATH)  


@dataclass
class RolloutBatch:
    state_Ks: torch.Tensor         # [T, cond_steps, obs_dim]
    state_local_maps: torch.Tensor  # [T, cond_steps, C, H, W]
    state_fused_maps: torch.Tensor  # [T, cond_steps, C, H, W]
    chains: torch.Tensor        # [T, K+1, horizon, action]
    logprobs: torch.Tensor      # [T, K, horizon, action]
    values: torch.Tensor        # [T]
    returns: torch.Tensor       # [T]
    advantages: torch.Tensor    # [T]

@dataclass
class BCDataset:
    state_local_maps: torch.Tensor  # [N, C, H, W]
    state_fused_maps: torch.Tensor  # [N, C, H, W]
    state_Ks: torch.Tensor  # [N, 2]
    actions: torch.Tensor  # [N, H, W]  # score maps
    masks: torch.Tensor   # [N]


def heuristic_policy(obs: np.ndarray) -> int:
    position, velocity, angle, angular_vel = obs
    score = angle + 0.5 * angular_vel + 0.05 * velocity + 0.01 * position
    return 1 if score > 0 else 0

def diffusion_supervised_loss(
    model,
    actions: torch.Tensor,
    cond: dict,
    timesteps: torch.Tensor,
    predict_epsilon: bool,
) -> torch.Tensor:
    noise = torch.randn_like(actions, device=actions.device)
    noisy = model.q_sample(x_start=actions, t=timesteps, noise=noise)
    if predict_epsilon:
        pred = model.actor_ft(noisy, timesteps, cond)
        target = noise
    else:
        pred = model.actor_ft(noisy, timesteps, cond)
        target = actions
    target = target.view(pred.shape[0], 1, -1)
    return torch.nn.functional.mse_loss(pred, target)

def collect_bc_dataset(
    env,
    policy,
    num_samples: int,
) -> BCDataset:
    samples_local_maps = []
    samples_fused_maps = []
    samples_Ks = []
    actions = []
    masks = []
    obs, _ = env.reset()
    steps = 0
    total_reward = 0.0
    while len(samples_local_maps) < num_samples:
        local_maps, adj, fused_maps, prev_b, curr_b = env.wrapper_state(obs)    # local_maps:[B,N,1,H,W], adj:[B,N,N], fused_maps:[B,N, 1, H,W], prev_b:[B,1], curr_b:[B,1]
        #print(f"local_maps shape: {local_maps.shape}, adj shape: {adj.shape}, fused_maps shape: {fused_maps.shape}, prev_b shape: {prev_b.shape}, curr_b shape: {curr_b.shape}")
        states_tuple = (local_maps, adj, fused_maps, prev_b, curr_b)
        max_k = curr_b[0][0]
        score_map, action, logp, value = env.action_select(states_tuple, max_k=int(max_k.item()), collection_policy = True)   # act = policy_fn(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        samples_local_maps.append(local_maps)
        samples_fused_maps.append(fused_maps)
        samples_Ks.append(torch.stack([prev_b, curr_b], dim = -1))
        actions.append(score_map)
        keep = 1 # TODO float(steps < 150 and total_reward >= -10.0)
        masks.append(torch.tensor(keep, dtype=torch.bool))
        total_reward += reward
        steps += 1
        if terminated or truncated:
            steps = 0
            total_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs
    state_local_maps = torch.cat(samples_local_maps, dim=0)
    state_fused_maps = torch.cat(samples_fused_maps, dim=0)
    state_Ks = torch.cat(samples_Ks, dim=0).squeeze(1)
    actions = torch.cat(actions, dim=0)
    masks = torch.stack(masks, dim=0)
    #print(f"collected bc dataset size {state_local_maps.shape}, {state_fused_maps.shape}, {state_Ks.shape}, {actions.shape}, {masks.shape}")
    return BCDataset(state_local_maps = state_local_maps, state_fused_maps = state_fused_maps, state_Ks = state_Ks,  actions=actions, masks=masks)


def collect_rollout(
    model,
    env: CarlaEnv,
    obs: np.ndarray,
    steps: int,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) :
    samples_local_maps = []
    samples_fused_maps = []
    samples_Ks = []
    chains = []
    logprobs = []
    rewards = []
    dones = []
    values = []

    for _ in range(steps):
        cond = {}
        local_maps, adj, fused_maps, prev_b, curr_b = env.wrapper_state(obs) 
        state_Ks = torch.stack([curr_b, prev_b], dim = -1)  # [B, 5, 2]
        cond = {"state_local_maps": local_maps, "state_fused_maps": fused_maps, "state_Ks": state_Ks }
        #print(f"cond shapes: state_local_maps {local_maps.shape}, state_fused_maps {fused_maps.shape}, state_Ks {state_Ks.shape}")
        with torch.no_grad():
            samples = model(cond=cond, deterministic=False, return_chain=True)
            #print(f"samples.trajectories shape: {samples.trajectories.shape}, samples.chains shape: {samples.chains.shape}")
            value = model.critic(cond).squeeze().item()
            chain_tensor = samples.chains
            logprob = model.get_logprobs(cond, samples.chains, get_ent=False)
            #print(f"logprob shape: {logprob.shape}, chain_tensor shape: {chain_tensor.shape}")
            logprob = logprob.view(model.ft_denoising_steps, model.horizon_steps, model.action_dim)
            #print(f"logprob reshaped to: {logprob.shape}")
            score_map = samples.trajectories[0, 0].view(-1, env.RL_agent.cfg.action_H, env.RL_agent.cfg.action_W)

        action_value = env.score_map2action(score_map, local_maps, int(curr_b[0][0].item()))   
        if score_map.shape[0] == 1:
            action_value = action_value[0]
        next_obs, reward, terminated, truncated, _ = env.step(action_value)
        done = terminated or truncated
        samples_Ks.append(torch.stack([prev_b, curr_b], dim = -1))
        samples_local_maps.append(local_maps)
        samples_fused_maps.append(fused_maps)
        chains.append(chain_tensor)
        logprobs.append(logprob)
        rewards.append(reward)
        dones.append(done)
        values.append(value)

        obs = next_obs if not done else env.reset()[0]

    # with torch.no_grad():
    #     final_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    #     cond_final = {"state": final_state.unsqueeze(1)}
    #     next_value = model.critic(cond_final).squeeze().item()
    # if dones[-1]:
    assert dones[-1] == True, f"final done flag should be True, got {dones[-1]}"
    next_value = 0.0

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
    values_t = torch.tensor(values, dtype=torch.float32, device=device)
    advantages = torch.zeros_like(rewards_t, device=device)
    returns = torch.zeros_like(rewards_t, device=device)

    gae = 0.0
    next_val = next_value
    for step in reversed(range(steps)):
        mask = 1.0 - dones_t[step]
        delta = rewards_t[step] + gamma * next_val * mask - values_t[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
        returns[step] = gae + values_t[step]
        next_val = values_t[step]

    batch = RolloutBatch(
        state_local_maps=torch.cat(samples_local_maps, dim=0),
        state_fused_maps=torch.cat(samples_fused_maps, dim=0),
        state_Ks=torch.cat(samples_Ks, dim=0).squeeze(1),
        chains=torch.cat(chains, dim = 0),
        logprobs=torch.stack(logprobs),
        values=values_t,
        returns=returns,
        advantages=advantages,
    )
    return batch, obs

def initialize_settings(args):
    pass

def main(args):
    global config_args
    config_args = args

    initialize_settings(args)

    master()
    # else:
    #     slave()

def evaluate_diffusion(env, times = 5):
    steps = 0
    total_reward = 0.0
    obs, _ = env.reset()
    res_r = []

    while steps < times:
        local_maps, adj, fused_maps, prev_b, curr_b = env.wrapper_state(obs)  
        states_tuple = (local_maps, adj, fused_maps, prev_b, curr_b)
        max_k = curr_b[0][0]
        score_map, action, logp, value = env.action_select(states_tuple, max_k=int(max_k.item()), collection_policy = False)   # act = policy_fn(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            steps += 1
            res_r.append(total_reward)
            total_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs
    print(f"evaluate diffusion pretrain: {res_r}")


def master():
    iters = config_args.epoch
    plt.close('all')
    env : CarlaEnv = make_env(args=config_args, dream_env=False, render_mode=False)    # build agent inside env

    if config_args.load_model:
        env.RL_agent.load_model(config_args.model_checkpt)

    # BC pretrain
    if config_args.bc_steps > 0 and config_args.bc_samples > 0:
        dataset = collect_bc_dataset(env, True, config_args.bc_samples)
        # env.close()
        state_local_maps = dataset.state_local_maps.to(config_args.device)
        state_fused_maps = dataset.state_fused_maps.to(config_args.device)
        state_Ks = dataset.state_Ks.to(config_args.device)
        actions = dataset.actions.to(config_args.device)
        env.RL_agent.model.train()

    for step in range(config_args.bc_steps):
        idx = torch.randint(0, state_local_maps.size(0), (env.RL_agent.batch_size,), device=env.RL_agent.device)
        cond = {"state_local_maps": state_local_maps[idx], "state_fused_maps": state_fused_maps[idx], "state_Ks": state_Ks[idx] }
        timesteps = torch.randint(0, env.RL_agent.model.denoising_steps, (env.RL_agent.batch_size,), device=env.RL_agent.device)
        filtered_actions = actions[idx]
        loss = diffusion_supervised_loss(
            env.RL_agent.model, filtered_actions, cond, timesteps, predict_epsilon = True
        )
        env.RL_agent.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(env.RL_agent.model.actor_ft.parameters(), 1.0)
        env.RL_agent.actor_opt.step()
        if (step + 1) % 10 == 0:
            print(f"[bc] step={step + 1:04d} loss={loss.item():.4f}")

    evaluate_diffusion(env, times=5)
    
    best_model_reward = -np.inf
    for it in range(iters):
        print(f"--------------------------- iter: {it}/{iters} --------------------------------------")
        is_best_model = False
        obs, info = env.reset()  
        rollout, obs = collect_rollout(env.RL_agent.model, env, obs, env.time_step_length, env.RL_device, env.RL_agent.cfg.gamma, env.RL_agent.cfg.gae_lambda)
        info = env.RL_agent.ppo_update(
            rollout,
            epochs=config_args.epoch,  
            minibatch_size=env.RL_agent.batch_size,
            ent_coef=env.RL_agent.cfg.ent_coef,
            vf_coef=env.RL_agent.cfg.vf_coef,
            update_actor=True)
        
        total_rewards = rollout.returns.sum()
        if total_rewards > best_model_reward:
            env.RL_agent.save_model(is_best = True)
            best_model_reward = total_rewards
        
        if it % 10 == 0:
            env.RL_agent.save_model(it)
            total_rewards = rollout.returns.sum()

        env.RL_agent.itr += 1
        # env.sumary_writer.add_scalar('loss', info['loss'], it)
        # env.sumary_writer.add_scalar('policy_loss', info['policy_loss'], it)
        # env.sumary_writer.add_scalar('value_loss', info['value_loss'], it)
        # env.sumary_writer.add_scalar('entropy', info['entropy'], it)
        # env.sumary_writer.add_scalar('gae_mean', info['gae_mean'], it)
        env.sumary_writer.add_scalar('reward', total_rewards.item(), it)

        # #print(f"Iter {it}: loss={info['loss']:.4f} policy={info['policy_loss']:.4f} value={info['value_loss']:.4f} ent={info['entropy']:.2f} KL={info['approx_kl']:.4f} gae_mean={info['gae_mean']:.4f} gae_std={info['gae_std']:.4f} gae_nonzero={info['gae_nonzero']:.4f} ratio_mean={info['ratio_mean']:.4f} ratio_std={info['ratio_std']:.4f} ratio_min={info['ratio_min']:.4f} ratio_max={info['ratio_max']:.4f} ratio_close1={info['ratio_close1']:.4f} alpha_beta_sum_mean={info['alpha_beta_sum_mean']:.4f}")

    evaluate_diffusion(env, times=5)

    print("Training loop finished.")
    env.sumary_writer.close()

if __name__ == '__main__':
    
    args = PARSER.parse_args()
    level_str = "info"
    init_log(level_str)
    #print(args)
    #print(args.env_name)
    #print(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # multi-processing
    # from mpi4py import MPI
    # if "parent" == mpi_fork(args.controller_num_worker+1): os.exit()

    main(args)