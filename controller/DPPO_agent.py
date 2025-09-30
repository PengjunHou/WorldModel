import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import logging
# import wandb
import random
from torch.utils.tensorboard import SummaryWriter
from controller.cfg.dppo_cfg import DPPOConfig
from controller.model.diffusion_ppo import PPODiffusion
# from controller.model.scheduler import CosineAnnealingWarmupRestarts
log = logging.getLogger(__name__)

class DPPO:

    def __init__(self, cfg: DPPOConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.tensorboard_writer = SummaryWriter(log_dir=cfg.summary_dir)

        # Make vectorized env
        self.env_name = cfg.env_name
        self.checkpoint_dir = cfg.checkpoint_dir

        # Batch size for gradient update
        self.batch_size: int = cfg.batch_size

        # Build model and load checkpoint
        self.model = PPODiffusion(gamma_denoising=cfg.gamma_denoising,
                                  clip_ploss_coef=cfg.clip_ploss_coef,
                                  clip_ploss_coef_base=cfg.clip_ploss_coef_base,
                                  clip_ploss_coef_rate=cfg.clip_ploss_coef_rate,
                                  actor = cfg.actor,
                                  critic = cfg.critic,
                                  ft_denoising_steps = cfg.ft_denoising_steps,
                                  min_sampling_denoising_std = cfg.min_sampling_denoising_std,
                                  min_logprob_denoising_std = cfg.min_logprob_denoising_std,
                                  eta = cfg.eta,
                                  learn_eta = cfg.learn_eta,
                                  horizon_steps = cfg.horizon_steps,
                                  obs_dim = cfg.obs_dim,
                                  action_dim = cfg.action_dim,
                                  device = cfg.device,
                                  randn_clip_value = cfg.randn_clip_value, 
                                  ddim_steps = cfg.ddim_steps,
                                  use_ddim = cfg.use_ddim,
                                  denosing_steps = cfg.denoising_steps,)

        self.actor_opt = torch.optim.Adam(self.model.actor_ft.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.model.critic.parameters(), lr=cfg.critic_lr)

        # Optimizer
        # self.actor_optimizer = torch.optim.AdamW(
        #     self.model.actor_ft.parameters(),
        #     lr=cfg.train.actor_lr,
        #     weight_decay=cfg.train.actor_weight_decay,
        # )
        # # use cosine scheduler with linear warmup
        # self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
        #     self.actor_optimizer,
        #     first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
        #     cycle_mult=1.0,
        #     max_lr=cfg.train.actor_lr,
        #     min_lr=cfg.train.actor_lr_scheduler.min_lr,
        #     warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
        #     gamma=1.0,
        # )
        # self.critic_optimizer = torch.optim.AdamW(
        #     self.model.critic.parameters(),
        #     lr=cfg.train.critic_lr,
        #     weight_decay=cfg.train.critic_weight_decay,
        # )
        # self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
        #     self.critic_optimizer,
        #     first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
        #     cycle_mult=1.0,
        #     max_lr=cfg.train.critic_lr,
        #     min_lr=cfg.train.critic_lr_scheduler.min_lr,
        #     warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
        #     gamma=1.0,
        # )

        # Training params
        self.itr = 0
        self.n_train_itr = cfg.n_train_itr
        self.val_freq = cfg.val_freq
        self.force_train = False
        self.n_steps = cfg.n_steps
        self.max_grad_norm = None

        # Logging, rendering, checkpoints
        # self.logdir = cfg.logdir
        # self.render_dir = os.path.join(self.logdir, "render")
        # self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        # self.result_path = os.path.join(self.logdir, "result.pkl")
        # os.makedirs(self.render_dir, exist_ok=True)
        # os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_trajs =  False
        self.log_freq = 1
        self.save_model_freq = cfg.save_model_freq
        # self.render_freq = cfg.render.freq
        # self.n_render = cfg.render.num
        self.render_video = False
        # assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        # assert not (
        #     self.n_render <= 0 and self.render_video
        # ), "Need to set n_render > 0 if saving video"

    def ppo_update(
        self,
        batch,
        epochs: int,
        minibatch_size: int,
        ent_coef: float,
        vf_coef: float,
        update_actor: bool,
    ) -> None:
        self.model.train()
        ft_steps = self.model.ft_denoising_steps
        total_steps = batch.state_Ks.size(0)
        # batch.state = batch.state.to(self.model.device)
        # batch.chains = batch.chains.to(self.model.device)
        # batch.logprobs = batch.logprobs.to(self.model.device)
        # batch.values = batch.values.to(self.model.device)
        # batch.returns = batch.returns.to(self.model.device)
        # batch.advantages = batch.advantages.to(self.model.device)

        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            perm = torch.randperm(total_steps, device=self.model.device)
            for start in range(0, total_steps, minibatch_size):
                idx = perm[start : start + minibatch_size]
                obs_b = {"state_Ks": batch.state_Ks[idx], "state_local_maps": batch.state_local_maps[idx], "state_fused_maps": batch.state_fused_maps[idx]}
                denoising_inds = torch.randint(0, ft_steps, (idx.size(0),), device=self.model.device)

                chains_b = batch.chains[idx] 
                #print(f"shape of chains_b: {chains_b.shape}")  # [B, ft_steps+1, horizon, action_dim]
                #print(f"denoising_inds: {denoising_inds.cpu().numpy()}")  # [B]
                di = denoising_inds.view(-1, 1, 1, 1)
                chains_prev = torch.gather(chains_b, 1, di.expand(-1, 1, 1, self.cfg.action_H*self.cfg.action_W)).squeeze(1)
                chains_next = torch.gather(chains_b, 1, (di+1).view(-1,1,1,1).expand(-1,1,1,self.cfg.action_H*self.cfg.action_W)).squeeze(1)

                # chains_prev = batch.chains[idx, denoising_inds]
                # chains_next = batch.chains[idx, denoising_inds + 1]
                returns_b = batch.returns[idx]
                values_b = batch.values[idx]
                adv_b = advantages[idx]
                oldlogprobs_b = batch.logprobs[idx, denoising_inds]
                ##print("idx", idx)
                ##print("denoising_inds", denoising_inds)
                #print(f"shape of obs_b[state_Ks]: {obs_b['state_Ks'].shape}, shape of chains_prev: {chains_prev.shape}, shape of chains_next: {chains_next.shape}, shape of denoising_inds: {denoising_inds.shape}, shape of returns_b: {returns_b.shape}, shape of values_b: {values_b.shape}, shape of adv_b: {adv_b.shape}, shape of oldlogprobs_b: {oldlogprobs_b.shape}")

                (
                    pg_loss,
                    entropy_loss,
                    v_loss,
                    _,
                    approx_kl,
                    _,
                    bc_loss,
                    _,
                ) = self.model.loss(
                    obs_b,
                    chains_prev,
                    chains_next,
                    denoising_inds,
                    returns_b,
                    values_b,
                    adv_b,
                    oldlogprobs_b,
                    use_bc_loss=False,
                    reward_horizon=1,
                )
                loss = pg_loss + entropy_loss * ent_coef + v_loss * vf_coef + bc_loss

                #self.actor_optimizer.zero_grad(set_to_none=True)
                #self.critic_optimizer.zero_grad(set_to_none=True)
                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 1.0)
                if update_actor:
                    self.actor_opt.step()
                self.critic_opt.step()

                self.tensorboard_writer.add_scalar("loss", loss.mean().item(), self.itr)
                self.tensorboard_writer.add_scalar("policy_loss", pg_loss.mean().item(), self.itr)
                self.tensorboard_writer.add_scalar("value_loss", v_loss.mean().item(), self.itr)
                self.tensorboard_writer.add_scalar("entropy", entropy_loss.mean().item(), self.itr)
                self.tensorboard_writer.add_scalar("approx_kl", approx_kl, self.itr)
                self.tensorboard_writer.add_scalar("bc_loss", bc_loss, self.itr)

                print(f"loss: {loss.mean().item()}, pg_loss: {pg_loss.mean().item()}, v_loss: {v_loss.mean().item()}, entropy: {entropy_loss.mean().item()}, approx_kl: {approx_kl}, bc_loss: {bc_loss}")
                
                if approx_kl > 0.2:  # crude early-stop safeguard
                    return
                
    def action_select(self, states):
        self.model.eval()
        local_maps, adjs, fused_maps, prev_b, curr_b = states
        # #print(f"shapes in action_select: local_maps {local_maps.shape}, adjs {adjs.shape}, fused_maps {fused_maps.shape}, prev_b {prev_b.shape}, curr_b {curr_b.shape}")
        state_local_maps = local_maps.squeeze(2)  # [1, N, H, W]
        state_fused_maps = fused_maps.squeeze(2)  # [1, N, H, W]
        state_Ks = torch.stack([prev_b, curr_b], axis=-1) # [1, N, 2]
        # #print(f"shapes after tensor: state_local_maps {state_local_maps.shape}, state_fused_maps {state_fused_maps.shape}, state_Ks {state_Ks.shape}")
        cond = {"state_local_maps": state_local_maps, "state_fused_maps": state_fused_maps, "state_Ks": state_Ks }

        with torch.no_grad():
            samples = self.model(
                cond=cond,
                deterministic=False,
                return_chain=True,
            )
            # #print(f"samples.trajectories shape: {samples.trajectories.shape}")
            # #print(f"samples.chains shape: {samples.chains.shape}")
            # #print(f"samples: {samples}")
        return samples.trajectories[0, 0].view(-1, self.cfg.action_H, self.cfg.action_W), 0, 0
    
    def save_model(self, is_best=False):
        """
        saves model to disk; no ema
        """
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
        }  # right now `model` includes weights for `network`, `actor`, `actor_ft`. Weights for `network` is redundant, and we can use `actor` weights as the base policy (earlier denoising steps) and `actor_ft` weights as the fine-tuned policy (later denoising steps) during evaluation.
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, savepath)
        if is_best:
            savepath = os.path.join(self.checkpoint_dir, f"best_state.pt")
            torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")

    def load(self, itr = -1):
        """
        loads model from disk
        """
        if itr == -1:
            # load latest
            loadpath = os.path.join(self.checkpoint_dir, f"best_state.pt")
        else:
            loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
        data = torch.load(loadpath, weights_only=True)

        self.itr = data["itr"]
        self.model.load_state_dict(data["model"])

    def run(self):

        # Start training loop
        # timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            # firsts_trajs 用来标记是不是对应step是不是trajectory的第一步，
            # 后续切分 episode：用 firsts_trajs 中值为 1 的索引来找每个 env 的 episode [start, end) 区间，统计回报/成功率；
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # Holder
            # obs_trajs[k][step] 第 step 时刻用于决策的条件观测（而不是这一步环境返回的新观测）。k 是观测类型键（如 "rgb", "state"）。
            obs_trajs = {
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dims[k])
                )
                for k in self.obs_dims
            }
            # 扩散采样出的整条“去噪链”（从噪声到动作的多步序列）。(n_envs, ft_denoising_steps + 1, horizon_steps, action_dim)。
            # PPO 在这里把每个去噪步当作“一个决策”，要计算新旧策略在这些步上的 log-prob（以及 KL、clipratio）
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            ) 
            # 这一时间步是否自然终止（terminated，不是 truncated），形状 (n_envs,)。
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            # 这一步环境返回的即时奖励 r_t，形状 (n_envs,)。
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(prev_obs_venv[key])
                        .float()
                        .to(self.device)
                        for key in self.obs_dims
                    }  # batch each type of obs and put into dict
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )  # n_env x horizon x act
                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                for k in obs_trajs:
                    obs_trajs[k][step] = prev_obs_venv[k]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            if not eval_mode:
                with torch.no_grad():
                    # apply image randomization
                    obs_trajs["rgb"] = (
                        torch.from_numpy(obs_trajs["rgb"]).float().to(self.device)
                    )
                    obs_trajs["state"] = (
                        torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                    )
                    if self.augment:
                        rgb = einops.rearrange(
                            obs_trajs["rgb"],
                            "s e t c h w -> (s e t) c h w",
                        )
                        rgb = self.aug(rgb)
                        obs_trajs["rgb"] = einops.rearrange(
                            rgb,
                            "(s e t) c h w -> s e t c h w",
                            s=self.n_steps,
                            e=self.n_envs,
                        )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(
                        self.n_envs * self.n_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    for k in obs_trajs:
                        obs_k = einops.rearrange(
                            obs_trajs[k],
                            "s e ... -> (s e) ...",
                        )
                        obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                        for i, obs_t in enumerate(obs_ts_k):
                            obs_ts[i][k] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = (
                            self.model.critic(obs, no_augment=True)
                            .cpu()
                            .numpy()
                            .flatten()
                        )
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        )

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    obs_venv_ts = {
                        key: torch.from_numpy(obs_venv[key]).float().to(self.device)
                        for key in self.obs_dims
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                self.model.critic(obs_venv_ts, no_augment=True)
                                .reshape(1, -1)
                                .cpu()
                                .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # k for environment step
                obs_k = {
                    k: einops.rearrange(
                        obs_trajs[k],
                        "s e ... -> (s e) ...",
                    )
                    for k in obs_trajs
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs, device=self.device).float(),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = (
                    torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                )
                values_k = (
                    torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                )
                advantages_k = (
                    torch.tensor(advantages_trajs, device=self.device).float().reshape(-1)
                )
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                clipfracs = []
                for update_epoch in range(self.update_epochs):

                    # for each epoch, go through all data in batches
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )
                        obs_b = {k: obs_k[k][batch_inds_b] for k in obs_k}
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        loss.backward()
                        if (batch + 1) % self.grad_accumulate == 0:
                            if self.itr >= self.n_critic_warmup_itr:
                                if self.max_grad_norm is not None:
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.actor_ft.parameters(),
                                        self.max_grad_norm,
                                    )
                                self.actor_optimizer.step()
                                if (
                                    self.learn_eta
                                    and batch % self.eta_update_interval == 0
                                ):
                                    self.eta_optimizer.step()
                            self.critic_optimizer.step()
                            self.actor_optimizer.zero_grad()
                            self.critic_optimizer.zero_grad()
                            if self.learn_eta:
                                self.eta_optimizer.zero_grad()
                            log.info(f"run grad update at batch {batch}")
                            log.info(
                                f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                            )

                            # Stop gradient update if KL difference reaches target
                            if (
                                self.target_kl is not None
                                and approx_kl > self.target_kl
                                and self.itr >= self.n_critic_warmup_itr
                            ):
                                flag_break = True
                                break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0][
                                    "lr"
                                ],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1


    # def save_model(self):
    #     """save_model
    #     saves model to disk; no ema
    #     """
    #     data = {
    #         "itr": self.itr,
    #         "model": self.model.state_dict(),
    #     }  # right now `model` includes weights for `network`, `actor`, `actor_ft`. Weights for `network` is redundant, and we can use `actor` weights as the base policy (earlier denoising steps) and `actor_ft` weights as the fine-tuned policy (later denoising steps) during evaluation.
    #     savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
    #     torch.save(data, savepath)
    #     log.info(f"Saved model to {savepath}")

    # def load(self, itr):
    #     """
    #     loads model from disk
    #     """
    #     loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
    #     data = torch.load(loadpath, weights_only=True)

    #     self.itr = data["itr"]
    #     self.model.load_state_dict(data["model"])

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # convert to OrderedDict if obs_venv is a list of dict
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
    

if __name__ == "__main__":
    config = {'env_name': 'CarlaVehSensors-v0',}
    cfg = DPPOConfig(config)
    agent = DPPO(cfg)
    #print(agent.model)

