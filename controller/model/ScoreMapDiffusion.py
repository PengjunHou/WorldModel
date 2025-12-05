from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F

# 复用原来的工具函数与常量
from controller.model.sampling import extract, cosine_beta_schedule, make_timesteps

Sample = namedtuple("Sample", "trajectories chains")

class ScoreMapDiffusion(nn.Module):
    """
    按照原始 DiffusionModel 的接口与流程，面向 2D score map 的版本。
    - 约定：horizon_steps = 1, action_dim = H*W
    - self.network: VisionUnet2D，接口与 DiffusionModel.network 一致：network(x_t, t, cond) -> noise
      其中 noise 是预测的 ε（predict_epsilon=True）或 x0（predict_epsilon=False），形状 [B, 1, H*W]
    - cond: 透传给 VisionUnet2D（例如包含 x_in/state/rgb/global 等）
    """

    def __init__(
        self,
        network,                 # VisionUnet2D
        H, W,                    # map 尺寸（仅用于注释/理解，不参与数值）
        horizon_steps,           # 必须=1
        obs_dim,                 # 保留接口（未使用）
        action_dim,              # 必须=H*W
        network_path=None,
        device="cuda:0",
        # clipping
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,     # DDIM only
        # DDPM
        denoising_steps=100,
        predict_epsilon=True,
        # DDIM
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        **kwargs,
    ):
        super().__init__()
        assert horizon_steps == 1, "ScoreMapDiffusion 要求 horizon_steps=1（单张 score map）"
        assert action_dim == H * W, f"action_dim 必须等于 H*W, 当前 {action_dim} vs {H*W}"
        self.H, self.W = H, W

        self.device = device
        self.horizon_steps = horizon_steps     # =1
        self.obs_dim = obs_dim
        self.action_dim = action_dim           # =H*W
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon

        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        self.denoised_clip_value = denoised_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.randn_clip_value = randn_clip_value
        self.eps_clip_value = eps_clip_value

        # 网络
        self.network = network.to(device)
        if network_path is not None:
            ckpt = torch.load(network_path, map_location=device, weights_only=True)
            if "ema" in ckpt:
                self.load_state_dict(ckpt["ema"], strict=False)
            else:
                self.load_state_dict(ckpt["model"], strict=False)

        """
        DDPM parameters

        """
        """
        βₜ
        """
        self.betas = cosine_beta_schedule(self.denoising_steps).to(device)
        """
        αₜ = 1 - βₜ
        """
        self.alphas = 1.0 - self.betas
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ 
        """
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        """
        α̅ₜ₋₁
        """        
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]]
        )
        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # 变分项
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        # ===== DDIM 参数（可选；与原实现保持一致）=====
        if use_ddim:
            assert predict_epsilon, "DDIM 目前要求 predict_epsilon=True"
            if ddim_discretize == "uniform":
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = torch.arange(0, ddim_steps, device=device) * step_ratio
            else:
                raise ValueError("Unknown DDIM discretization method")

            self.ddim_alphas = self.alphas_cumprod[self.ddim_t].clone().to(torch.float32)
            self.ddim_alphas_sqrt = torch.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = torch.cat(
                [torch.tensor([1.0], device=device, dtype=torch.float32),
                 self.alphas_cumprod[self.ddim_t[:-1]]]
            )
            self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas) ** 0.5

            ddim_eta = 0.0
            self.ddim_sigmas = (
                ddim_eta
                * (
                    (1 - self.ddim_alphas_prev)
                    / (1 - self.ddim_alphas)
                    * (1 - self.ddim_alphas / self.ddim_alphas_prev)
                ) ** 0.5
            )

            # 反向顺序（与原版一致）
            self.ddim_t = torch.flip(self.ddim_t, [0])
            self.ddim_alphas = torch.flip(self.ddim_alphas, [0])
            self.ddim_alphas_sqrt = torch.flip(self.ddim_alphas_sqrt, [0])
            self.ddim_alphas_prev = torch.flip(self.ddim_alphas_prev, [0])
            self.ddim_sqrt_one_minus_alphas = torch.flip(self.ddim_sqrt_one_minus_alphas, [0])
            self.ddim_sigmas = torch.flip(self.ddim_sigmas, [0])

    # ---------- Sampling ---------- #

    def p_mean_var(self, x, t, cond, index=None, network_override=None):
        """
        与原 DiffusionModel 一致：给定 x_t, t, cond，返回 (mu, logvar)
        - self.network 返回的是噪声 ε（predict_epsilon=True）或 x0（predict_epsilon=False）
        - x, mu, logvar 形状： [B, 1, H*W]
        """
        if network_override is not None:
            noise = network_override(x, t, cond=cond)  # [B, 1, H*W]
        else:
            noise = self.network(x, t, cond=cond)       # [B, 1, H*W]

        if self.predict_epsilon:
            if self.use_ddim:
                # x0 = (x_t - sqrt(1-alpha_t)*eps) / sqrt(alpha_t)
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha ** 0.5)
            else:
                # x0 = sqrt(1/alphabar_t) * x_t - sqrt(1/alphabar_t - 1) * eps
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:
            x_recon = noise

        # 可选：clip x0
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # 重新计算 eps（可选，和原实现保持一致）
                noise = (x - (extract(self.ddim_alphas, index, x.shape) ** 0.5) * x_recon) / \
                       extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)

        # 可选：clip epsilon（仅 DDIM）
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # 计算 mu, logvar
        if self.use_ddim:
            # mu = sqrt(alpha_prev) * x0 + sqrt(1-alpha_prev - sigma^2) * eps
            alpha = extract(self.ddim_alphas, index, x.shape)
            alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
            sigma = extract(self.ddim_sigmas, index, x.shape)
            dir_xt = (1.0 - alpha_prev - sigma ** 2).sqrt() * noise
            mu = (alpha_prev ** 0.5) * x_recon + dir_xt
            var = sigma ** 2
            logvar = torch.log(var + 1e-20)
        else:
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)

        return mu, logvar

    @torch.no_grad()
    def forward(self, cond, deterministic=True):
        """
        采样。与原版一致：返回 Sample(trajectories, chains)
        - trajectories: [B, 1, H*W]（最终 score map 向量）
        - chains:      None（如需保存中间链条，可自行扩展）
        """
        device = self.betas.device
        # 用 cond 里的某个键确定 batch 大小（与原实现一致）
        if "state" in cond and cond["state"] is not None:
            B = cond["state"].shape[0]
        elif "rgb" in cond and cond["rgb"] is not None:
            B = cond["rgb"].shape[0]
        else:
            # 最稳妥：从 x_in 取
            B = cond["x_in"].shape[0]

        # 初始噪声
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)

        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))


        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar = self.p_mean_var(
                x=x, t=t_b, cond=cond, index=index_b, network_override=None
            )
            std = torch.exp(0.5 * logvar)

            # 确定噪声项
            if self.use_ddim or t == 0:
                std = torch.zeros_like(std)
            else:
                std = torch.clip(std, min=1e-3)

            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

        return Sample(x, None)

    # ---------- Supervised training ---------- #

    def loss(self, x, *args):
        """
        与原实现一致：采样随机 t，调 p_losses
        x: [B, 1, H*W] —— 监督训练时的目标 x0（score map 向量）
        """
        batch_size = x.shape[0]
        t = torch.randint(0, self.denoising_steps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def p_losses(self, x_start, cond: dict, t):
        """
        与原实现一致：预测 ε 或 x0 的 MSE。
        x_start: [B, 1, H*W]（把 [B,1,H,W] 展平成向量后再喂）
        """
        device = x_start.device
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.network(x_noisy, t, cond=cond)  # [B,1,H*W], 预测 ε 或 x0
        if self.predict_epsilon:
            return F.mse_loss(x_recon, noise, reduction="mean")
        else:
            return F.mse_loss(x_recon, x_start, reduction="mean")

    def q_sample(self, x_start, t, noise=None):
        """
        q(x_t | x_0) = N( sqrt(alphabar_t) x0, (1-alphabar_t) I )
        形状与原实现一致：[B, 1, H*W]
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
