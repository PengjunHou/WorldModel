

import torch
import os

from controller.model.eta import EtaFixed
from controller.model.unet import VisionUnet2D
from controller.model.vit import VitEncoder, VitEncoderConfig
from controller.model.critic import ViTCritic

class DPPOConfig:
    def __init__(self, config, **kwargs):
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Agent
        self.env_name: str = config.env_name
        self.obs_dim = kwargs.get("obs_dim", 10)
        action_dim = kwargs.get("action_dim", 10)
        self.action_H = action_dim[0]
        self.action_W = action_dim[1]
        self.action_dim = self.action_H * self.action_W
        #print(f"obs dim {self.obs_dim}")
        #print(f"action dim {self.action_dim}")
        self.denoising_steps = 100
        self.ft_denoising_steps = 5    # TODO
        self.cond_steps = 1
        self.img_cond_steps = 1
        self.horizon_steps = 1
        self.act_steps = 1
        self.use_ddim = True
        self.summary_dir = "./logs"
        # PPODiffusion
        self.gamma_denoising = 0.99
        self.clip_ploss_coef = 1.0
        self.clip_ploss_coef_base = 1e-3
        self.clip_ploss_coef_rate = 3
        self.min_sampling_denoising_std = 0.1
        self.min_logprob_denoising_std = 0.1
        
        # VPGDiffusion
        self.randn_clip_value = 3
        self.ddim_steps = 5
        self.learn_eta = False

        self.eta = EtaFixed(base_eta = 1, min_eta = 0.1, max_eta = 1.0) \
            if not self.learn_eta else None
        
        self.obs_rgb_shape = (10, self.obs_dim[0], self.obs_dim[1])
        self.obs_state_shape = 2
        self.acotr_backbone = VitEncoder(obs_shape=self.obs_rgb_shape,
                                   cfg=VitEncoderConfig(patch_size=8,
                                                        depth=1,
                                                        embed_dim=128,
                                                        num_heads=4,
                                                        embed_style="embed2",
                                                        embed_norm=0),
                                    num_channel=self.obs_rgb_shape[0],
                                    img_h=self.obs_rgb_shape[1],
                                    img_w=self.obs_rgb_shape[2])
        
        # VisionUnet2D
        self.actor = VisionUnet2D(backbone=self.acotr_backbone,
                                  H=self.obs_rgb_shape[1],
                                  W=self.obs_rgb_shape[2],
                                  ctx_channels= 0, # self.obs_rgb_shape[0],
                                  diffusion_step_embed_dim=32,
                                  dim=32,
                                  dim_mults=[1, 2, 4, 8],
                                  cond_dim=10, # obs_dim * cond_steps
                                  larger_encoder=False, #True,
                                  cond_mlp_dims=None,
                                  kernel_size=3,
                                  img_cond_steps=self.img_cond_steps,
                                  n_groups=None,
                                  spatial_emb=128,
                                  cond_predict_scale=True,
        )

        self.critic_backbone = VitEncoder(obs_shape=self.obs_rgb_shape,
                                 cfg=VitEncoderConfig(patch_size=8,
                                                      depth=1,
                                                      embed_dim=128,
                                                      num_heads=4,
                                                      embed_style="embed2",
                                                      embed_norm=0),
                                  num_channel=self.obs_rgb_shape[0],
                                  img_h=self.obs_rgb_shape[1],
                                  img_w=self.obs_rgb_shape[2])
        self.critic = ViTCritic(obs_shape=self.obs_rgb_shape,
                                backbone=self.critic_backbone,
                                augment=False,
                                spatial_emb=128,
                                cond_dim=10, # obs_dim * cond_steps
                                img_cond_steps=self.img_cond_steps,
                                mlp_dims=[256, 256, 256],
                                residual_style = True,
                                activation_type = 'Mish',)
        
        # training
        self.n_train_itr = 151
        self.n_steps = 10
        self.gamma = 0.999
        self.actor_lr = 5e-5
        self.actor_weight_decay = 0
        self.critic_lr = 1e-3
        self.critic_weight_decay = 0
        self.save_model_freq = 100
        self.val_freq = 10
        # PPO specific
        self.reward_scale_running = True
        self.reward_scale_const = 1.0
        self.gae_lambda = 0.95
        self.batch_size = 4
        self.logprob_batch_size = 500
        self.update_epochs = 10
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.target_kl = 1e-2
        self.checkpoint_dir = config.model_checkpt
        
    
    def print(self):
        print("Diffusion PPO Config:")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("\n")
                                






