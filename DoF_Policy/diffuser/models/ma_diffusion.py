from diffuser.models.helpers import SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffuser.utils.utils import Progress, Silent

import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning
from diffuser.models.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)



class MA_Diffusion(nn.Module):
    def __init__(
            self, 
            obs_dim, 
            action_dim,
            n_agents: int, 
            model, 
            max_action,
            device,
            beta_schedule='linear', 
            n_timesteps=100,
            noise_factorization_mode='concat',  
            loss_type='l2', 
            clip_denoised=True, 
            predict_epsilon=True):
        super(MA_Diffusion, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.model = model
        self.max_action = max_action
        self.device = device
        self.noise_factorization_mode = noise_factorization_mode
    
        if self.noise_factorization_mode == "w-concat":
            self.noise_weights = nn.Parameter(
                torch.ones(self.n_agents, device=self.device),
                requires_grad=True
            )
            
            
            
            
        else:
            self.noise_weights = None
        
        
        if n_agents > 1:
            self.x_shape = (n_agents, action_dim)
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        
        else:
            betas = cosine_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        
        
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):

        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        
        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def sample(self, state, *args, **kwargs):
        
        batch_size = state.shape[0]
        n_agents = state.shape[1]
        shape = (batch_size, n_agents, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)

    

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        epsilon_recon = self.model(x_noisy, t, state)

        assert noise.shape == epsilon_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(epsilon_recon, noise, weights)
        else:
            loss = self.loss_fn(epsilon_recon, x_start, weights)

        return loss

    def p_losses_i(self, x_start, state, t, a_i, weights=1.0):
        noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        epsilon_recon = self.model(x_noisy, t, state)
        
        assert noise.shape == epsilon_recon.shape
        
        if self.noise_factorization_mode == "concat":
            epsilon_recon_i = epsilon_recon[:, a_i].unsqueeze(1)
        
        else:  
            assert self.noise_weights is not None, "noise_weights not init..."
            noise_weight = self.noise_weights[a_i].view(1, 1, 1)
            epsilon_recon_i = epsilon_recon[:, a_i].unsqueeze(1) * self.noise_weight
            
        noise_i = noise[:, a_i].unsqueeze(1)
        x_start_i = x_start[:, a_i].unsqueeze(1)
        
        if self.predict_epsilon:
            loss_i = self.loss_fn(epsilon_recon_i, noise_i, weights)
        else:
            loss_i = self.loss_fn(epsilon_recon_i, x_start_i, weights)
            
        return loss_i
        
        
    
    
    def loss(self, x, state, weights=1.0):   
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def loss_i(self, x, state, a_i, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses_i(x, state, t, a_i, weights)


    def forward(self, state, *args, **kwargs):
 
        return self.sample(state, *args, **kwargs)
