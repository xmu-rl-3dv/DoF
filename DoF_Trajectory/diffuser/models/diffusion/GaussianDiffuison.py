from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_consistency_models import CMStochasticIterativeScheduler


import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning

class QMixNet(nn.Module):
    def __init__(self, state_dim, n_agents, action_dim):
        super(QMixNet, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.action_dim = action_dim
        
        self.hyper_w = nn.Linear(state_dim, n_agents * action_dim)
        self.hyper_b = nn.Linear(state_dim, action_dim)

    def forward(self, actions, states):
        batch_size = actions.shape[0]
        w = torch.abs(self.hyper_w(states)).view(batch_size, self.n_agents, self.action_dim)
        b = self.hyper_b(states).view(batch_size, 1, self.action_dim)
        
        mixed_actions = torch.bmm(actions.view(batch_size, 1, -1), w).squeeze(1) + b
        return mixed_actions

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        n_agents: int,
        horizon: int,
        history_horizon: int,
        observation_dim: int,
        action_dim: int,
        use_inv_dyn: bool = True,
        discrete_action: bool = False,
        num_actions: int = 0,  
        n_timesteps: int = 1000,
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        action_weight: float = 1.0,
        hidden_dim: int = 256,
        loss_discount: float = 1.0,
        loss_weights: np.ndarray = None,
        state_loss_weight: float = None,
        opponent_loss_weight: float = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 1.2,
        returns_loss_guided: bool = False,
        loss_guidence_w: float = 0.1,
        value_diffusion_model: nn.Module = None,
        train_only_inv: bool = False,
        share_inv: bool = True,
        joint_inv: bool = False,
        data_encoder: utils.Encoder = utils.IdentityEncoder(),
        use_learnable_agent_weights=False,  
        use_qmix_combiner=False,  
        use_data_agent_weights=False,
        **kwargs,
    ):
        assert action_dim > 0
        assert (
            not returns_condition or not returns_loss_guided
        ), "Can't do both returns conditioning and returns loss guidence"

        super().__init__()
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_loss_weight = state_loss_weight
        self.opponent_loss_weight = opponent_loss_weight
        self.discrete_action = discrete_action
        self.num_actions = num_actions
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.use_inv_dyn = use_inv_dyn
        self.train_only_inv = train_only_inv
        self.share_inv = share_inv
        self.joint_inv = joint_inv
        self.data_encoder = data_encoder
        self.use_learnable_agent_weights = use_learnable_agent_weights  
        self.use_qmix_combiner = use_qmix_combiner
        self.use_data_agent_weights = use_data_agent_weights
        self.agent_models = nn.ModuleList([model for _ in range(n_agents)])
        if self.use_qmix_combiner:
            self.qmix_net = QMixNet(observation_dim, n_agents, action_dim)


        if self.use_learnable_agent_weights:
            self.agent_weights = nn.Parameter(torch.ones(n_agents))

        if self.use_data_agent_weights:
            self.data_agent_weights = nn.Parameter(torch.ones(n_agents))

        if self.use_inv_dyn:
            self.inv_model = self._build_inv_model(
                hidden_dim,
                output_dim=action_dim if not discrete_action else num_actions,
            )

        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        self.returns_loss_guided = returns_loss_guided
        self.loss_guidence_w = loss_guidence_w
        self.value_diffusion_model = value_diffusion_model
        if self.value_diffusion_model is not None:
            self.value_diffusion_model.requires_grad_(False)

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.use_ddim_sample = False
        self.use_consistency_models_sample = False

        loss_weights = self.get_loss_weights(loss_discount, action_weight)
        loss_type = "state_l2" if self.use_inv_dyn else "l2"
        self.loss_fn = Losses[loss_type](loss_weights)

    def _build_inv_model(self, hidden_dim: int, output_dim: int):
        if self.joint_inv:
            print("\n USE JOINT INV \n")
            inv_model = nn.Sequential(
                nn.Linear(self.n_agents * (2 * self.observation_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_agents * output_dim),
            )

        elif self.share_inv:
            print("\n USE SHARED INV \n")
            inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        else:
            print("\n USE INDEPENDENT INV \n")
            inv_model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.observation_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                        nn.Softmax(dim=-1) if self.discrete_action else nn.Identity(),
                    )
                    for _ in range(self.n_agents)
                ]
            )

        return inv_model
    
    def set_ddim_scheduler(self, n_ddim_steps: int = 15):
        self.ddim_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.ddim_noise_scheduler.set_timesteps(n_ddim_steps)
        self.use_ddim_sample = True
    
    def set_consistency_models_scheduler(self, n_consistency_model_steps: int = 15):
        self.consistency_models_scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps = self.n_timesteps,
            sigma_min = 0.002,
            sigma_max = 80,
            rho = 7.0
        )
        self.consistency_models_scheduler.set_timesteps(n_consistency_model_steps)
        self.use_consistency_models_sample = True

    def get_loss_weights(self, discount: float, action_weight: Optional[float] = None):
        """
        sets loss coefficients for trajectory

        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        """

        if self.use_inv_dyn:
            dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)
        else:
            dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        discounts = torch.cat([torch.zeros(self.history_horizon), discounts])
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.unsqueeze(1).expand(-1, self.n_agents, -1).clone()

        
        if not self.use_inv_dyn:
            loss_weights[self.history_horizon, :, : self.action_dim] = action_weight
        return loss_weights

    

    def get_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        if self.returns_condition:
            
            per_epsilons_con = []
            for i, per_model in enumerate(self.agent_models):
                per_epsilon = per_model(
                        x[:,:,i,:],
                        t,
                        returns = returns[:,:,i],
                        env_timestep = env_ts,
                        attention_masks = attention_masks,
                        use_dropout=False,
                    )
                per_epsilons_con.append(per_epsilon)
            epsilon_cond = torch.stack(per_epsilons_con, dim=2)

            if self.use_learnable_agent_weights:

                weighted_epsilon_cond = epsilon_cond * self.agent_weights.view(1, 1, -1, 1)
                epsilon_cond = weighted_epsilon_cond / self.agent_weights.sum()
            
            per_epsilons_uncon = []
            for i, per_model in enumerate(self.agent_models):
                    per_epsilon_un = per_model(
                        x[:,:,i,:],
                        t,
                        returns = returns[:,:,i],
                        env_timestep = env_ts,
                        attention_masks = attention_masks,
                        use_dropout=True,
                    )
                    per_epsilons_uncon.append(per_epsilon_un)

            epsilon_uncond = torch.stack(per_epsilons_uncon, dim=2)

            if self.use_learnable_agent_weights:

                weighted_epsilon_uncond = epsilon_uncond * self.agent_weights.view(1, 1, -1, 1)
                epsilon_uncond = weighted_epsilon_uncond / self.agent_weights.sum()

            epsilon = epsilon_uncond + self.condition_guidance_w * (
                            epsilon_cond - epsilon_uncond
                        )

        else:
            per_epsilons = []
            for i, per_model in enumerate(self.agent_models):
                    per_epsilon = per_model(
                        x[:,:,i,:],
                        t,
                        returns = returns[:,:,i],
                        env_timestep = env_ts,
                        attention_masks = attention_masks,
                        use_dropout=False,
                    )
                    per_epsilons.append(per_epsilon)
            epsilons = per_epsilons
            epsilon =  torch.stack(epsilons, dim=2)   

        return epsilon

    @torch.no_grad()
    def conditional_sample(
        self,
        cond: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        horizon: int = None,
        attention_masks: Optional[torch.Tensor] = None,
        verbose: bool = True,
        return_diffusion: bool = False,
    ):
        """
        conditions : [ (time, state), ... ]
        """

        batch_size = cond["x"].shape[0]
        horizon = horizon or self.horizon + self.history_horizon
        
        shape = (batch_size, horizon, self.n_agents, self.observation_dim)

        device = list(cond.values())[0].device
        if self.use_ddim_sample:
            scheduler = self.ddim_noise_scheduler
        else:
            scheduler = self.noise_scheduler

        if self.use_consistency_models_sample:
            scheduler = self.consistency_models_scheduler
        else:
            scheduler = self.noise_scheduler
            
        x = 0.5 * torch.randn(shape, device=device)  

        if return_diffusion:
            diffusion = [x]

        
        
        timesteps = scheduler.timesteps

        progress = utils.Progress(len(timesteps)) if verbose else utils.Silent()
        for t in timesteps:
            
            x = apply_conditioning(x, cond)
            x = self.data_encoder(x)

            
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
            model_output = self.get_model_output(
                x, ts, returns, env_ts, attention_masks
            )
            print("model_output: ", model_output.shape)
            
            x = scheduler.step(model_output, t, x).prev_sample

            progress.update({"t": t})
            if return_diffusion:
                diffusion.append(x)

        
        x = apply_conditioning(x, cond)
        x = self.data_encoder(x)

        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        loss_masks: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        noise = torch.randn_like(x_start)

        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        x_noisy = apply_conditioning(x_noisy, cond)
        x_noisy = self.data_encoder(x_noisy)

        per_epsilons = []
        for i, per_model in enumerate(self.agent_models):
            per_epsilon = per_model(
                x_noisy[:, :, i, : ],
                t,
                returns = returns[:, : , i],
                env_timestep = env_ts,
                attention_masks = attention_masks,
            )
            per_epsilons.append(per_epsilon)
        epsilon = torch.stack(per_epsilons, dim = 2) 


        if self.use_learnable_agent_weights:
            weighted_x_recon = epsilon * self.agent_weights.view(1, 1, -1, 1)
            epsilon = weighted_x_recon / self.agent_weights.sum()
        
        if self.use_qmix_combiner:
            batch_size, seq_len, n_agents, action_dim = epsilon.shape
            epsilon = epsilon.view(batch_size, seq_len, n_agents * action_dim)
            
            if states is None:
                states = x_start[:, :, :, :self.observation_dim].reshape(batch_size, seq_len, -1)
            else:
                states = states.reshape(batch_size, seq_len, -1)
            
            epsilon = self.qmix_net(epsilon, states)
            epsilon = epsilon.view(batch_size, seq_len, n_agents, action_dim)

        if not self.predict_epsilon:
            epsilon = apply_conditioning(epsilon, cond)
            epsilon = self.data_encoder(epsilon)

        assert noise.shape == epsilon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(epsilon, noise)
        else:
            loss, info = self.loss_fn(epsilon, x_start)

        if "agent_idx" in cond.keys() and self.opponent_loss_weight is not None:
            opponent_loss_weight = torch.ones_like(loss) * self.opponent_loss_weight
            indices = (
                cond["agent_idx"]
                .to(torch.long)[..., None]
                .repeat(
                    1, opponent_loss_weight.shape[1], 1, opponent_loss_weight.shape[-1]
                )
            )
            opponent_loss_weight.scatter_(dim=2, index=indices, value=1)
            loss = loss * opponent_loss_weight


        loss = (
            (loss * loss_masks).mean(dim=[1, 2]) / loss_masks.mean(dim=[1, 2])
        ).mean()

        if self.returns_loss_guided:
            returns_loss = self.r_losses(x_noisy, t, epsilon, cond)
            info["returns_loss"] = returns_loss
            loss = loss + returns_loss * self.loss_guidence_w

        return loss, info

    def r_losses(self, x_t, t, noise, cond):
        b = x_t.shape[0]
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x_t, t, noise)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, _, model_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )

        noise = 0.5 * torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))

        x_t_minus_1 = (
            model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        )
        x_t_minus_1 = apply_conditioning(x_t_minus_1, cond)
        x_t_minus_1 = self.data_encoder(x_t_minus_1)

        
        value_pred = self.value_diffusion_model(x_t_minus_1, t)

        
        return -1.0 * value_pred.mean()  

    def compute_inv_loss(
        self,
        x: torch.Tensor,
        loss_masks: torch.Tensor,
        legal_actions: Optional[torch.Tensor] = None,
        use_data_agent_weights: bool = False,
    ):
        info = {}
        
        x_t = x[:, :-1, :, self.action_dim :]
        a_t = x[:, :-1, :, : self.action_dim]
        x_t_1 = x[:, 1:, :, self.action_dim :]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        
        
        
        masks_t = loss_masks[:, 1:]
        if legal_actions is not None:
            legal_actions_t = legal_actions[:, :-1].reshape(
                -1, *legal_actions.shape[2:]
            )
        
        if use_data_agent_weights:
            x_comb_t = x_comb_t.sum(dim=2, keepdim=True)
            a_t = a_t.sum(dim=2, keepdim=True)
            masks_t = masks_t.sum(dim=2, keepdim=True)
            if legal_actions is not None:
                legal_actions_t = legal_actions_t.sum(dim=2, keepdim=True)
            
            x_comb_t = x_comb_t.reshape(-1, x_comb_t.shape[-1])
            a_t = a_t.reshape(-1, a_t.shape[-1])
            masks_t = masks_t.reshape(-1)
            if legal_actions is not None:
                legal_actions_t = legal_actions[:, :-1].sum(dim=2, keepdim=True)   
        else:
            x_comb_t = x_comb_t.reshape(-1, x_comb_t.shape[2], 2 * self.observation_dim)
            a_t = a_t.reshape(-1, a_t.shape[2], self.action_dim)
            masks_t = masks_t.reshape(-1, masks_t.shape[2])
            if legal_actions is not None:
                legal_actions_t = legal_actions[:, :-1].reshape(-1, *legal_actions.shape[2:])


        if self.joint_inv or self.share_inv:
            if self.joint_inv:
                pred_a_t = self.inv_model(
                    x_comb_t.reshape(x_comb_t.shape[0], -1)  
                ).reshape(x_comb_t.shape[0], x_comb_t.shape[1], -1)
            else:
                pred_a_t = self.inv_model(x_comb_t)

            if legal_actions is not None:
                pred_a_t[legal_actions_t == 0] = -1e10
            if self.discrete_action:
                inv_loss = (
                    F.cross_entropy(
                        pred_a_t.reshape(-1, pred_a_t.shape[-1]),
                        a_t.reshape(-1).long(),
                        reduction="none",
                    )
                    * masks_t.reshape(-1)
                ).mean() / masks_t.mean()
                inv_acc = (
                    (pred_a_t.argmax(dim=-1, keepdim=True) == a_t)
                    .to(dtype=float)
                    .squeeze(-1)
                    * masks_t
                ).mean() / masks_t.mean()
                info["inv_acc"] = inv_acc
            else:
                inv_loss = (
                    F.mse_loss(pred_a_t, a_t, reduction="none") * masks_t.unsqueeze(-1)
                ).mean() / masks_t.mean()

        else:
            inv_loss = 0.0
            for i in range(self.n_agents):
                pred_a_t = self.inv_model[i](x_comb_t[:, i])
                if self.discrete_action:
                    inv_loss += (
                        F.cross_entropy(
                            pred_a_t, a_t[:, i].reshape(-1).long(), reduction="none"
                        )
                        * masks_t[:, i]
                    ).mean() / masks_t[:, i].mean()
                else:
                    inv_loss += (
                        F.mse_loss(pred_a_t, a_t[:, i]) * masks_t[:, i].unsqueeze(-1)
                    ).mean() / masks_t[:, i].mean()

        return inv_loss, info

    def loss(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        legal_actions: Optional[torch.Tensor] = None,
    ):
        if self.train_only_inv:
            assert self.use_inv_dyn, "If train_only_inv, must use inv_dyn"
            info = {}
        else:
            batch_size = len(x)
            t = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=x.device,
            ).long()

            if self.use_inv_dyn:
                diffuse_loss, info = self.p_losses(
                    x[..., self.action_dim :],
                    cond,
                    t,
                    loss_masks,
                    attention_masks,
                    returns,
                    env_ts,
                    states,
                )
            else:
                diffuse_loss, info = self.p_losses(
                    x,
                    cond,
                    t,
                    loss_masks,
                    attention_masks,
                    returns,
                    env_ts,
                    states,
                )

        if self.use_inv_dyn:
            inv_loss, inv_info = self.compute_inv_loss(x, loss_masks, legal_actions, self.use_data_agent_weights)
            info = {**info, **inv_info}
            info["inv_loss"] = inv_loss

            if self.train_only_inv:
                return inv_loss, info

            loss = (1 / 2) * (diffuse_loss + inv_loss)
        else:
            loss = diffuse_loss

        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

