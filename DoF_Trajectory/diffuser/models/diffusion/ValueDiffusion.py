import torch

from .GaussianDiffuison import GaussianDiffusion
from diffuser.models.nn_diffusion.basic import apply_conditioning

class ValueDiffusion(GaussianDiffusion):
    def __init__(self, *args, clean_only=False, **kwargs):
        assert "value" in kwargs["loss_type"]
        super().__init__(*args, **kwargs)
        if clean_only:
            print("[ models/diffusion ] Info: Only train on clean samples!")
        self.clean_only = clean_only
        self.sqrt_alphas_cumprod = torch.cat(
            [
                torch.ones(1, device=self.betas.device),
                torch.sqrt(self.alphas_cumprod[:-1]),
            ]
        )
        self.sqrt_one_minus_alphas_cumprod = torch.cat(
            [
                torch.zeros(1, device=self.betas.device),
                torch.sqrt(1 - self.alphas_cumprod[:-1]),
            ]
        )

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        value_loss, info = self.p_losses(x, cond, returns, t - 1)
        value_loss = value_loss.mean()
        return value_loss, info

    def p_losses(self, x_start, cond, target, t):
        if self.clean_only:
            pred = self.model(x_start, torch.zeros_like(t))

        else:
            t = t + 1
            noise = torch.randn_like(x_start)

            
            
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_noisy = apply_conditioning(x_noisy, cond)
            x_noisy = self.data_encoder(x_noisy)
            pred = self.model(x_noisy, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, t):
        return self.model(x, t)
