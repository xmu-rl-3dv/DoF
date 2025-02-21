from typing import Tuple

import einops
import torch
from torch import nn
from torch.distributions import Bernoulli

from .basic import (
    MlpSelfAttention, 
    SelfAttention,
    Downsample1d,
    ResidualTemporalBlock,
    SinusoidalPosEmb,
    TemporalMlpBlock,
    TemporalSelfAttention,
    TemporalUnet,
)



class SharedConvAttentionDeconv(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        nhead: int = 4,
        n_agents: int = 2,
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = True,
        use_layer_norm: bool = False,
        max_path_length: int = 100,
        use_temporal_attention: bool = True,
    ):
        super().__init__()

 
 
 
        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        self.net = TemporalUnet(
            horizon=horizon,
            history_horizon=history_horizon,
            transition_dim=transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            env_ts_condition=env_ts_condition,
            condition_dropout=condition_dropout,
            max_path_length=max_path_length,
            kernel_size=kernel_size,
        )

        self.self_attn = [
            TemporalSelfAttention(
                in_out[-1][1],
                in_out[-1][1] // 16,
                in_out[-1][1] // 4,
                residual=residual_attn,
                embed_dim=self.net.embed_dim,
            )

 
        ]

        for dims in reversed(in_out):
            self.self_attn.append(
                TemporalSelfAttention(
                    dims[1],
                    dims[1] // 16,
                    dims[1] // 4,
                    residual=residual_attn,
                    embed_dim=self.net.embed_dim,
                )

 
            )
        self.self_attn = nn.ModuleList(self.self_attn)

        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            horizon_ = horizon
            self.layer_norm = []
            for dims in in_out:
                self.layer_norm.append(nn.LayerNorm([dims[1], horizon_]))
                horizon_ = horizon_ // 2
            horizon_ = horizon_ * 2
            self.layer_norm.append(nn.LayerNorm([in_out[-1][1], horizon_]))
            self.layer_norm = list(reversed(self.layer_norm))
            self.layer_norm = nn.ModuleList(self.layer_norm)

            horizon_ = horizon
            self.layer_norm_cat = []
            for dims in in_out:
                self.layer_norm_cat.append(nn.LayerNorm([dims[1] * 2, horizon_]))
                horizon_ = horizon_ // 2
            self.layer_norm_cat = list(reversed(self.layer_norm_cat))
            self.layer_norm_cat = nn.ModuleList(self.layer_norm_cat)

    def forward(
        self,
        x,
        time,
        returns=None,
        states=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [ batch x horizon x agent ]
        """

        assert (
            x.shape[2] == self.n_agents
        ), f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"

        x = einops.rearrange(x, "b t a f -> b a f t")
        bs = x.shape[0]

        t = self.net.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))
        if self.returns_condition:
            assert returns is not None
            returns = einops.rearrange(returns, "b t a -> b a t")
            returns_embed = self.net.returns_mlp(returns)
            if use_dropout:
 
                mask = self.net.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), returns_embed.size(1), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        if self.env_ts_condition:
            assert env_timestep is not None
            env_timestep = env_timestep.to(dtype=torch.int64)
            env_timestep = env_timestep[:, self.history_horizon]
            env_ts_embed = self.net.env_ts_mlp(env_timestep)
            env_ts_embed = einops.repeat(env_ts_embed, "b f -> b a f", a=x.shape[1])
            t = torch.cat([t, env_ts_embed], dim=-1)

        h = []
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
        for resnet, resnet2, downsample in self.net.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.net.mid_block1(x, t)
        x = self.net.mid_block2(x, t)

        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
        if self.use_layer_norm:
            x = self.layer_norm[0](x)
        if self.use_temporal_attention:
            t = t.reshape(bs, t.shape[0] // bs, t.shape[1])
            x = self.self_attn[0](x, t)  
            t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
        else:
            x = self.self_attn[0](x)  

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        for layer_idx in range(len(self.net.ups)):
            hiddens = h.pop()
            hiddens = hiddens.reshape(
                bs, hiddens.shape[0] // bs, hiddens.shape[1], hiddens.shape[2]
            )
            if self.use_layer_norm:
                hiddens = self.layer_norm[layer_idx + 1](hiddens)
            if self.use_temporal_attention:
                t = t.reshape(bs, t.shape[0] // bs, t.shape[1])
                hiddens = self.self_attn[layer_idx + 1](hiddens, t)
                t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
            else:
                hiddens = self.self_attn[layer_idx + 1](hiddens)

            hiddens = hiddens.reshape(
                hiddens.shape[0] * hiddens.shape[1], hiddens.shape[2], hiddens.shape[3]
            )
            resnet, resnet2, upsample = self.net.ups[layer_idx]
            x = torch.cat((x, hiddens), dim=1)
            if self.use_layer_norm:
                x = self.layer_norm_cat[layer_idx](x)

            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.net.final_conv(x)
        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])

        x = einops.rearrange(x, "b a f t -> b t a f")

        return x


class SharedAttentionAutoEncoder(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4),
        n_agents: int = 2,
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
    ):
        assert (
            horizon == 1
        ), f"Only horizon=1 is supported for AttentionAutoEncoder, but got horizon={horizon}"
        super().__init__()

        self.n_agents = n_agents
        self.condition_dropout = condition_dropout

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/stationary ] Hidden dimensions: {in_out}")

        act_fn = nn.Mish()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )

            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim = 2 * dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                TemporalMlpBlock(
                    dim_in,
                    dim_out,
                    embed_dim,
                    act_fn,
                    out_act_fn=act_fn if not is_last else nn.Identity(),
                )
            )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                TemporalMlpBlock(
                    dim_out * 2,
                    dim_in,
                    embed_dim,
                    act_fn,
                    out_act_fn=act_fn,
                )
            )

        self.final_mlp = nn.Sequential(
            nn.Linear(dim, dim), act_fn, nn.Linear(dim, transition_dim)
        )

        self.self_attn = [MlpSelfAttention(in_out[-1][1])]
        for dims in reversed(in_out):
            self.self_attn.append(MlpSelfAttention(dims[1]))
        self.self_attn = nn.ModuleList(self.self_attn)

    def forward(self, x, time, returns=None, use_dropout=True, force_dropout=False):
        """
        x : [ batch x horizon(1) x agent x transition ]
        returns : [batch x horizon(1) x agent]
        """

        assert (
            x.shape[2] == self.n_agents
        ), f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"

        x = x.squeeze(1)  
        bs = x.shape[0]

        t = self.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))

        if self.returns_condition:
            assert returns is not None
 
            returns = einops.rearrange(returns, "b t a -> b a t")
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
 
                mask = self.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), returns_embed.size(1), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed

            t = torch.cat([t, returns_embed], dim=-1)

        h = []
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
        for mlp in self.downs:
            x = mlp(x, t)
            h.append(x)

        x = x.reshape(bs, x.shape[0] // bs, x.shape[1])
        x = self.self_attn[0](x)  

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        for layer_idx in range(len(self.ups)):
            hiddens = h.pop()
            hiddens = hiddens.reshape(bs, hiddens.shape[0] // bs, hiddens.shape[1])
            hiddens = self.self_attn[layer_idx + 1](hiddens)
            hiddens = hiddens.reshape(
                hiddens.shape[0] * hiddens.shape[1], hiddens.shape[2]
            )
            mlp = self.ups[layer_idx]
            x = torch.cat([x, hiddens], dim=-1)
            x = mlp(x, t)

        x = self.final_mlp(x)
        x = x.reshape(bs, 1, x.shape[0] // bs, x.shape[1])

        return x

    

class SharedConvAttentionTemporalValue(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon,
        transition_dim,
        n_agents,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.n_agents = n_agents
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print("ConvAttentionTemporalValue: ", in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16

        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim
        )
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )
        self.self_attn = nn.ModuleList(
            [SelfAttention(dim[1], dim[1] // 16) for dim in in_out]
        )

    def forward(self, x, time, *args):
        """
        x : [ batch x horizon x n_agents x transition ]
        """

        assert (
            x.shape[2] == self.n_agents
        ), f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"

        x = einops.rearrange(x, "b t a f -> b a f t")
        bs = x.shape[0]

        t = self.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])

        for layer_idx, (resnet, resnet2, downsample) in enumerate(self.blocks):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)
            x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
            x = self.self_attn[layer_idx](x)
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        x = x.view(len(x), -1)
        x = self.final_block(torch.cat([x, t], dim=-1))  

        x = x.reshape(bs, -1)  
 
        out = x.mean(axis=1, keepdim=True)  

        return out