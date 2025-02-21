from typing import Tuple

import einops
import torch
from torch import nn
from torch.distributions import Bernoulli


from .basic import (
    SelfAttention,
    Downsample1d,
    ResidualTemporalBlock,
    SinusoidalPosEmb,
    TemporalUnet,
)


class ConvAttentionDeconv(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
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

        self.nets = nn.ModuleList(
            [
                TemporalUnet(
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
                for _ in range(n_agents)
            ]
        )

        if self.use_temporal_attention:
            print("\n USE TEMPORAL ATTENTION !!! \n")
            AttentionModule = TemporalSelfAttention

            self.self_attn = [
                AttentionModule(
                    in_out[-1][1],
                    in_out[-1][1] // 16,
                    in_out[-1][1] // 4,
                    residual=residual_attn,
                    
                    embed_dim=2,
                )
            ]
            for dims in reversed(in_out):
                self.self_attn.append(
                    AttentionModule(
                        dims[1],
                        dims[1] // 16,
                        dims[1] // 4,
                        residual=residual_attn,
                        
                        embed_dim=2,
                    )
                )
        else:
            self.self_attn = [
                SelfAttention(
                    in_out[-1][1],
                    in_out[-1][1] // 16,
                    in_out[-1][1] // 4,
                    residual=residual_attn,
                )
            ]
            for dims in reversed(in_out):
                self.self_attn.append(
                    SelfAttention(
                        dims[1],
                        dims[1] // 16,
                        dims[1] // 4,
                        residual=residual_attn,
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
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  

        t = [self.nets[i].time_mlp(time) for i in range(self.n_agents)]

        if self.returns_condition:
            assert returns is not None
            returns_embed = [
                self.nets[i].returns_mlp(returns[:, :, i]) for i in range(self.n_agents)
            ]
            if use_dropout:
                
                mask = (
                    self.nets[0]
                    .mask_dist.sample(sample_shape=(returns_embed[0].size(0), 1))
                    .to(returns_embed[0].device)
                )
                returns_embed = [
                    returns_embed[i] * mask for i in range(len(returns_embed))
                ]
            if force_dropout:
                returns_embed = [
                    returns_embed[i] * 0 for i in range(len(returns_embed))
                ]

            t = [torch.cat([t[i], returns_embed[i]], dim=-1) for i in range(len(t))]

        if self.env_ts_condition:
            assert env_timestep is not None
            env_ts_embed = [
                self.nets[i].env_ts_mlp(env_timestep) for i in range(self.n_agents)
            ]
            t = [torch.cat([t[i], env_ts_embed[i]], dim=-1) for i in range(len(t))]

        h = [[] for _ in range(self.n_agents)]

        for layer_idx in range(len(self.nets[0].downs)):
            for i in range(self.n_agents):
                resnet, resnet2, downsample = self.nets[i].downs[layer_idx]
                x[i] = resnet(x[i], t[i])
                x[i] = resnet2(x[i], t[i])
                h[i].append(x[i])
                x[i] = downsample(x[i])

        for i in range(self.n_agents):
            x[i] = self.nets[i].mid_block1(x[i], t[i])
            x[i] = self.nets[i].mid_block2(x[i], t[i])

        x = self.self_attn[0](torch.stack(x, dim=1))  
        if self.use_layer_norm:
            x = self.layer_norm[0](x)
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  

        for layer_idx in range(len(self.nets[0].ups)):
            hiddens = torch.stack([hid.pop() for hid in h], dim=1)  
            if self.use_layer_norm:
                hiddens = self.layer_norm[layer_idx + 1](hiddens)
            hiddens = self.self_attn[layer_idx + 1](hiddens)
            for i in range(self.n_agents):
                resnet, resnet2, upsample = self.nets[i].ups[layer_idx]
                x[i] = torch.cat((x[i], hiddens[:, i]), dim=1)
                if self.use_layer_norm:
                    x[i] = self.layer_norm_cat[layer_idx](x[i])
                x[i] = resnet(x[i], t[i])
                x[i] = resnet2(x[i], t[i])
                x[i] = upsample(x[i])

        for i in range(self.n_agents):
            x[i] = self.nets[i].final_conv(x[i])

        x = torch.stack(x, dim=1)
        x = einops.rearrange(x, "b a f t -> b t a f")

        return x



class ConvAttentionTemporalValue(nn.Module):
    agent_share_parameters = False

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
        self.time_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    SinusoidalPosEmb(dim),
                    nn.Linear(dim, dim * 4),
                    nn.Mish(),
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(n_agents)
            ]
        )

        self.blocks = nn.ModuleList([nn.ModuleList([]) for _ in range(n_agents)])
        num_resolutions = len(in_out)

        print("ConvAttentionTemporalValue: ", in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            for i in range(n_agents):
                self.blocks[i].append(
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

        self.mid_block1 = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    mid_dim,
                    mid_dim_2,
                    kernel_size=5,
                    embed_dim=time_dim,
                )
                for _ in range(n_agents)
            ]
        )
        self.mid_block2 = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    mid_dim_2,
                    mid_dim_3,
                    kernel_size=5,
                    embed_dim=time_dim,
                )
                for _ in range(n_agents)
            ]
        )
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fc_dim + time_dim, fc_dim // 2),
                    nn.Mish(),
                    nn.Linear(fc_dim // 2, out_dim),
                )
                for _ in range(n_agents)
            ]
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
        
        
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  

        t = [self.time_mlp[i](time) for i in range(self.n_agents)]

        for layer_idx in range(len(self.blocks[0])):
            for i in range(self.n_agents):
                resnet, resnet2, downsample = self.blocks[i][layer_idx]
                x[i] = resnet(x[i], t[i])
                x[i] = resnet2(x[i], t[i])
                x[i] = downsample(x[i])
            x = self.self_attn[layer_idx](torch.stack(x, dim=1))
            x = [x[:, a_idx] for a_idx in range(x.shape[1])]  

        for i in range(self.n_agents):
            x[i] = self.mid_block1[i](x[i], t[i])
            x[i] = self.mid_block2[i](x[i], t[i])
            x[i] = x[i].view(len(x[i]), -1)
            x[i] = self.final_block[i](torch.cat([x[i], t[i]], dim=-1))
        x = torch.stack(x, dim=1).squeeze(-1)

        
        out = x.mean(axis=1, keepdim=True)  

        return out