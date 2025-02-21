from typing import Tuple

import einops
import torch
import torch.nn as nn

from .basic import (
    TemporalUnet,
    Downsample1d,
    ResidualTemporalBlock,
    SinusoidalPosEmb,
    TemporalUnet,
)


class ConcatenatedTemporalUnet(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = False,  
        use_layer_norm: bool = False,
        max_path_length: int = 100,
        use_temporal_attention: bool = False,  
    ):
        super().__init__()

        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        self.net = TemporalUnet(
            horizon=horizon,
            history_horizon=history_horizon,
            transition_dim=transition_dim * n_agents,
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            env_ts_condition=env_ts_condition,
            condition_dropout=condition_dropout,
            kernel_size=kernel_size,
            max_path_length=max_path_length,
        )

    def forward(
        self,
        x,
        time,
        returns=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [batch x 1 x agent]
        """

        assert x.shape[2] == self.n_agents, f"{x.shape}, {self.n_agents}"

        concat_x = einops.rearrange(x, "b h a f -> b h (a f)")
        concat_x = self.net(
            concat_x,
            time=time,
            returns=returns.mean(dim=2) if returns is not None else None,
            env_timestep=env_timestep,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )
        x = einops.rearrange(concat_x, "b h (a f) -> b h a f", a=self.n_agents)

        return x


class IndependentTemporalUnet(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = False,  
        max_path_length: int = 100,
        use_temporal_attention: bool = False,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

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
                    kernel_size=kernel_size,
                    max_path_length=max_path_length,
                )
                for _ in range(n_agents)
            ]
        )

    def forward(
        self,
        x,
        time,
        returns=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [batch x agent x horizon]
        """

        assert x.shape[2] == self.n_agents, f"{x.shape}, {self.n_agents}"

        x_list = []
        for a_idx in range(self.n_agents):
            x_list.append(
                self.nets[a_idx](
                    x[:, :, a_idx, :],
                    time=time,
                    returns=returns[:, :, a_idx] if returns is not None else None,
                    env_timestep=env_timestep,
                    use_dropout=use_dropout,
                    force_dropout=force_dropout,
                )
            )
        x_list = torch.stack(x_list, dim=2)
        return x_list

class ConcatTemporalValue(nn.Module):
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

        dims = [transition_dim * n_agents, *map(lambda m: dim * m, dim_mults)]
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

    def forward(self, x, time, *args):
        """
        x : [ batch x horizon x n_agents x transition ]
        """

        assert (
            x.shape[2] == self.n_agents
        ), f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"

        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = einops.rearrange(x, "b t f -> b f t")
        t = self.time_mlp(time)

        for layer_idx, (resnet, resnet2, downsample) in enumerate(self.blocks):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))  

        return out
