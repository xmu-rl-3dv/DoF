from typing import Tuple

import einops
import torch
import torch.nn as nn

from .basic import TemporalUnet, TemporalValue



class SharedIndependentTemporalUnet(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        history_horizon: int,  
        transition_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = False,  
        max_path_length: int = 100,
    ):
        super().__init__()

        self.n_agents = n_agents

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition
        self.history_horizon = history_horizon

        self.net = TemporalUnet(
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

    def forward(
        self,
        x,
        time,
        returns=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout=True,
        force_dropout=False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [batch x agent x horizon]
        """

        assert x.shape[2] == self.n_agents, f"{x.shape}, {self.n_agents}"

        x = einops.rearrange(x, "b t a f -> b a t f")
        bs = x.shape[0]

        x = self.net(
            x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]),
            time=torch.cat([time for _ in range(x.shape[1])], dim=0),
            returns=torch.cat(
                [returns[:, :, a_idx] for a_idx in range(self.n_agents)], dim=0
            )
            if returns is not None
            else None,
            env_timestep=torch.cat([env_timestep for _ in range(x.shape[1])], dim=0)
            if env_timestep is not None
            else None,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )
        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
        x = einops.rearrange(x, "b a t f -> b t a f")
        return x

class SharedIndependentTemporalValue(nn.Module):
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

        self.n_agents = n_agents
        self.net = TemporalValue(
            horizon=horizon,
            transition_dim=transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            out_dim=out_dim,
        )

    def forward(self, x, time, *args):
        """
        x : [ batch x horizon x n_agents x transition ]
        """

        assert (
            x.shape[2] == self.n_agents
        ), f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"

        x = einops.rearrange(x, "b t a f -> b a t f")
        bs = x.shape[0]

        out = self.net(
            x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]),
            time=torch.cat([time for _ in range(x.shape[1])], dim=0),
        )
        out = out.reshape(bs, out.shape[0] // bs, out.shape[1])

        return out
