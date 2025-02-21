import math
from typing import Tuple
from numbers import Number
import numpy as np
import einops
import torch
import torch.nn as nn
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli

from .modules import Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class TemporalLinearAttention(nn.Module):
    def __init__(self, dim, embed_dim: int, heads=4, dim_head=128, residual: bool = False,):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, hidden_dim * 3)     
        )
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.zeros([1]))

    def forward(self, x, time):
        y = x.clone()
        x = rearrange(x, "b a f t -> b f a t")
        time = self.time_mlp(time)
        time = rearrange(time, "b a f -> b f a 1")
        b, c, h, w = x.shape
        
        qkv = self.to_qkv(x)
        
        q, k, v = rearrange(
            qkv + time, "b (qkv heads c) h w -> qkv (b h) heads c w", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "(b h) heads c w -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        out = self.to_out(out)
        out = rearrange(out, "b f a t -> b a f t")
        if self.residual:
            out = y + self.gamma * out
        return out

class TemporalSelfAttention(nn.Module):
    def __init__(
        self,
        n_channels: int,
        qk_n_channels: int,
        v_n_channels: int,
        embed_dim: int,
        nheads: int = 4,
        residual: bool = False,
    ):
        super().__init__()
        self.nheads = nheads

        self.query_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.key_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.value_layer = nn.Conv1d(n_channels, v_n_channels * nheads, kernel_size=1)

        self.query_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, qk_n_channels * nheads),
            Rearrange("batch t -> batch t 1"),
        )
        self.key_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, qk_n_channels * nheads),
            Rearrange("batch t -> batch t 1"),
        )
        self.value_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, v_n_channels * nheads),
            Rearrange("batch t -> batch t 1"),
        )

        self.attend = nn.Softmax(dim=-1)
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.zeros([1]))

    def forward(self, x, time):
        x_flat = rearrange(x, "b a f t -> (b a) f t")
        time = rearrange(time, "b a f -> (b a) f")
        query, key, value = (
            self.query_layer(x_flat) + self.query_time_mlp(time),
            self.key_layer(x_flat) + self.key_time_mlp(time),
            self.value_layer(x_flat) + self.value_time_mlp(time),
        )

        query = rearrange(
            query, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )
        key = rearrange(
            key, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )
        value = rearrange(
            value, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )

        dots = einsum(query, key, "h b a1 f, h b a2 f -> h b a1 a2") / math.sqrt(
            query.shape[-1]
        )
        attn = self.attend(dots)
        out = einsum(attn, value, "h b a1 a2, h b a2 f -> h b a1 f")

        out = rearrange(out, "h b a f -> b a (h f)")
        out = out.reshape(x.shape)
        if self.residual:
            out = x + self.gamma * out
        return out


class TemporalMlpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim, act_fn, out_act_fn):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    act_fn,
                ),
                nn.Sequential(
                    nn.Linear(dim_out, dim_out),
                    out_act_fn,
                ),
            ]
        )
        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, dim_out),
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """

        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out




class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
                Conv1dBlock(out_channels, out_channels, kernel_size, mish),
            ]
        )

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """

        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)




class TemporalUnet(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        history_horizon: int = 0,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        max_path_length: int = 100,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        mish = True
        act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )
        embed_dim = dim

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition
        self.condition_dropout = condition_dropout
        self.history_horizon = history_horizon

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim += dim

        if self.env_ts_condition:
            self.env_ts_mlp = nn.Sequential(
                nn.Embedding(max_path_length + 1, dim),
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            embed_dim += dim

        self.embed_dim = embed_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            mish=mish,
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            mish=mish,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
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
    ):
        """
        x : [ batch x horizon x transition ]
        returns : [batch x horizon]
        """

        x = einops.rearrange(x, "b t f -> b f t")

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        if self.env_ts_condition:
            assert env_timestep is not None
            env_timestep = env_timestep.to(dtype=torch.int64)
            env_timestep = env_timestep[:, self.history_horizon]
            env_ts_embed = self.env_ts_mlp(env_timestep)
            t = torch.cat([t, env_ts_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b f t -> b t f")
        return x


class TemporalValue(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon,
        transition_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
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

    def forward(self, x, cond, time, *args):
        """
        x : [ batch x horizon x transition ]
        """

        x = einops.rearrange(x, "b h t -> b t h")

        # mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
