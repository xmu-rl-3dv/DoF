import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from einops.layers.torch import Rearrange

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

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_channels: int,
        qk_n_channels: int,
        v_n_channels: int,
        nheads: int = 4,
        residual: bool = False,
        use_state: bool = False,
    ):
        super().__init__()
        self.nheads = nheads
        self.query_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.key_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.value_layer = nn.Conv1d(n_channels, v_n_channels * nheads, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.residual = residual
        self.use_state = use_state
        if use_state:
            self.state_query_layer = nn.Conv1d(n_channels, qk_n_channels, kernel_size=1)
            self.state_key_layer = nn.Conv1d(n_channels, qk_n_channels, kernel_size=1)
            self.state_value_layer = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        if residual:
            self.gamma = nn.Parameter(torch.zeros([1]))

    def forward(self, x, states: torch.Tensor = None):
        x_flat = rearrange(x, "b a f t -> (b a) f t")
        query, key, value = (
            self.query_layer(x_flat),
            self.key_layer(x_flat),
            self.value_layer(x_flat),
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

        if self.use_state:
            assert states is not None  # b f t
            state_query, state_key, state_value = (
                self.state_query_layer(states),
                self.state_key_layer(states),
                self.state_value_layer(states),
            )
            state_query = rearrange(
                state_query, "b (h d) t -> h b 1 (d t)", h=self.nheads
            )
            state_key = rearrange(state_key, "b (h d) t -> h b 1 (d t)", h=self.nheads)
            state_value = rearrange(
                state_value, "b (h d) t -> h b 1 (d t)", h=self.nheads
            )
            query = torch.cat((query, state_query), dim=2)
            key = torch.cat((key, state_key), dim=2)
            value = torch.cat((value, state_value), dim=2)

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


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout: float = 0, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class MlpSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_hidden=128):
        super().__init__()
        self.query_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.key_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_in),
        )

    def forward(self, x):
        x_flat = x.reshape(x.shape[0] * x.shape[1], -1)
        query, key, value = (
            self.query_layer(x_flat),
            self.key_layer(x_flat),
            self.value_layer(x_flat),
        )
        query = query.reshape(x.shape[0], x.shape[1], -1)
        key = key.reshape(x.shape[0], x.shape[1], -1)
        value = value.reshape(x.shape[0], x.shape[1], -1)

        beta = F.softmax(
            torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]), dim=-1
        )
        output = torch.bmm(beta, value).reshape(x.shape)
        return output