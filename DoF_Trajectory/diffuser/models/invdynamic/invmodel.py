import torch
import torch.nn as nn

class JointInvModel(nn.Module):
    def __init__(self, n_agents, observation_dim, hidden_dim, output_dim):
        super(JointInvModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_agents * (2 * observation_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * output_dim)
        )

    def forward(self, x):
        return self.model(x)

class SharedInvModel(nn.Module):
    def __init__(self, observation_dim, hidden_dim, output_dim):
        super(SharedInvModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class IndependentInvModel(nn.Module):
    def __init__(self, n_agents, observation_dim, hidden_dim, output_dim, discrete_action=False):
        super(IndependentInvModel, self).__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Softmax(dim=-1) if discrete_action else nn.Identity()
            ) for _ in range(n_agents)
        ])

    def forward(self, x):   
        return torch.stack([model(agent_x) for model, agent_x in zip(self.models, x.transpose(0, 1))], dim=1)
