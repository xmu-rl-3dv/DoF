
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuser.models.helpers import SinusoidalPosEmb
from models.model import MLP




class MA_MLP(nn.Module):
    """
    MA_MLP Model
    """
    def __init__(
        self,
        obs_dim,
        action_dim,
        
        n_agents: int = 2,
        concat: bool = True
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.n_agents = n_agents
        
        
        self.concat = concat
        self.net = nn.ModuleList(
            [
                MLP(
                    state_dim=obs_dim,
                    action_dim=action_dim,
                    
                )
                for _ in range(int(n_agents))
            ]
            
        )
    
    
    def forward(self, x, time, obs):

        
        assert (
            x.shape[1] == self.n_agents
            and obs.shape[1] == self.n_agents
        ), "n_agents mismatch"
        
        x = [x[:, i] for i in range(x.shape[1])]  
        obs = [obs[:, i] for i in range(obs.shape[1])]
        
        action = []
        if self.concat:
            for i in range(self.n_agents):
                action.append(self.net[i](x[i], time, obs[i]))

        action = torch.stack(action, dim=1)  

        
        return action
        


    

