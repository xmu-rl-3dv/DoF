from models import ma_diffusion
from models import ma_model
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from utils.arrays import apply_dict, batch_to_device, to_device, to_np
from models.helpers import EMA
from models.ma_diffusion import *
from torch.optim.lr_scheduler import CosineAnnealingLR


def cycle(dl):
        while True:
            for data in dl:
                yield data

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class MA_Diffusion_agent(object):
    def __init__(
        self,
        obs_dim,
        action_dim,
        n_agents,
        model,
        policy,
        dataset,
        max_action,
        device,
        discount,
        tau,
        train_batch_size,
        max_q_backup=False, 
        eta=1.0,
        beta_schedule='linear',
        n_timesteps=100,
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=3e-4,
        lr_decay=False,
        lr_maxt=1000,
        grad_norm=0.5,
        flag_q_separate=True,
        
        data_factorization_mode="default", 
        flag_target_q=False,
    ):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.max_action = max_action
        self.device = device
        self.discount = discount
        self.tau = tau
        self.eta = eta
        self.max_q_backup=max_q_backup,
        
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        
        
        if dataset is not None:
            self.dataloader = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=self.train_batch_size,
                    num_workers=16,
                    shuffle=True,
                    pin_memory=True,
                )
            )
        
        
        self.model = ma_model.MA_MLP(
            obs_dim,
            action_dim,
            self.n_agents,
            device,
            
        )
        
        
        self.policy = ma_diffusion.MA_Diffusion(
            obs_dim,
            action_dim,
            n_agents,
            model,
            max_action,
            device,
        )
        
        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.policy)
        self.update_ema_every = update_ema_every
        

        self.grad_norm = grad_norm
        
        self.update_ema_every = update_ema_every

        self.MA_Critic = nn.ModuleList(
            [
                Critic(
                    obs_dim, 
                    action_dim
                ).float()
                for _ in range(self.n_agents)
                

            ]
        ).to(device)

        self.MA_Critic_target = copy.deepcopy(self.MA_Critic)

        self.MA_Critic_optimizers = [torch.optim.Adam(model.parameters(), lr=3e-4) for model in self.MA_Critic]
        self.Critic_optimizer = torch.optim.Adam(self.MA_Critic.parameters(), lr=3e-4)
        
        
        self.flag_q_separate = flag_q_separate  
        
        self.data_factorization_mode = data_factorization_mode
        self.flag_target_q = flag_target_q  
        
        
        self.policy_optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.policy.model.net]

        
        if self.data_factorization_mode == "w-concat":
            
            self.agent_weights = nn.Parameter(
                torch.ones(self.n_agents, device=device) / self.n_agents,
                requires_grad=True
            )
            self.policy_optimizer = torch.optim.Adam([
                {'params': self.policy.parameters()},
                {'params': self.agent_weights}
            ], lr=lr)
        else:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
            
        self.lr_decay = lr_decay
        if lr_decay:
            self.policy_lr_scheduler = CosineAnnealingLR(self.policy_optimizers, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.Critic_optimizer, T_max=lr_maxt, eta_min=0.)
        


    
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.policy)


    def train(self, iterations, curr_steps, log_writer=None):
        metric = {'bc_loss': [], 'ql_loss': [], 'policy_loss': []}
        
        
        
        for _ in range(iterations):
            
            batch = next(self.dataloader)
            batch = batch_to_device(batch, device=self.device)
            obs = batch['observation']
            
            next_obs = batch['next_observation']
            actions = batch['action']  
            rewards = batch['reward']
            not_dones = 1. - batch['terminal']

            obs_agents = [obs[:, i] for i in range(obs.shape[1])]
            next_obs_agents = [next_obs[:, i] for i in range(next_obs.shape[1])]
            actions_agents = [actions[:, i] for i in range(actions.shape[1])]
            rewards_agents = [rewards[:, i] for i in range(rewards.shape[1])]
            not_dones_agents = [not_dones[:, i] for i in range(not_dones.shape[1])]  
            
            
            
        

            """  <<<CRITIC TRAINING  """
            if self.flag_q_separate:
                
                next_actions = self.ema_model(next_obs)
                next_action_agents = [next_actions[:, i] for i in range(next_actions.shape[1])]
                
                
                for a_i in range(self.n_agents):
                    obs_agent = obs_agents[a_i]
                    next_obs_agent = next_obs_agents[a_i]
                    actions_agent = actions_agents[a_i]
                    rewards_agent = rewards_agents[a_i]
                    not_dones_agent = not_dones_agents[a_i]
                    
                    next_action_agent = next_action_agents[a_i]
                    
                    
                    current_q1, current_q2 = self.MA_Critic[a_i](obs_agent.to(torch.float32), actions_agent.to(torch.float32))
                    
                    target_q1, target_q2 = self.MA_Critic_target[a_i](next_obs_agent.to(torch.float32), next_action_agents[a_i].to(torch.float32))
                    target_q_next = torch.min(target_q1, target_q2)
                    
                    target_q = (rewards_agent.to(torch.float32).unsqueeze(-1) + self.discount * not_dones_agent.to(torch.float32).unsqueeze(-1) * target_q_next).detach()

                    critic_a_i_loss = F.mse_loss(current_q1, target_q.detach()) + F.mse_loss(current_q2, target_q.detach())
                    critic_a_i_optimizer = self.MA_Critic_optimizers[a_i]
                    critic_a_i_optimizer.zero_grad()
                    critic_a_i_loss.backward()
                    
                    if self.grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.MA_Critic[a_i].parameters(), max_norm=self.grad_norm, norm_type=2)
                    
                    critic_a_i_optimizer.step()

            else:
                next_actions = self.ema_model(next_obs)
                next_action_agents = [next_actions[:, i] for i in range(next_actions.shape[1])]
                critic_loss = 0.
                current_Q1 = current_Q2 = target_Q = 0.
                
                for a_i in range(self.n_agents):
                    obs_agent = obs_agents[a_i]
                    next_obs_agent = next_obs_agents[a_i]
                    actions_agent = actions_agents[a_i]
                    rewards_agent = rewards_agents[a_i]
                    not_dones_agent = not_dones_agents[a_i]
                    
                    next_action_agent = next_action_agents[a_i]
                    
                    current_q1, current_q2 = self.MA_Critic[a_i](obs_agent.to(torch.float32), actions_agent.to(torch.float32))
                    
                    current_Q1 += current_q1
                    current_Q2 += current_q2
                    
                    if self.flag_target_q:
                        target_q1, target_q2 = self.MA_Critic_target[a_i](next_obs_agent.to(torch.float32), next_action_agent.to(torch.float32))
                        target_q_next = torch.min(target_q1, target_q2)
                        
                        target_q = (rewards_agent.to(torch.float32).unsqueeze(-1) + self.discount * not_dones_agent.to(torch.float32).unsqueeze(-1) * target_q_next).detach()

                        target_Q += target_q
                        
                    else:
                        current_q = torch.min(current_q1, current_q2)
                        target_q = (rewards_agent.to(torch.float32).unsqueeze(-1) + self.discount * not_dones_agent.to(torch.float32).unsqueeze(-1) * current_q).detach()

                        target_Q += target_q
                
                critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach())
                critic_optimizer = self.Critic_optimizer
                critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.MA_Critic.parameters(), max_norm=self.grad_norm, norm_type=2)
                critic_optimizer.step()
            
            """  >>>END CRITIC TRAINING  """
            
            
            """  <<<POLICY TRAINING  """
            
            
            if self.data_factorization_mode == "default": 
                for a_i in range(self.n_agents):
                    self.policy_optimizers[a_i].zero_grad()
                new_actions = self.policy(obs)  
                new_actions_agents = [new_actions[:, i] for i in range(new_actions.shape[1])] 
                q_loss = 0.
                bc_loss = 0.
                policy_loss = 0.
                for a_i in range(self.n_agents):
                    obs_agent = obs_agents[a_i]
                    new_actions_agent = new_actions_agents[a_i]
                    q1_i_new_action, q2_i_new_action = self.MA_Critic[a_i](obs_agent, new_actions_agent)
                    if np.random.uniform() > 0.5:
                        q_i_loss = - q1_i_new_action.mean().detach() / (q2_i_new_action.clone()).abs().mean().detach()
                    else:
                        q_i_loss = - q2_i_new_action.mean().detach() / (q1_i_new_action.clone()).abs().mean().detach()
                   
                    bc_loss_i = self.policy.loss_i(actions.clone(), obs.clone(), a_i)
                    
                    
                    policy_i_loss = bc_loss_i + self.eta * q_i_loss
                    policy_i_optimizer = self.policy_optimizers[a_i]
                    policy_i_loss.backward(retain_graph=True)
                    if self.grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.policy.model.net[a_i].parameters(), max_norm=self.grad_norm, norm_type=2)
                    policy_i_optimizer.step()
                    q_loss += q_i_loss
                    policy_loss += policy_i_loss
                    bc_loss += bc_loss_i
            else: 
                bc_loss = self.policy.loss(actions, obs)
                new_actions = self.policy(obs)  
                if self.data_factorization_mode == "concat":
                    new_actions_agents = [new_actions[:, i] for i in range(new_actions.shape[1])] 
                
                
                else:
                    weighted_actions = new_actions * self.agent_weights.view(1, -1, 1).to(new_actions.device)
                    new_actions_concat = weighted_actions.sum(dim=1)
                    new_actions_agents = [new_actions_concat for _ in range(self.n_agents)]
                    
                q_loss = 0.
                policy_loss = 0.
                policy_i_loss_list = []
                q_i_loss_list = []
                for a_i in range(self.n_agents):
                    obs_agent = obs_agents[a_i]
                    new_actions_agent = new_actions_agents[a_i]
                    q1_i_new_action, q2_i_new_action = self.MA_Critic[a_i](obs_agent, new_actions_agent)
                    if np.random.uniform() > 0.5:
                        q_i_loss = - q1_i_new_action.mean().detach() / (q2_i_new_action.clone()).abs().mean().detach()
                    else:
                        q_i_loss = - q2_i_new_action.mean().detach() / (q1_i_new_action.clone()).abs().mean().detach()
                     
                    policy_i_loss_list.append(bc_loss.clone() + self.eta * q_i_loss)
                    q_i_loss_list.append(q_i_loss)
                    
                    
                
                
                for a_i in range(self.n_agents):
                    
                    
                    self.policy_optimizers[a_i].zero_grad()
                    policy_i_loss_list[a_i].backward(retain_graph=True)
                    if self.grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.policy.model.net[a_i].parameters(), max_norm=self.grad_norm, norm_type=2)
                    
                for a_i in range(self.n_agents):
                    self.policy_optimizers[a_i].step()
                                        
                policy_loss = sum(policy_i_loss_list)
                q_loss = sum(q_i_loss_list)
            """  >>>END POLICY TRAINING  """
            
            
            
            """  <<<UPDATE TARGET NETWORKS  """
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.MA_Critic.parameters(), self.MA_Critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            self.step += 1
            torch.cuda.empty_cache()
            """  >>>END UPDATE TARGET NETWORKS  """
            metric['policy_loss'] = policy_loss.item()
            metric['bc_loss'] = bc_loss.item()
            metric['ql_loss'] = q_loss.item()
            
            """  LOGGING  """
        
        if self.lr_decay: 
            self.policy_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            
        return metric









    def step_update(self, observations):
        
        action = []
        
        observations = torch.FloatTensor(observations).to(self.device)
        observations_rpt = torch.repeat_interleave(observations, repeats = 50, dim = 0)
        
        with torch.no_grad():
            
            actions_rpt_to_critic = self.policy(observations_rpt)
            q_max_action = False
            if q_max_action:
                observations_rpt_to_critic_agents = [observations_rpt[:, i] for i in range(observations_rpt.shape[1])]
                action_rpt_to_critic_agents = [actions_rpt_to_critic[:, i] for i in range(actions_rpt_to_critic.shape[1])]
                for a_i in range(self.n_agents):
                    q1_a_i_step, q2_a_i_step = self.MA_Critic[a_i](observations_rpt_to_critic_agents[a_i], action_rpt_to_critic_agents[a_i])
                    q_a_i_value = torch.min(q1_a_i_step, q2_a_i_step)
                    q_a_i_value = torch.flatten(q_a_i_value)
                    idx = torch.multinomial(F.softmax(q_a_i_value, dim=-1), 1)
                    action.append(action_rpt_to_critic_agents[a_i][idx])
            else:
                idx = np.random.randint(0, 49)
                actions = actions_rpt_to_critic[idx]
                for a_i in range(self.n_agents):
                    action_a_i = actions[a_i].clamp(-1, 1)
                    action.append(action_a_i)
                
        return action





    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.policy.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.MA_Critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.policy.state_dict(), f'{dir}/actor.pth')
            torch.save(self.MA_Critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.policy.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.MA_Critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.policy.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.MA_Critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


