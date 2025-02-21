import argparse
import torch
import time
import os, sys, tempfile
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from diffuser.utils.eval_env_wrappers import SubprocVecEnv, DummyVecEnv
import json
import datetime
import random
import copy
import shutil
import yaml
import h5py
from diffuser.utils.utils import print_banner
try:
    from third_party.multiagent_mujoco.src.multiagent_mujoco.mujoco_multi import MujocoMulti
except:
    print ('MujocoMulti not installed')
import diffuser.utils as utils
from diffuser.models.ma_diffusion import MA_Diffusion
from diffuser.models.ma_diffusion_agent import MA_Diffusion_agent
from diffuser.models.ma_model import MA_MLP
import importlib
from typing import Callable, List, Optional
from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.datasets.policy_mpe import make_eval_env




def policy_dataset_gen_mamujoco(env_type, env):
    
    preprocess_fns: List[Callable] = []
    
    env_mod_name = {
            "mamujoco": "diffuser.datasets.policy_mamujoco",   
            "mpe": "diffuser.datasets.policy_mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[env_type]
    env_mode = importlib.import_module(env_mod_name)
    preprocess_fn = get_preprocess_fn(preprocess_fns, env)
    env = env_mode.load_environment(env)
    itr = env_mode.policy_dataset(env, preprocess_fn)
        
    return itr

def policy_dataset_gen_mpe(env_type, env, use_seed_dataset, seed):
    
    preprocess_fns: List[Callable] = []
    
    env_mod_name = {
            "mamujoco": "diffuser.datasets.policy_mamujoco",   
            "mpe": "diffuser.datasets.policy_mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[env_type]
    env_mode = importlib.import_module(env_mod_name)
    
    preprocess_fn = get_preprocess_fn(preprocess_fns, env)
    env = env_mode.load_environment(env)
    
    if env_type == "mpe":
        if use_seed_dataset:
            itr = env_mode.policy_dataset(env, preprocess_fn, seed=seed)
            dataset = list(itr)
        else:
            itr = env_mode.policy_dataset(env, preprocess_fn)
            dataset = list(itr)
    else:
        return NotImplementedError
    return dataset


def train_policy(Config):

    # set dataset
    if Config.env_type != "mpe":
        dataset = policy_dataset_gen_mamujoco(Config.env_type, Config.dataset,)
        
    else:
        dataset = policy_dataset_gen_mpe(Config.env_type, Config.dataset, Config.use_seed_dataset, Config.mpe_seed)


    obs_dim = dataset[0]['observation'].shape[1]
    action_dim = dataset[0]['action'].shape[1]
    
    
    # set ma_model
    model = MA_MLP(
        
        obs_dim=obs_dim,
        action_dim=action_dim,
        # device=Config.device,
        n_agents=Config.n_agents,
        concat=True,
        
    ).to(Config.device)
    
    # set ma_diffusion
    ma_diffusion = MA_Diffusion(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=Config.n_agents,
        model=model,
        max_action=Config.max_action,
        device=Config.device,
        beta_schedule=Config.beta_schedule,
        n_timesteps=Config.n_timesteps,
        noise_factorization_mode=Config.noise_factorization_mode,
    )
    
    # set ma_diffusion_agent
    ma_diffusion_agent = MA_Diffusion_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=Config.n_agents,
        model=model,
        policy=ma_diffusion,
        dataset=dataset,
        max_action=Config.max_action,
        device=Config.device,
        discount=Config.discount,
        tau=Config.tau,
        train_batch_size=Config.train_batch_size,
        eta=Config.eta,
        lr=Config.lr,
        # flag_policy_separate=Config.if_p_separate,
        data_factorization_mode=Config.data_factorization_mode,
        flag_q_separate=Config.if_q_separate,
        flag_target_q=Config.if_target_q,
        
    )
    
    if Config.env_name in ["HalfCheetah-v2"]:
        
        env_args = {
            "scenario": "HalfCheetah-v2",
            "agent_conf": "2x3",
            "agent_obsk": 1,
            "global_categories": "qvel,qpos",
            "episode_limit": 1000
            }
        
    elif Config.env_name in ["Ant-v2"]:
        
        env_args = {
            "scenario": "Ant-v2",
            "agent_conf": "4x2",
            "agent_obsk": 1,
            "global_categories": "qvel,qpos",
            "episode_limit": 1000
            }
        
    elif Config.dataset == "simple_spread-medium" or Config.dataset == "simple_spread-expert" or Config.dataset == "simple_spread-random" or Config.dataset == "simple_spread-medium-replay":
        env_args = {
            "scenario": "simple_spread",
            "eval_episode_lenth": 25,
            "seed": Config.seed,
        }
    elif Config.dataset == "simple_tag-medium" or Config.dataset == "simple_tag-expert" or Config.dataset == "simple_tag-random" or Config.dataset == "simple_tag-medium-replay":
        env_args = {
            "scenario": "simple_tag",
            "eval_episode_lenth": 25,
            "seed" : Config.seed,
        }
    elif Config.dataset == "simple_world-medium" or Config.dataset == "simple_world-expert" or Config.dataset == "simple_world-random" or Config.dataset == "simple_world-medium-replay":
        env_args = {
            "scenario": "simple_world",
            "eval_episode_lenth": 25,
            "seed" : Config.seed,
        }
   
    
    result_dir = os.path.join(Config.dir, "policy_result")
    if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    files = os.listdir(result_dir)
    
    
    reward_dir = os.path.join(result_dir, "reward")
    if not os.path.exists(reward_dir):
        os.makedirs(reward_dir)
    
    with open(os.path.join(reward_dir, f"result_{Config.dataset}_{Config.exp_num}_{Config.data_factorization_mode}_{Config.noise_factorization_mode}.txt"), "a") as f:
        f.write(json.dumps(Config.__dict__) + "\n")
    
    # ************************************
    # -----------Start training-----------
    # ************************************
    print_banner(f"Training Start", separator="*", num_star=90)
         
    for i in range(Config.eval_iterations):
        
        curr_step = int ((i+1) * Config.train_iterations_step)

        loss_matrix = ma_diffusion_agent.train(Config.train_iterations_step, curr_step)  
        
        average_ep_reward, std_ep_reward = eval_policy(ma_diffusion_agent, Config.env_name, env_args=env_args, eval_episodes=Config.eval_episodes, seed=Config.seed)
        
        curr_step_result = {
            "curr_step": curr_step,
            "average_ep_reward": average_ep_reward,
            "std_ep_reward": std_ep_reward,
        }
        
        file_count = len(files)
        
        # curr_step_result logging  (json.dump)
        with open(os.path.join(reward_dir, f"result_{Config.dataset}_{Config.exp_num}_{Config.data_factorization_mode}_{Config.noise_factorization_mode}.txt"), "a") as f:
            f.write(json.dumps(curr_step_result) + "\n")
        
        loss_dir = os.path.join(result_dir, "loss")
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
        with open(os.path.join(loss_dir, f"loss_{Config.dataset}_{Config.exp_num}_{Config.data_factorization_mode}_{Config.noise_factorization_mode}.txt"), "a") as f:
            f.write(json.dumps(loss_matrix) + "\n")



def eval_policy(agent, env_name, env_args, eval_episodes, seed):
    
    with torch.no_grad():
        # init env
        if env_name in ["HalfCheetah-v2", "Ant-v2"]:
            env = MujocoMulti(env_args = env_args)
            env.seed(seed + 100)
            
            episodes_rewards_sum = []
            scores = []
            
            for i in range(eval_episodes):
                
                env.reset()
                done = False
                episode_reward = 0
                episode_num = 0
                
                while not np.any(done): #
                    obs = env.get_obs()
                    obs = np.expand_dims(obs, axis=0)
                    obs = torch.tensor(obs, dtype=torch.float32)                    
                    action = agent.step_update(obs)
                    actions = [tensor.cpu() for tensor in action]
                    next_obs, reward, done, _ = env.step(actions)
                    episode_reward += reward
                      
                print("Total reward in episode {} = {}".format(i, episode_reward))
                scores.append(episode_reward)
                episode_num += 1
            
            avg_episode_reward = np.mean(scores)
            std_episode_reward = np.std(scores)
            print("Average reward = {}".format(avg_episode_reward))
            print("Std_reward = {}".format(std_episode_reward))
            
            
            return avg_episode_reward, std_episode_reward   

        else:
            
            scenario_name = env_args["scenario"]
            seed = env_args["seed"]
            
            env = make_eval_env(scenario_name=scenario_name)
            
            env.env.seed(seed = seed)
          
            scores = []
            for i in range(eval_episodes):
                
                obs = env.reset()
                done = False
                episode_reward = np.zeros(env.n)
                
                for et_i in range(env_args["eval_episode_lenth"]): 
                    
                    obs = np.expand_dims(obs, axis=0)
                    
                    action = agent.step_update(obs)
                    
                    action = [a.cpu() for a in action] 
                    next_obs, reward, done, _ = env.step(action)
                    
                    
                    episode_reward = reward + episode_reward
                    
                    obs = next_obs
                

                if i >= 0:
                    print("Total reward in episode {} = {}".format(i, episode_reward))
                    avg_agent_episode_reward = np.mean(episode_reward)
                
                    scores.append(avg_agent_episode_reward)
                
            avg_episode_reward = np.mean(scores)
            std_episode_reward = np.std(scores)  
            
            
            
            print("Std reward = {}".format(std_episode_reward))
            print("Average reward = {}".format(avg_episode_reward))

            
            env.close()   
            return avg_episode_reward, std_episode_reward   
    
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ConfigObject:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()  
    parser.add_argument("--config", type=str, default="Policy_exp_Config_yaml/mpe_spread_expert_seed0.yaml")
    
    config_path = parser.parse_args().config
    Config = ConfigObject(load_config(config_path))
    
    train_policy(Config)
    
