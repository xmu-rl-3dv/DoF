import numpy
import importlib
import importlib
from typing import Callable, List, Optional

import numpy as np
import torch
import os

from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.preprocessing import get_preprocess_fn




import json





def veiw_shape_npy(path):

    
   
    data6 = numpy.load(os.path.join(path, "seed_2_data/rews_0.npy"))
    data7 = numpy.load(os.path.join(path, "seed_2_data/rews_1.npy"))
    data8 = numpy.load(os.path.join(path, "seed_2_data/rews_2.npy"))
     
    print(data6[0:25])
    print(data7[0:25])
    print(data8[0:25])
    
        
    
    print("Done.")
    
    
    


from multiagent_mujoco.mujoco_multi import MujocoMulti
import numpy as np
import time

def HalfCheetah_env_show_main():
    print("**************************************")
    print("HalfCheetah_env_show_main")
    print("**************************************")
    env_args = {"scenario": "HalfCheetah-v2",
                  "agent_conf": "2x3",
                  "agent_obsk": 1,
                  "episode_limit": 1000,
                  
                  "global_categories": "qvel,qpos",
                  
                  }
    env = MujocoMulti(env_args=env_args)
    env_info = env.get_env_info()
    

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    n_episodes = 2

    for e in range(n_episodes):
        obs = env.reset()
        terminated = False
        episode_reward = 0
        scores = []
        while not np.all(terminated):
            obs = env.get_obs()
            state = env.get_state()
            
            
            obs = np.expand_dims(obs, axis=0)
            obs = torch.tensor(obs, dtype=torch.float32)
            
            obs_rpt = torch.repeat_interleave(obs, repeats = 50, dim=0, )
            print(obs.shape)
            print(obs_rpt.shape)
            
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.uniform(-1.0, 1.0, n_actions)
                actions.append(action)
            
            print(actions)
            print("actions.shape:")
            print(np.array(actions).shape)
            reward, terminated, _, _ = env.step(actions)
            episode_reward += reward
        
            time.sleep(0.1)
            
        scores.append(episode_reward)
        avg_reward = np.mean(scores)
        
        print("Average reward = {}".format(avg_reward))

    env.close()
    
    
def Ant_env_show_main():
    print("**************************************")
    print("Ant_env_show_main")
    print("**************************************")
    env_args = {"scenario": "Ant-v2",
                  "agent_conf": "4x2",
                  "agent_obsk": 1,
                  "episode_limit": 1000,
                  
                  "k_categories" : "qpos,qvel,cfrc_ext|qpos",
                  }
    env = MujocoMulti(env_args=env_args)
    env_info = env.get_env_info()
    

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    n_episodes = 1

    for e in range(n_episodes):
        obs = env.reset()
        terminated = False
        episode_reward = 0
        scores = []
        while not np.all(terminated):
            obs = env.get_obs()
            state = env.get_state()
            
            
            obs = np.expand_dims(obs, axis=0)
            obs = torch.tensor(obs, dtype=torch.float32)
            
            obs_rpt = torch.repeat_interleave(obs, repeats = 50, dim=0, )
            print(obs.shape)
            print(obs_rpt.shape)
            
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.uniform(-1.0, 1.0, n_actions)
                actions.append(action)
            
            
            print("actions.shape:")
            print(np.array(actions).shape)
            reward, terminated, _, _ = env.step(actions)
            episode_reward += reward
        
            time.sleep(0.1)
            
        scores.append(episode_reward)
        avg_reward = np.mean(scores)
        
        print("Average reward = {}".format(avg_reward))

    env.close()

if __name__ == "__main__":
    veiw_shape_npy(path = "./diffuser/datasets/data/mpe/simple_spread/expert")
    
    