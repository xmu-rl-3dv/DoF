import os

import numpy as np
from third_party.multiagent_mujoco.src.multiagent_mujoco.mujoco_multi import MujocoMulti
import collections

import gym

    
    
def load_environment(name):
    if type(name) is not str:
        
        return name

    idx = name.find("-")
    env_name, data_split = name[:idx], name[idx + 1 :]

    env_kwargs = {
        "agent_obsk": 1,
        "episode_limit": 1000,
        "global_categories": "qvel,qpos",
    }
    if env_name == "4ant":
        env_kwargs["scenario"] = "Ant-v2"
        env_kwargs["agent_conf"] = "4x2"
    elif env_name == "2ant":
        env_kwargs["scenario"] = "Ant-v2"
        env_kwargs["agent_conf"] = "2x4"
    elif env_name == "2halfcheetah":
        env_kwargs["scenario"] = "HalfCheetah-v2"
        env_kwargs["agent_conf"] = "2x3"
    else:
        raise NotImplementedError(
            f"Multi-agent Mujoco environment {env_name} not supported."
        )

    env = MujocoMulti(env_args=env_kwargs, )
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    env.metadata["global_feats"] = []
    
    
    return env
    


def policy_dataset(env, preprocess_fn):
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "data/mamujoco",
        env.metadata["name"],
        env.metadata["data_split"],
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))

    observations = np.load(os.path.join(dataset_path, "obs.npy"))
    
    rewards = np.load(os.path.join(dataset_path, "rewards.npy"))
    actions = np.load(os.path.join(dataset_path, "actions.npy"))
    path_lengths = np.load(os.path.join(dataset_path, "path_lengths.npy"))

    
    start = 0
    for path_length in path_lengths:
        end = start + path_length
        episode_datas = {}
        episode_datas["observations"] = observations[start:end]    
        episode_datas["rewards"] = rewards[start:end]
        episode_datas["actions"] = actions[start:end]
        episode_datas["terminals"] = np.zeros(
            (path_length, observations.shape[1]), dtype=float
        )
        episode_datas["terminals"][-1] = True
        for i in range(0, len(episode_datas["observations"])): 
            episode_data = {}
            episode_data["observation"] = episode_datas["observations"][i]
            episode_data["reward"] = episode_datas["rewards"][i]
            episode_data["action"] = episode_datas["actions"][i]
            episode_data["terminal"] = episode_datas["terminals"][i]
            if np.any(episode_data["terminal"]):
                episode_data["next_observation"] = episode_datas["observations"][i]
            else:
                episode_data["next_observation"] = episode_datas["observations"][i+1]
            
            yield episode_data
        
        start = end