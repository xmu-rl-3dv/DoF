import importlib
from typing import Callable, List, Optional

import numpy as np
import torch

from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.preprocessing import get_preprocess_fn
# from diffuser.utils.mask_generator import MultiAgentMaskGenerator


class PolicyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        n_agents: int = 2,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        use_action: bool = True,
        discrete_action: bool = False,
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        termination_penalty: float = 0,
       
        discount: float = 0.99,
        returns_scale: float = 400.0,
       
       
       
        agent_share_parameters: bool = False,
        use_seed_dataset: bool = False,
       
       
        use_zero_padding: bool = True,
        agent_condition_type: str = "single",
        pred_future_padding: bool = False,
        seed: Optional[int] = None,

       
        train_policy_only: bool = False,
        train_value_only: bool = True,
        train_both: bool = False,
    ):
        if use_seed_dataset:
            assert (
                env_type == "mpe"
            ), f"Seed dataset only supported for MPE, not {env_type}"

        assert agent_condition_type in ["single", "all", "random"], agent_condition_type
        self.agent_condition_type = agent_condition_type

        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
            "mamujoco": "diffuser.datasets.policy_mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[env_type]
        env_mod = importlib.import_module(env_mod_name)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = env_mod.load_environment(env)
        self.global_feats = env.metadata["global_feats"]

       
        self.returns_scale = returns_scale
        self.n_agents = n_agents
       
       
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None, None]
       
        self.use_action = use_action
        self.discrete_action = discrete_action
       
       
       
        self.use_zero_padding = use_zero_padding
        self.agent_share_parameters = agent_share_parameters
       

        if env_type == "mpe":
            if use_seed_dataset:
                itr = env_mod.policy_dataset(env, self.preprocess_fn, seed=seed)
            else:
                itr = env_mod.policy_dataset(env, self.preprocess_fn)
        elif env_type == "smac" or env_type == "smacv2":
            itr = env_mod.policy_dataset(env, self.preprocess_fn)
       
        else:
            itr = env_mod.policy_dataset(env, self.preprocess_fn)

       
        
       
        fields = ReplayBuffer(
            n_agents,
            max_n_episodes,
            max_path_length,
            termination_penalty,
            global_feats=self.global_feats,
           
        )
        for _, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
            global_feats=self.global_feats,
        )
        
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1] if self.use_action else 0
        self.fields = fields
        self.n_episodes = fields.n_episodes

        self.train_policy_only = train_policy_only
        self.train_value_only = train_value_only
        self.train_both = train_both
        
        if self.discrete_action:
            self.normalize(["observations"])
        else:
            self.normalize()

        print("####################")
        print("PLANNER'S DATASET:")
        print(fields)
        print("####################")
    
    def normalize(self, keys: List[str] = None):
        """
        normalize fields that will be predicted by the diffusion model
        """
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]

        for key in keys:
            shape = self.fields[key].shape
            array = self.fields[key].reshape(shape[0] * shape[1], *shape[2:])
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(shape)


    def __len__(self):
        return self.fields.n_episodes

    def __getitem__(self, idx: int):
        batch = {}
        observations = self.fields.normed_observations[idx]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[idx]
            else:
                actions = self.fields.normed_actions[idx]

        batch = {
            "x": actions,
            "obs": observations,
        }

        if "legal_actions" in self.fields.keys:
            batch["legal_actions"] = self.fields.legal_actions[
                path_ind, history_start:end
            ]
        if self.train_policy_only:
            return batch

        if self.train_value_only:
            rewards = self.fields.rewards[idx]
            next_observations = self.fields.observations[idx]
            terminals = self.fields.terminals[idx]
            batch["next_obs"] = next_observations
            batch["rewards"] = rewards
            batch["not_done"] = 1. - terminals
            return batch
