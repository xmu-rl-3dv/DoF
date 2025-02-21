import importlib
from typing import Callable, List, Optional
from collections import namedtuple
import numpy as np
import torch

from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.preprocessing import get_preprocess_fn

RewardBatch = namedtuple("Batch", "trajectories conditions masks returns")
Batch = namedtuple("Batch", "trajectories conditions masks")
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        n_agents: int = 2,
        horizon: int = 64,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        use_action: bool = True,
        discrete_action: bool = False,
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        termination_penalty: float = 0,
        use_padding: bool = True,  
        discount: float = 0.99,
        returns_scale: float = 400.0,
        include_returns: bool = False,
        include_env_ts: bool = False,
        use_state: bool = False,
        history_horizon: int = 0,
        agent_share_parameters: bool = False,
        use_seed_dataset: bool = False,
        decentralized_execution: bool = False,
        use_inverse_dynamic: bool = True,
        seed: int = None,
    ):
        if use_seed_dataset:
            assert (
                env_type == "mpe"
            ), f"Seed dataset only supported for MPE, not {env_type}"

        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
            "mamujoco": "diffuser.datasets.mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[env_type]
        env_mod = importlib.import_module(env_mod_name)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = env_mod.load_environment(env)
        self.global_feats = env.metadata["global_feats"]

        self.use_inverse_dynamic = use_inverse_dynamic
        self.returns_scale = returns_scale
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None, None]
        self.use_padding = use_padding
        self.use_action = use_action
        self.discrete_action = discrete_action
        self.include_returns = include_returns
        self.include_env_ts = include_env_ts
        self.use_state = use_state
        self.decentralized_execution = decentralized_execution

        if env_type == "mpe":
            if use_seed_dataset:
                itr = env_mod.sequence_dataset(env, self.preprocess_fn, seed=seed)
            else:
                itr = env_mod.sequence_dataset(env, self.preprocess_fn)
        elif env_type == "smac" or env_type == "smacv2":
            itr = env_mod.sequence_dataset(env, self.preprocess_fn, use_state=use_state)
        else:
            itr = env_mod.sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents,
            max_n_episodes,
            max_path_length,
            termination_penalty,
            global_feats=self.global_feats,
        )
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
            global_feats=self.global_feats,
        )

        self.state_dim = fields.states.shape[-1] if self.use_state else 0
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1] if self.use_action else 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        self.indices = self.make_indices(fields.path_lengths)

        if self.discrete_action:
            
            self.normalize(
                keys=["states", "observations"] if self.use_state else ["observations"]
            )
        else:
            self.normalize()

        self.pad_future()
        if self.history_horizon > 0:
            self.pad_history()

        print(fields)

    def pad_future(self, keys=None):
        if keys is None:
            keys = ["normed_observations", "rewards"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

            if self.use_state:
                keys.append("normed_states")

        for key in keys:
            shape = self.fields[key].shape
            self.fields[key] = np.concatenate(
                [
                    self.fields[key],
                    np.zeros(
                        (shape[0], self.horizon - 1, *shape[2:]),
                        dtype=self.fields[key].dtype,
                    ),
                ],
                axis=1,
            )

    def pad_history(self, keys=None):
        if keys is None:
            keys = ["normed_observations", "rewards"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

            if self.use_state:
                keys.append("normed_states")

        for key in keys:
            shape = self.fields[key].shape
            self.fields[key] = np.concatenate(
                [
                    np.zeros(
                        (shape[0], self.history_horizon, *shape[2:]),
                        dtype=self.fields[key].dtype,
                    ),
                    self.fields[key],
                ],
                axis=1,
            )

    def normalize(self, keys: List[str] = None):
        """
        normalize fields that will be predicted by the diffusion model
        """
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]
            if self.use_state:
                keys.append("states")

        for key in keys:
            shape = self.fields[key].shape
            array = self.fields[key].reshape(shape[0] * shape[1], *shape[2:])
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(shape)

    def make_indices(self, path_lengths):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, path_length in enumerate(path_lengths):
            if self.use_padding:
                max_start = path_length - 1
            else:
                max_start = path_length - self.horizon
                if max_start < 0:
                    continue

            
            for start in range(max_start):
                end = start + self.horizon
                mask_end = min(end, path_length)
                indices.append((i, start, end, mask_end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations, agent_idx: Optional[int] = None):
        """
        condition on current observations for planning
        """

        ret_dict = {}
        if self.decentralized_execution:
            cond_observations = np.zeros_like(observations[: self.history_horizon + 1])
            cond_observations[:, agent_idx] = observations[
                : self.history_horizon + 1, agent_idx
            ]
            ret_dict["agent_idx"] = torch.LongTensor([[[agent_idx]]]) 
        else:
            cond_observations = observations[: self.history_horizon + 1]
        ret_dict[(0, self.history_horizon + 1)] = cond_observations
        return ret_dict

    def __len__(self):
        if self.decentralized_execution:
            return len(self.indices) * self.n_agents
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.decentralized_execution:
            path_ind, start, end, mask_end = self.indices[idx // self.n_agents]
            agent_idx = idx % self.n_agents
        else:
            path_ind, start, end, mask_end = self.indices[idx]
            agent_idx = None

        
        history_start = start
        start = history_start + self.history_horizon
        end = end + self.history_horizon
        mask_end = mask_end + self.history_horizon

        observations = self.fields.normed_observations[path_ind, history_start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]
        if self.use_state:
            states = self.fields.normed_states[path_ind, history_start:end]

        loss_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        loss_masks[self.history_horizon : mask_end - history_start] = 1.0
        if self.use_inverse_dynamic:
            if self.decentralized_execution:
                loss_masks[self.history_horizon, agent_idx] = 0.0
            else:
                loss_masks[self.history_horizon] = 0.0

        attention_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        attention_masks[self.history_horizon : mask_end - history_start] = 1.0
        if self.decentralized_execution:
            attention_masks[: self.history_horizon, agent_idx] = 1.0
        else:
            attention_masks[: self.history_horizon] = 1.0

        conditions = self.get_conditions(observations, agent_idx)
        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        batch = {
            "x": trajectories,
            "cond": conditions,
            "loss_masks": loss_masks,
            "attention_masks": attention_masks,
        }

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start : -self.horizon + 1]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch["returns"] = returns

        if self.include_env_ts:
            env_ts = (
                np.arange(history_start, start + self.horizon) - self.history_horizon
            )
            env_ts[np.where(env_ts < 0)] = self.max_path_length
            env_ts[np.where(env_ts >= self.max_path_length)] = self.max_path_length
            batch["env_ts"] = env_ts

        if self.use_state:
            batch["states"] = states

        if "legal_actions" in self.fields.keys:
            batch["legal_actions"] = self.fields.legal_actions[
                path_ind, history_start:end
            ]

        return batch


class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.include_returns == True

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        value_batch = {
            "x": batch["x"],
            "cond": batch["cond"],
            "returns": batch["returns"].mean(axis=-1),
        }
        return value_batch


class BCSequenceDataset(SequenceDataset):
    def __init__(
        self,
        env_type="d4rl",
        env="hopper-medium-replay",
        n_agents=2,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        agent_share_parameters=False,
    ):
        super().__init__(
            env_type=env_type,
            env=env,
            n_agents=n_agents,
            normalizer=normalizer,
            preprocess_fns=preprocess_fns,
            max_path_length=max_path_length,
            max_n_episodes=max_n_episodes,
            agent_share_parameters=agent_share_parameters,
            horizon=1,
            use_action=True,
            termination_penalty=0.0,
            use_padding=False,
            discount=1.0,
            include_returns=False,
        )

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end, _ = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        batch = {"observations": observations, "actions": actions}
        return batch
    
class HistoryCondSequenceDataset(SequenceDataset):
    def __init__(
        self,
        env_type="d4rl",
        env="hopper-medium-replay",
        n_agents=2,
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        use_action=True,
        discrete_action=False,
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        discount=0.99,
        returns_scale=1000,
        include_returns=False,
        history_horizon=4,
        agent_share_parameters=False,
        use_state=False,
        include_env_ts = False,
        use_seed_dataset = False,
        seed: int = None,
        use_inverse_dynamic = True,
        decentralized_execution = False
    ):
        assert (
            history_horizon > 0
        ), f"history_horizon {history_horizon} must be larger than zero, otherwise use SequenceDataset"

        if env_type == "d4rl":
            from .d4rl import load_environment, sequence_dataset
        elif env_type == "ma_mujoco":
            from .mamujoco import load_environment, sequence_dataset

            assert preprocess_fns == [], "MA Mujoco does not support preprocessing"
        elif env_type == "mpe":
            from .mpe import load_environment, sequence_dataset

            assert preprocess_fns == [], "MPE does not support preprocessing"
        elif env_type == "smac":
            from .smac_env import load_environment, sequence_dataset

            assert preprocess_fns == [], "SMAC does not support preprocessing"
        elif env_type == "smac-mat":
            from .smac_mat import load_environment, sequence_dataset

            assert preprocess_fns == [], "SMAC does not support preprocessing"
        elif env_type == "nba":
            from .nba import load_environment, sequence_dataset

            assert preprocess_fns == [], "NBA does not support preprocessing"

        else:
            raise NotImplementedError(env_type)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = (
            load_environment(env)
            if env_type != "nba"
            else load_environment(env, nba_hz=self.nba_hz)
        )

        self.returns_scale = returns_scale
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None, None]
        self.use_padding = use_padding
        self.use_action = use_action
        self.discrete_action = discrete_action
        self.include_returns = include_returns
        self.use_state = use_state
        self.include_env_ts = include_env_ts
        self.use_seed_dataset = use_seed_dataset
        self.seed = seed
        self.use_inverse_dynamic = use_inverse_dynamic
        self.decentralized_execution = decentralized_execution

        if env_type == "nba":
            itr = sequence_dataset(env, self.preprocess_fn, mode=env.metadata["mode"])
        else:
            itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents, max_n_episodes, max_path_length, termination_penalty
        )
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
        )

        self.observation_dim = fields.observations.shape[-1]
        self.state_dim = fields.states.shape[-1] if self.use_state else 0
        if self.use_action:
            self.action_dim = fields.actions.shape[-1]
        else:
            self.action_dim = 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        if self.discrete_action:
            
            self.normalize(keys=["observations"])
        elif env_type != "nba" or env != "test":
            self.normalize()
        else:
            print(
                "NBA evaluation doesn't need normalizer, use training normalizer instead"
            )

        if env_type != "nba":  
            if self.discrete_action:
                self.pad_history(keys=["normed_observations", "actions"])
            else:
                self.pad_history()

        if env_type == "nba":
            self.indices = self.nba_make_indices(
                fields.path_lengths,
                fields.player_idxs,
                fields.player_hoop_sides,
                horizon,
                history_horizon,
                test_partially=False if self.env.metadata["mode"] == "train" else True,
            )
        else:
            self.indices = self.make_indices(fields.path_lengths, horizon, history_horizon)

        print(fields)
        
        

    def pad_history(self, keys=None):
        if keys is None:
            keys = (
                ["normed_observations", "normed_actions"]
                if self.use_action
                else ["normed_observations"]
            )

        for key in keys:
            shape = self.fields[key].shape
            self.fields[key] = np.concatenate(
                [
                    np.zeros(
                        (shape[0], self.history_horizon, *shape[2:]),
                        dtype=self.fields[key].dtype,
                    ),
                    self.fields[key],
                ],
                axis=1,
            )

    def normalize(self, keys=None):
        """
        normalize fields that will be predicted by the diffusion model
        """

        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]

        for key in keys:
            array = self.fields[key].reshape(
                self.n_episodes * self.max_path_length, self.n_agents, -1
            )
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, self.n_agents, -1
            )

    def make_indices(self, path_lengths, horizon, history_horizon):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - 1)
            if not self.use_padding:
                max_start = min(max_start, path_length - 1)
            for start in range(max_start):
                end = start + horizon
                if not self.use_padding:
                    mask_end = min(start + horizon, path_length)
                else:
                    mask_end = min(start + horizon, self.max_path_length)
                indices.append(
                    (
                        i,
                        start,  
                        start
                        + history_horizon,  
                        end + history_horizon,  
                        mask_end + history_horizon,  
                    )
                )
        indices = np.array(indices)
        return indices

    def nba_make_indices(
        self,
        path_lengths,
        player_idxs,
        player_hoop_sides,
        horizon,
        history_horizon,
        test_partially=False,
    ):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        N = self.nba_eval_valid_samples
        partially_samps_per_gameid = int(np.ceil(N / len(path_lengths)))
        indices = []
        for i, path_length in enumerate(path_lengths):
            
            consistent = False
            max_start = min(path_length - 1, self.max_path_length - 1)
            if not self.use_padding:
                max_start = min(max_start, path_length - 1)
            gaps = max_start // partially_samps_per_gameid
            tot_indeces = (
                range(max_start)
                if test_partially == False
                else range(0, max_start, gaps)
            )
            for start in tot_indeces:
                end = start + horizon
                if not self.use_padding:
                    mask_end = min(start + horizon, path_length)
                else:
                    mask_end = min(start + horizon, self.max_path_length)
                
                
                if consistent == False:
                    if (
                        len(
                            np.unique(
                                player_idxs[i, start : mask_end + history_horizon]
                            )
                        )
                        == 10
                        and np.unique(
                            player_hoop_sides[i, start : mask_end + history_horizon],
                            axis=0,
                        ).shape[0]
                        == 1
                    ):
                        consistent = True
                        indices.append(
                            (
                                i,
                                start,
                                start + history_horizon,
                                end + history_horizon,
                                mask_end + history_horizon,
                            )
                        )
                else:
                    if np.all(
                        player_idxs[i, mask_end + history_horizon - 2]
                        == player_idxs[i, mask_end + history_horizon - 1]
                    ) and np.all(
                        player_hoop_sides[i, mask_end + history_horizon - 1]
                        == player_hoop_sides[i, mask_end + history_horizon - 2]
                    ):
                        indices.append(
                            (
                                i,
                                start,
                                start + history_horizon,
                                end + history_horizon,
                                mask_end + history_horizon,
                            )
                        )
                    else:
                        consistent = False
        indices = np.array(indices)
        return indices
    
    def get_conditions(self, observations, history_horizon):
        """
        condition on current observation for planning
        """
        
        return {(0, history_horizon + 1): observations[: history_horizon + 1]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, history_start, start, end, mask_end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, history_start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end] 
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]

        if mask_end < end:
            observations = np.concatenate(
                [
                    observations,
                    np.zeros(
                        (end - mask_end, self.n_agents, observations.shape[-1]),
                        dtype=observations.dtype,
                    ),
                ],
                axis=0,
            )
            if self.use_action:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (end - mask_end, self.n_agents, actions.shape[-1]),
                            dtype=actions.dtype,
                        ),
                    ],
                    axis=0,
                )

        
        masks = np.zeros((observations.shape[0], observations.shape[1], 1)) 

        masks[: mask_end - history_start] = 1.0

        conditions = self.get_conditions(observations, self.history_horizon)
        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        batch = {
            'x' : trajectories,
            "cond" : conditions,
            "loss_masks" : masks,
        }
        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            
            batch['returns'] = returns

        return batch