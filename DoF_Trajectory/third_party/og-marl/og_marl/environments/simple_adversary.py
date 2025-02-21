"""Wraper for Simple adversary."""
from typing import Dict, List, Union

from dm_env import specs
import numpy as np
from pettingzoo.mpe import  simple_adversary_v3
from og_marl.environments.base import OLT
from og_marl.environments.pettingzoo_base import PettingZooBase

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

class SimpleAdversary(PettingZooBase):
    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(
        self,
    ):
        """Constructor for parallel PZ wrapper.

        Args:
            environment (ParallelEnv): parallel PZ env.
            env_preprocess_wrappers (Optional[List], optional): Wrappers
                that preprocess envs.
                Format (env_preprocessor, dict_with_preprocessor_params).
            return_state_info: return extra state info
        """
        self._environment = simple_adversary_v3.parallel_env(render_mode="rgb_array")

        self.num_actions = 5
        self.num_agents = 3
        self._agents = self._environment.possible_agents
        self._reset_next_step = True
        self._done = False
        self.environment_label = "pettingzoo/simple_adversary"

    def _create_state_representation(self, observations):

        observations_list = []
        for agent in self._agents:
            agent_obs = observations[agent]

            if agent_obs.shape[0] == 8:
                agent_obs = np.concatenate((agent_obs, np.zeros((2,), "float32")), axis=-1)

            observations_list.append(agent_obs)

        state = np.concatenate(
            observations_list, axis=-1
        )

        return state

    def _add_zero_obs_for_missing_agent(self, observations):
        for agent in self._agents:
            if agent not in observations:
                observations[agent] = np.zeros((10,), "float32")
        return observations

    def _convert_observations(
        self, observations: List, done: bool
    ):
        """Convert SMAC observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        olt_observations = {}
        for i, agent in enumerate(self._agents):

            agent_obs = observations[agent]

            if agent_obs.shape[0] == 8: # Zero-pad observations so that they are all same size
                agent_obs = np.concatenate((agent_obs, np.zeros((2,), "float32")), axis=-1)

            olt_observations[agent] = OLT(
                observation=agent_obs,
                legal_actions=np.ones(5, "float32"),  # five actions in adversary, all legal
                terminal=np.asarray(done, dtype="float32"),
            )

        return olt_observations

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        state_spec = {"s_t": np.zeros((30,), "float32")}  # four stacked frames

        return state_spec

    def observation_spec(self) -> Dict:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_specs = {}
        for agent in self._agents:

            obs = np.zeros((10,), "float32")

            observation_specs[agent] = OLT(
                observation=obs,
                legal_actions=np.ones(5, "float32"),
                terminal=np.asarray(True, "float32"),
            )

        return observation_specs

    def action_spec(
        self,
    ) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=5, dtype="int64"
            )
        return action_specs
    
