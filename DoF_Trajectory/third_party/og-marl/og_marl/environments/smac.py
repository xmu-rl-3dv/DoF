# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wraper for SMAC."""
from typing import Any, Dict, List, Optional, Union
import dm_env
import numpy as np
from dm_env import specs
from og_marl.environments.base import OLT, BaseEnvironment, parameterized_restart
from smac.env import StarCraft2Env


class SMAC(BaseEnvironment):
    """Environment wrapper SMAC."""

    def __init__(
        self,
        map_name: str,
    ):
        self._environment = StarCraft2Env(map_name=map_name, obs_last_action=False)
        self._agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.environment_label = f"smac_v1/{map_name}"
        self.num_agents = len(self._agents)
        self.num_actions = self._environment.n_actions
        self._reset_next_step = True
        self._done = False
        self.max_episode_length = self._environment.episode_limit

    def reset(self) -> dm_env.TimeStep:
        """Resets the env."""

        # Reset the environment
        self._environment.reset()
        self._done = False
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # Get observation from env
        observation = self.environment.get_obs()
        legal_actions = self._get_legal_actions()
        observations = self._convert_observations(
            observation, legal_actions, self._done
        )

        # Set env discount to 1 for all agents
        discount_spec = self.discount_spec()
        self._discounts = {agent: np.array(1.0, "float32") for agent in self._agents}

        # Set reward to zero for all agents
        rewards = {agent: np.array(0, "float32") for agent in self._agents}

        # State info
        state = self.get_state()
        extras = {"s_t": state}

        return parameterized_restart(rewards, self._discounts, observations), extras

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env."""

        # Possibly reset the environment
        if self._reset_next_step:
            return self.reset()

        # Convert dict of actions to list for SMAC
        smac_actions = []
        for agent in self._agents:
            smac_actions.append(actions[agent])

        # Step the SMAC environment
        reward, self._done, self._info = self._environment.step(smac_actions)

        # Get the next observations
        next_observations = self._environment.get_obs()
        legal_actions = self._get_legal_actions()
        next_observations = self._convert_observations(
            next_observations, legal_actions, self._done
        )

        # Convert team reward to agent-wise rewards
        rewards = {agent: np.array(reward, "float32") for agent in self.agents}

        # State info
        state = self.get_state()
        extras = {"s_t": state}

        if self._done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True

            # Discount on last timestep set to zero
            self._discounts = {
                agent: np.array(0.0, "float32") for agent in self._agents
            }
        else:
            self._step_type = dm_env.StepType.MID

        # Create timestep object
        timestep = dm_env.TimeStep(
            observation=next_observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

        return timestep, extras

    def _get_legal_actions(self) -> List:
        """Get legal actions from the environment."""
        legal_actions = []
        for i, _ in enumerate(self._agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return legal_actions

    def _convert_observations(
        self, observations: List, legal_actions: List, done: bool
    ):
        """Convert SMAC observation so it's dm_env compatible."""
        olt_observations = {}
        for i, agent in enumerate(self._agents):
            olt_observations[agent] = OLT(
                observation=observations[i],
                legal_actions=legal_actions[i],
                terminal=np.asarray([done], dtype=np.float32),
            )

        return olt_observations

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env."""
        return {"s_t": self._environment.get_state()}

    def observation_spec(self):
        """Observation spec."""
        observation_spec = np.zeros(self._environment.get_obs_size(), "float32")
        legal_actions_spec = np.zeros(self.num_actions, "float32")

        observation_specs = {}
        for i, agent in enumerate(self._agents):
            observation_specs[agent] = OLT(
                observation=observation_spec,
                legal_actions=legal_actions_spec,
                terminal=np.asarray([True], dtype=np.float32),
            )

        return observation_specs

    def action_spec(
        self,
    ) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec."""
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=self._environment.n_actions, dtype=int
            )
        return action_specs

    def get_stats(self) -> Optional[Dict]:
        """Return extra stats to be logged."""
        return self._environment.get_stats()
