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

import tensorflow as tf
import os

def get_system(system_name, environment, logger, **kwargs) :
    if system_name == "idrqn":
        from og_marl.tf2.systems.idrqn import IDRQNSystem
        return IDRQNSystem(environment, logger, **kwargs)
    elif system_name == "idrqn+cql":
        from og_marl.tf2.systems.idrqn_cql import IDRQNCQLSystem
        return IDRQNCQLSystem(environment, logger, **kwargs)
    elif system_name == "idrqn+bcq":
        from og_marl.tf2.systems.idrqn_bcq import IDRQNBCQSystem
        return IDRQNBCQSystem(environment, logger, **kwargs)
    elif system_name == "qmix":
        from og_marl.tf2.systems.qmix import QMIXSystem
        return QMIXSystem(environment, logger, **kwargs)
    elif system_name == "qmix+cql":
        from og_marl.tf2.systems.qmix_cql import QMIXCQLSystem
        return QMIXCQLSystem(environment, logger, **kwargs)
    elif system_name == "maicq":
        from og_marl.tf2.systems.maicq import MAICQSystem
        return MAICQSystem(environment, logger, **kwargs)
    elif system_name == "qmix+bcq":
        from og_marl.tf2.systems.qmix_bcq import QMIXBCQSystem
        return QMIXBCQSystem(environment, logger, **kwargs)
    elif system_name == "iddpg":
        from og_marl.tf2.systems.iddpg import IDDPGSystem
        return IDDPGSystem(environment, logger, **kwargs)
    elif system_name == "iddpg+cql":
        from og_marl.tf2.systems.iddpg_cql import IDDPGCQLSystem
        return IDDPGCQLSystem(environment, logger, **kwargs)
    elif system_name == "omar":
        from og_marl.tf2.systems.omar import OMARSystem
        return OMARSystem(environment, logger, **kwargs)

def set_growing_gpu_memory() -> None:
    """Solve gpu mem issues."""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)


def gather(values, indices, axis=-1, keepdims=False):
    one_hot_indices = tf.one_hot(indices, depth=values.shape[axis])
    if len(values.shape) > 4:  # we have extra dim for distributional q-learning
        one_hot_indices = tf.expand_dims(one_hot_indices, axis=-1)
    gathered_values = tf.reduce_sum(
        values * one_hot_indices, axis=axis, keepdims=keepdims
    )
    return gathered_values


def switch_two_leading_dims(x):
    trailing_perm = []
    for i in range(2, len(x.shape)):
        trailing_perm.append(i)
    x = tf.transpose(x, perm=[1, 0, *trailing_perm])
    return x


def merge_batch_and_agent_dim_of_time_major_sequence(x):
    T, B, N = x.shape[:3]  # assume time major
    trailing_dims = x.shape[3:]
    x = tf.reshape(x, shape=(T, B * N, *trailing_dims))
    return x


def merge_time_batch_and_agent_dim(x):
    T, B, N = x.shape[:3]  # assume time major
    trailing_dims = x.shape[3:]
    x = tf.reshape(x, shape=(T * B * N, *trailing_dims))
    return x


def expand_time_batch_and_agent_dim_of_time_major_sequence(x, T, B, N):
    TNB = x.shape[:1]  # assume time major
    assert TNB == T * B * N
    trailing_dims = x.shape[1:]
    x = tf.reshape(x, shape=(T, B, N, *trailing_dims))
    return x


def expand_batch_and_agent_dim_of_time_major_sequence(x, B, N):
    T, NB = x.shape[:2]  # assume time major
    assert NB == B * N
    trailing_dims = x.shape[2:]
    x = tf.reshape(x, shape=(T, B, N, *trailing_dims))
    return x


def concat_agent_id_to_obs(obs, agent_id, N):
    is_vector_obs = len(obs.shape) == 1

    if is_vector_obs:
        agent_id = tf.one_hot(agent_id, depth=N)
    else:
        h, w = obs.shape[:2]
        agent_id = tf.zeros((h, w, 1), "float32") + (agent_id / N) + 1 / (2 * N)

    if not is_vector_obs and len(obs.shape) == 2:  # if no channel dim
        obs = tf.expand_dims(obs, axis=-1)

    obs = tf.concat([agent_id, obs], axis=-1)

    return obs


def batch_concat_agent_id_to_obs(obs):
    B, T, N = obs.shape[:3]  # batch size, timedim, num_agents
    is_vector_obs = len(obs.shape) == 4

    agent_ids = []
    for i in range(N):
        if is_vector_obs:
            agent_id = tf.one_hot(i, depth=N)
        else:
            h, w = obs.shape[3:5]
            agent_id = tf.zeros((h, w, 1), "float32") + (i / N) + 1 / (2 * N)
        agent_ids.append(agent_id)
    agent_ids = tf.stack(agent_ids, axis=0)

    # Repeat along time dim
    agent_ids = tf.stack([agent_ids] * T, axis=0)

    # Repeat along batch dim
    agent_ids = tf.stack([agent_ids] * B, axis=0)

    if not is_vector_obs and len(obs.shape) == 5:  # if no channel dim
        obs = tf.expand_dims(obs, axis=-1)

    obs = tf.concat([agent_ids, obs], axis=-1)

    return obs

def batched_agents(agents, batch_dict):

    batched_agents = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "truncations": [],
    }

    if f"{agents[0]}_legals" in batch_dict:
        batched_agents["legals"] = []

    for agent in agents:
        for key in batched_agents:
            batched_agents[key].append(batch_dict[agent + "_" + key])
    
    for key, value in batched_agents.items():
        batched_agents[key] = tf.stack(value, axis=2)

    batched_agents["mask"] = tf.convert_to_tensor(batch_dict["mask"], "float32")

    if "state" in batch_dict:
        batched_agents["state"] = tf.convert_to_tensor(batch_dict["state"], "float32")

    return batched_agents