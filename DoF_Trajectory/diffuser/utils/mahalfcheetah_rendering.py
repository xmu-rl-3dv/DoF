import warnings
from copy import copy

import gym
import matplotlib.pyplot as plt
import mujoco_py as mjc
import numpy as np
from ml_logger import logger

from diffuser.datasets.mahalfcheetah import load_environment

from .arrays import to_np
from .video import save_video, save_videos



def env_map(env_type, env_name):
    """
    map D4RL dataset names to custom fully-observed
    variants for rendering
    """

    assert env_type == "mahalfcheetah", env_type
    if "HalfCheetah" in env_name:
        return "HalfCheetahFullObs-v2"
    else:
        raise NotImplementedError(env_type)




def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask


def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x


def update_agent_obs_to_states(env, env_states, agent_observations):
    # NOTE: only support ma halfcheetah now
    assert (
        len(env_states) == agent_observations.shape[0]
    ), f"{len(env_states)} != {agent_observations.shape[0]}"
    env_states = copy(env_states)

    k_categories = env.k_categories
    for agent_idx in range(env.n_agents):
        observations = agent_observations[:, agent_idx]
        k_dict = env.k_dicts[agent_idx]

        cnt = 0
        for k in sorted(list(k_dict.keys())):
            cats = k_categories[k]
            for _t in k_dict[k]:
                for c in cats:
                    dim = getattr(_t, "{}_ids".format(c))
                    env_states[:, dim] = observations[:, cnt]
                    cnt += 1

    return env_states



class MAHalfCheetahRenderer:
    """
    default ma halfcheetah renderer
    """

    def __init__(self, env_type, env):
        if type(env) is str:
            self.ma_env = load_environment(env)
            env = env_map(env_type, env)
            self.env = gym.make(env)
        else:
            self.env = env
        self.initial_state = self.ma_env.get_state()

        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print(
                "[ utils/rendering ] Warning: could not initialize offscreen renderer"
            )
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate(
            [
                np.zeros(1),
                observation,
            ]
        )
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size

        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate(
            [
                xpos[:, None],
                observations,
            ],
            axis=-1,
        )
        return states

    def render(
        self,
        observation,
        dim=256,
        partial=False,
        qvel=True,
        render_kwargs=None,
        conditions=None,
    ):
        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                "trackbodyid": 2,
                "distance": 3,
                "lookat": [xpos, -0.5, 1],
                "elevation": -20,
            }

        for key, val in render_kwargs.items():
            if key == "lookat":
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):
        render_kwargs = {
            "trackbodyid": 2,
            "distance": 10,
            "lookat": [5, 2, 0.5],
            "elevation": 0,
        }
        images = []
        for path in paths:

            env_states = self.initial_state.reshape(1, -1).repeat(path.shape[0], axis=0)
            path = update_agent_obs_to_states(self.ma_env, env_states, path)
            path = atmost_2d(path)
            img = self.renders(
                to_np(path),
                dim=dim,
                partial=True,
                qvel=True,
                render_kwargs=render_kwargs,
                **kwargs,
            )
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            fig = plt.figure()
            plt.imshow(images)
            logger.savefig(savepath, fig)
            print(f"Saved {len(paths)} samples to: {savepath}")

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list:
            states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):

        observations_real = rollouts_from_state(self.env, state, actions)

        observations_real = observations_real[:, :-1]

        images_pred = np.stack(
            [self._renders(obs_pred, partial=True) for obs_pred in observations_pred]
        )

        images_real = np.stack(
            [self._renders(obs_real, partial=False) for obs_real in observations_real]
        )

        # [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        """
        diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        """
        render_kwargs = {
            "trackbodyid": 2,
            "distance": 10,
            "lookat": [10, 2, 0.5],
            "elevation": 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f"[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}")

            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[
                :, :, : self.observation_dim
            ]

            frame = []
            for states in states_l:
                img = self.composite(
                    None,
                    states,
                    dim=(1024, 256),
                    partial=True,
                    qvel=True,
                    render_kwargs=render_kwargs,
                )
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)


def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f"[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, "
            f"but got state of size {state.size}"
        )
        state = state[: qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])


def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack(
        [rollout_from_state(env, state, actions) for actions in actions_l]
    )
    return rollouts


def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions) + 1):
        # if terminated early, pad with zeros
        observations.append(np.zeros(obs.size))
    return np.stack(observations)
