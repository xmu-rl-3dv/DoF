import os

import einops
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import base64
    import io

    from IPython import display as ipythondisplay
    from IPython.display import HTML
except:
    print("[ utils/colab ] Warning: not importing colab dependencies")

from .arrays import to_np, to_torch
from .serialization import mkdir
from .video import save_video


def run_diffusion(
    model, dataset, obs, n_samples=1, device="cuda:0", **diffusion_kwargs
):
    
    obs = dataset.normalizer.normalize(obs, "observations")

    
    
    obs = obs[None].repeat(n_samples, axis=0)

    
    conditions = {0: to_torch(obs, device=device)}

    samples, diffusion = model.conditional_sample(
        conditions, return_diffusion=True, verbose=False, **diffusion_kwargs
    )

    
    diffusion = to_np(diffusion)

    
    
    normed_observations = diffusion[:, :, :, dataset.action_dim :]

    
    observations = dataset.normalizer.unnormalize(normed_observations, "observations")

    
    observations = einops.rearrange(
        observations, "batch steps horizon dim -> steps batch horizon dim"
    )

    return observations


def show_diffusion(
    renderer,
    observations,
    n_repeat=100,
    substep=1,
    filename="diffusion.mp4",
    savebase="/content/videos",
):
    """
    observations : [ n_diffusion_steps x batch_size x horizon x observation_dim ]
    """
    mkdir(savebase)
    savepath = os.path.join(savebase, filename)

    subsampled = observations[::substep]

    images = []
    for t in tqdm(range(len(subsampled))):
        observation = subsampled[t]

        img = renderer.composite(None, observation)
        images.append(img)
    images = np.stack(images, axis=0)

    
    images = np.concatenate([images, images[-1:].repeat(n_repeat, axis=0)], axis=0)

    save_video(savepath, images)
    show_video(savepath)


def show_sample(
    renderer, observations, filename="sample.mp4", savebase="/content/videos"
):
    """
    observations : [ batch_size x horizon x observation_dim ]
    """

    mkdir(savebase)
    savepath = os.path.join(savebase, filename)

    images = []
    for rollout in observations:
        
        img = renderer._renders(rollout, partial=True)
        images.append(img)

    
    images = np.concatenate(images, axis=2)

    save_video(savepath, images)
    show_video(savepath, height=200)


def show_samples(renderer, observations_l, figsize=12):
    """
    observations_l : [ [ n_diffusion_steps x batch_size x horizon x observation_dim ], ... ]
    """

    images = []
    for observations in observations_l:
        path = observations[-1]
        img = renderer.composite(None, path)
        images.append(img)
    images = np.concatenate(images, axis=0)

    plt.imshow(images)
    plt.axis("off")
    plt.gcf().set_size_inches(figsize, figsize)


def show_video(path, height=400):
    video = io.open(path, "r+b").read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(
        HTML(
            data="""<video alt="test" autoplay 
              loop controls style="height: {0}px;">
              <source src="data:video/mp4;base64,{1}" type="video/mp4" />
           </video>""".format(
                height, encoded.decode("ascii")
            )
        )
    )
