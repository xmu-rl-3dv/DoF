from .invmodel import JointInvModel, SharedInvModel, IndependentInvModel
import torch.nn as nn

class InvModelBuilder:
    def __init__(self, model_type: str, n_agents: int, observation_dim: int, hidden_dim: int, output_dim: int, discrete_action: bool = False, device=None):
        """
        Initialize builder

        :param model_type: The type of the model ('joint', 'shared', or 'independent').
        :param n_agents: The number of agents.
        :param observation_dim: The dimension of observations.
        :param hidden_dim: The dimension of hidden layers.
        :param output_dim: The dimension of the output.
        :param discrete_action: Whether to use discrete actions.
        """
        self.model = self._select_model(model_type, n_agents, observation_dim, hidden_dim, output_dim, discrete_action, device)

    def _select_model(self, model_type: str, n_agents: int, observation_dim: int, hidden_dim: int, output_dim: int, discrete_action: bool, device=None):
        """
        Select and initialize the appropriate model.

        :param model_type: The type of the model ('joint', 'shared', or 'independent').
        :param n_agents: The number of agents.
        :param observation_dim: The dimension of observations.
        :param hidden_dim: The dimension of hidden layers.
        :param output_dim: The dimension of the output.
        :param discrete_action: Whether to use discrete actions.
        :param device: The target device for the model (e.g., 'cpu' or 'cuda').
        :return: An instance of the initialized model.
        """
        if model_type == "joint":
            print("\nUSE JOINT INV\n")
            model = JointInvModel(n_agents, observation_dim, hidden_dim, output_dim)
        elif model_type == "shared":
            print("\nUSE SHARED INV\n")
            model = SharedInvModel(observation_dim, hidden_dim, output_dim)
        elif model_type == "independent":
            print("\nUSE INDEPENDENT INV\n")
            model = IndependentInvModel(n_agents, observation_dim, hidden_dim, output_dim, discrete_action)
        else:
            raise ValueError("Unknown model type: choose 'joint', 'shared', or 'independent'.")

        
        if device is not None:
            model = model.to(device)

        return model

    def forward(self, x):
        return self.model(x)
    
    def __call__(self, x):
        return self.forward(x)
