import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class IPPO(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        #- Extract model arguments #
        # amount of agents
        self.n_agents = kwargs['n_agents']
        # amount of food
        self.n_food = sum(list(map(lambda f: f[0], kwargs['food'].values())))
        # amount of input features
        self.input_features = kwargs['input_features']
        # size of node embeddings
        self.hidden_dim = kwargs['hidden_dim']
        # amount of actions
        self.num_outputs = num_outputs

        #- Model Layers -#
        # fully connected layer
        self.fc1 = nn.Linear(self.input_features, self.hidden_dim)
        # final layer to produce actions
        self.final = nn.Linear(self.hidden_dim, self.num_outputs)
        # placeholder for values
        self.Vs = None
        # value function
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_dict, state, seq_lens):
        # Extract the agent's local observations
        x = input_dict['obs']['obs']
        # Process observation with fully connected layer
        x = F.relu(self.fc1(x))
        # Compute logits
        logits = self.final(x)
        # Compute values using value function
        self.Vs = self.value(x)
        # return computed action probabilities and empty states (only used for RNN)
        return logits, state

    def value_function(self):
        if self.Vs is None:
            raise Exception("Run forward() first")
        return torch.reshape(self.Vs, [-1])
