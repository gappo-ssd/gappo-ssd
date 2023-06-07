import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import TensorType

from torch_geometric.data import Batch
from torch_geometric.data.data import Data
from torch_geometric import utils as geo_utils
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.utils import unbatch, unbatch_edge_index, to_edge_index

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)

WEIGHTS_VIS_DICT = dict()

def create_geo_batch(m_obs, m_adj):
    """
      m_obs: [batch_size, n_agents, embedding_dim] batch of observations of agents
      m_adj: [batch_size, n_agents, n_agents] batch of adjacency matrices
    """
    # extract batch_size from m_obs:
    batch_size = m_obs.shape[0]
    # list of Data objects that will be converted into a Pytorch Geometric batch
    data_lst = []
    for i in range(batch_size):
        # extract observation of sample
        sample_obs = m_obs[i]
        # extract adjacency of sample
        sample_adj = m_adj[i]
        # convert adjacency matrix to sparse COO format
        sample_adj, _ = geo_utils.dense_to_sparse(sample_adj)
        # create Pytorch Geometric data object
        d = Data(x=sample_obs, edge_index=sample_adj)
        data_lst.append(d)
    # create a batch from the list of Data objects
    batch = Batch.from_data_list(data_lst)
    return batch


class MLPEncoder(nn.Module):
    """
        Encoder: creates embedding of node observations
    """
    def __init__(self, input_features, embedding_dim=128):
        super(MLPEncoder, self).__init__()
        self.mlp_encoder = nn.Linear(in_features=input_features, out_features=embedding_dim)

    def forward(self, x):
        """
            x: Tensor([batch_size, observation_shape]) observation of an agent

            returns: x: Tensor([batch_size, embedding_dim]) embedded observations of agent
        """
        # encode the received observation with MLP
        x = F.relu(self.mlp_encoder(x))
        # return embedded observation of agent
        return x


class GATLayer(nn.Module):
    """
        Graph Attention convolution mechanism: Creates latent features by combining agent's observations with
        observations of neighbors
    """
    def __init__(self, embedding_dim=128, heads=8):
        super(GATLayer, self).__init__()
        # Initialize GAT-layer
        self.gat_layer = GATv2Conv(embedding_dim, embedding_dim, heads=heads, concat=False)

    def forward(self, x, edge_index):
        """
            x: Tensor([n_agents, embedding_dim]): each agent is seen as a node in the graph with it's embedding as the
                node feature
            edge_index: Tensor(): the topology of the 'x' graph. The nodes of communicating agents are connected with
                each other

            returns: latent_features, att_weights: computed latent features and attention weights
        """
        # create latent features and attention weights
        latent_features, att_weights = self.gat_layer(x, edge_index, return_attention_weights=True)
        return latent_features, att_weights


class ActionLayer(nn.Module):
    """
        Action Layer: computes actions based on created latent features
    """
    def __init__(self, embedding_dim=128, num_actions=6):
        super(ActionLayer, self).__init__()
        # linear layer computes logits based on computed latent features
        self.fc = nn.Linear(in_features=embedding_dim*2, out_features=num_actions)

    def forward(self, i1, i2):
        x = torch.cat([i1, i2], dim=1)
        # compute logits based on latent features
        logits = self.fc(x)
        # return computed logits
        return logits


class GAPPO(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # - Extract model arguments #
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
        # encoder takes in local observations and returns encoded observations
        self.encoder = MLPEncoder(input_features=self.input_features, embedding_dim=self.hidden_dim)
        # GAT layer enables communication among agents
        self.gat = GATLayer(embedding_dim=self.hidden_dim, heads=2)
        # Q-Network computes final actions based on latent features
        self.action_layer = ActionLayer(embedding_dim=self.hidden_dim)
        # placeholder for values
        self.Vs = None
        # value function
        self.value_proc = lambda i1, i2: torch.cat([i1, i2], dim=1)
        self.value_branch = nn.Linear(self.hidden_dim*2, 1)

        self.save_weights = 'weights_dict' in kwargs

    def forward(self, input_dict, state, seq_lens):
        # -- Observations -- #
        # Extract the global observations [batch_size, n_agents, input_features]
        m_observations = input_dict['obs']['global_obs']
        # Extract the adjacency matrices [batch_size, n_agents, n_agents]
        m_adjacencies = input_dict['obs']['adj']

        # find size of batch
        batch_size = m_observations.shape[0]
        # agent_ids contains the ID of the agent of which we are getting the observations for each sample, as we are
        # dealing with global observations we use the 'agent_id' to extract the correct row when computing actions by
        # first converting them into one-hot encodings so we can later use batch multiplication to extract rows
        encoded_agent_ids = F.one_hot(input_dict['obs']['agent_id'].long(), num_classes=self.n_agents).float()

        #- 1. Encode observations using MLP encoder -#
        encoded_features = self.encoder(m_observations)

        #- 2. Enable agent communication with GAT-layer -#
        # convert RLlib batch into a Pytorch Geometric batch
        geo_batch = create_geo_batch(encoded_features, m_adjacencies)
        # compute the latent features using the GAT-layer
        geo_rel1, w_rel1 = self.gat(geo_batch.x, geo_batch.edge_index)
        # convert the Pytorch Geometric batch back to a normal batch
        unbatched_rel1 = torch.stack(unbatch(geo_rel1, geo_batch.batch))

        #- 3. Compute action probabilities based on the latent features using Q-Network -#
        # use the one-hot encoded agent_ids to extract the correct row as we are only computing probabilities for 1
        # agent instead of all agents
        obs_i = torch.bmm(encoded_agent_ids, encoded_features).squeeze(dim=1)
        r1_i = torch.bmm(encoded_agent_ids, unbatched_rel1).squeeze(dim=1)
        # compute action probabilities
        logits = self.action_layer(obs_i, r1_i).view(batch_size, self.num_outputs)
        # compute values using value function
        self.Vs = self.value_branch(self.value_proc(obs_i, r1_i)).view(batch_size, 1)

        # when visualizing we can visualize the attention weights to see which agent contributed more during
        # communication
        if self.save_weights:
            w_edge_index, w_weights = w_rel1
            coo_edge_index = w_edge_index
            n_edges = coo_edge_index[0].shape[0]
            global WEIGHTS_VIS_DICT
            if 'weights' in WEIGHTS_VIS_DICT:
                agent_id = input_dict['obs']['agent_id'][0][0]
                for i in range(n_edges):
                    # source node
                    source_n = coo_edge_index[1][i]
                    if source_n != agent_id:
                        continue
                    # target node
                    target_n = coo_edge_index[0][i]
                    # weight
                    weight_n = w_weights[i]
                    # tot_weights += torch.mean(weight_n)
                    WEIGHTS_VIS_DICT['weights'][source_n][target_n] = torch.mean(weight_n)

        # return computed action probabilities and empty states (only used for RNN)
        return logits, state

    def value_function(self) -> TensorType:
        if self.Vs is None:
            raise Exception("Run forward() first")
        # Return the output of the value function
        return torch.reshape(self.Vs, [-1])
