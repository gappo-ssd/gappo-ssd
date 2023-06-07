import numpy as np
import torch
from gym.spaces import Dict as GymDict, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from env.lbforaging.foraging.environment import ForagingEnv


# --------------------------#
# Communication Topologies #
# --------------------------#

def full_communication(m_obs, n_agents, positions):
    adj = torch.ones(n_agents, n_agents)
    return m_obs, adj


def range_communication(m_obs, n_agents, positions, com_range=2):
    adj = torch.zeros(n_agents, n_agents)
    for i in range(n_agents):
        for j in range(n_agents):
            y_dif = abs(positions[i][0] - positions[j][0])
            x_dif = abs(positions[i][1] - positions[j][1])
            if x_dif <= com_range and y_dif <= com_range:
                adj[i][j] = 1
    return m_obs, adj


def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def knn_based_communication(m_obs, n_agents, positions, K=2):
    adj = torch.zeros(n_agents, n_agents)
    for i, agent_pos in enumerate(positions):
        # add self-loop
        adj[i][i] = 1
        # list to keep track of the distances between agent_i and the other agents
        distances = []
        # compute and save distance for all other agents
        for j, other_pos in enumerate(positions):
            if i != j:
                dist = manhattan_dist(agent_pos, other_pos)
                distances.append((j, dist))
        # sort the distances based on Manhattan distance
        distances.sort(key=lambda x: x[1])
        # retrieve the 'n' closest agents
        for n in range(K):
            # retrieve the ID of one of the K closest agents
            other_id = distances[n][0]
            # create edge between agents
            adj[i][other_id] = 1
    return m_obs, adj


class RllibLBFComm(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True

        self.n_agents = env_config["players"]
        self.n_food = sum(list(map(lambda f: f[0], env_config["food"].values())))

        communication_conf = env_config.pop("communication")
        self.communication_topology = communication_conf["topology"]
        self.communication_args = communication_conf["communication_args"]
        if self.communication_topology == "range":
            self.communication_range = self.communication_args["range"]
        elif self.communication_topology == "knn":
            self.communication_K = self.communication_args["K"]

        self._spaces_in_preferred_format = True
        self.env = ForagingEnv(**env_config)

        self.action_space = self.env.action_space[0]
        # noinspection PyTypeCheckerres,PyTypeChecker
        self.observation_space = GymDict({
            "global_obs": Box(
                low=-100.0,
                high=100.0,
                shape=(env_config["players"], self.env.observation_space[0].shape[0],),
                dtype=self.env.observation_space[0].dtype),
            "adj": Box(
                low=-100.0,
                high=100.0,
                shape=(env_config["players"], env_config["players"]),
                dtype=int),
            "agent_id": Box(low=-100, high=100, shape=(1,), dtype=int)
        })
        self.num_agents = self.env.n_agents
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        self.env_config = env_config

    def reset(self, seed, options):
        original_obs, original_pos = self.env.reset()
        m_obs = None
        m_adj = None
        if self.communication_topology == "range":
            m_obs, m_adj = range_communication(np.array(original_obs), self.n_agents, original_pos,
                                               com_range=self.communication_range)
        if self.communication_topology == "knn":
            m_obs, m_adj = knn_based_communication(np.array(original_obs), self.n_agents, original_pos,
                                                   K=self.communication_K)
        obs = {}
        for x in range(self.num_agents):
            obs["agent_%d" % x] = {
                # "global_obs": np.array(original_obs).copy(),
                "global_obs": m_obs,
                "adj": m_adj,
                "agent_id": [int(x)]
            }
        return obs, {}

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, p, r, d, i = self.env.step(tuple(actions))
        m_obs = None
        m_adj = None
        if self.communication_topology == "range":
            m_obs, m_adj = range_communication(np.array(o), self.n_agents, p, com_range=self.communication_range)
        if self.communication_topology == "knn":
            m_obs, m_adj = knn_based_communication(np.array(o), self.n_agents, p, K=self.communication_K)
        rewards = {}
        obs = {}
        infos = {}
        done_flag = False
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            rewards[key] = r[pos]
            obs[key] = {
                # "global_obs": np.array(o).copy(),
                "global_obs": m_obs,
                "adj": m_adj,
                "agent_id": [int(key.split("_")[1])]
            }
            done_flag = d[pos] or done_flag
        dones = {"__all__": done_flag}
        truncateds = {"__all__": False}
        return obs, rewards, dones, truncateds, infos

    def render(self):
        # self.env.lbf_render()
        self.env.render()

    def close(self):
        self.env.close()


class RllibLBF(MultiAgentEnv):
    def __init__(self, env_config):
        env_config = env_config.copy()
        super().__init__()
        self._skip_env_checking = True

        self.n_agents = env_config["players"]
        self.n_food = sum(list(map(lambda f: f[0], env_config["food"].values())))

        communication_conf = env_config.pop("communication")
        self._spaces_in_preferred_format = True
        self.env = ForagingEnv(**env_config)

        self.action_space = self.env.action_space[0]
        # noinspection PyTypeCheckerres,PyTypeChecker
        self.observation_space = GymDict({
            "obs": Box(
                low=-100.0,
                high=100.0,
                shape=(self.env.observation_space[0].shape[0],),
                dtype=self.env.observation_space[0].dtype)
        })
        self.num_agents = self.env.n_agents
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        self.env_config = env_config

    def reset(self, seed, options):
        original_obs, player_pos = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            obs["agent_%d" % x] = {
                "obs": original_obs[x],
            }
        return obs, {}

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, p, r, d, i = self.env.step(tuple(actions))
        rewards = {}
        obs = {}
        infos = {}
        done_flag = False
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            rewards[key] = r[pos]
            obs[key] = {
                "obs": o[pos],
            }
            done_flag = d[pos] or done_flag
        dones = {"__all__": done_flag}
        truncateds = {"__all__": False}
        return obs, rewards, dones, truncateds, infos

    def render(self):
        # self.env.lbf_render()
        self.env.render()

    def close(self):
        self.env.close()
