from lbforaging.foraging.environment import ForagingEnv
import numpy as np
import copy

n_agents = 2
food = {'1': [2, 1]}
n_food = sum(list(map(lambda f: f[0], food.values())))


def prepare_obs(m_obs, n_agents, n_food):
    # [n_agents, batch_size, observation_shape]
    # initialize adjacency matrix
    adj = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        agents_layer = m_obs[i][0:25]
        neighbors = agents_layer[agents_layer != -1.]
        for j in neighbors:
            if int(j) != -1:
                adj[int(i)][int(j)] = 1

    return m_obs, adj


def range_communication(m_obs, n_agents, positions, com_range=4):
    adj = np.zeros((n_agents, n_agents))
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
    adj = np.zeros((n_agents, n_agents))
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
        # sort the distances based on Manhatten distance
        distances.sort(key=lambda x: x[1])
        # retrieve the 'n' closest agents
        for n in range(K):
            # retrieve the ID of one of the K closest agents
            other_id = distances[n][0]
            # create edge between agents
            adj[i][other_id] = 1
    return m_obs, adj


# configuration of the environment
env_config = {
    "players": n_agents,
    "max_player_level": 1,
    "field_size": (10, 10),
    "food": food,
    "sight": [2, 2, 2, 2, 2, 2, 2, 2],
    "max_episode_steps": 50,
    "force_coop": False,
    "food_respawn_interval": 0,
    "seed": 1,
    "energy_penalty": -0.1,
    "grid_observation": False,
    "food_respawn": False,
}

env = ForagingEnv(**env_config)
obs, positions = env.reset()


# Actions:
#  0: none
#  1: up
#  2: down
#  3: left
#  4: right
#  5: load

def get_action():
    user_input = input("Choose action: ")
    received_actions = user_input.split(",")
    received_actions = list(map(lambda x: int(x), received_actions))
    return received_actions


game_done = False
while not game_done:
    env.render()
    # for i in range(3):
    #     i_obs = obs[i]
    #     print(f"agent_{i}")
    #     print(f"- i_obs: {i_obs}")
    #     print(f"- Own: [{i_obs[n_food*3]}, {i_obs[(n_food*3)+1]}]")
    #     print(f"- Food:")
    #     for z in range(n_food):
    #         print(f"  * Sees food at [{i_obs[z*3]}, {i_obs[(z*3)+1]}] of level {i_obs[(z*3)+2]}")
    #     print(f"- Agents:")
    #     for z in range(1, n_agents):
    #         offset = (n_food+z)*3
    #         print(f"  * Sees agent {i_obs[offset+2]} at [{i_obs[offset]}, {i_obs[offset+1]}]")

    actions = get_action()
    obs, positions, rewards, dones, infos = env.step(actions)
    for i in range(n_agents):
        print(f"agent_{i}: {obs[i]}")
    # m_obs, m_adj = range_communication(m_obs=obs, n_agents=n_agents, positions=positions, com_range=4)
    # m_obs, m_adj = knn_based_communication(m_obs=obs, n_agents=n_agents, positions=positions)

    game_done = dones[0]
    # for i in range(n_agents):
    #     agents_layer, food_layer, wall_layer = np.split(obs[i], 3)
    #     agents_layer = agents_layer.reshape((5, 5))
    #     food_layer = food_layer.reshape((5, 5))
    #     wall_layer = wall_layer.reshape((5, 5))
    #     print(f"[agent_{i}] agents: ({agents_layer.shape}):\n {agents_layer}")
    #     print(f"[agent_{i}] food: ({food_layer.shape}):\n {food_layer}")
    #     print(f"[agent_{i}] wall: ({wall_layer.shape}):\n {wall_layer}")

env.close()
