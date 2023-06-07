# Import necessary libraries
import argparse
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
# Ray RLlib
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
# Ray Tune
from ray.tune import ExperimentAnalysis, register_env

# Custom imports
from env.lbf import RllibLBF, RllibLBFComm
from models.gappo import GAPPO, WEIGHTS_VIS_DICT
from models.ippo import IPPO


# Visualizes the weights of the Graph-Attention mechanism
def create_importance_matrix(n_agents, weights, ax):
    # create heatmap
    ax.clear()
    plt.pause(0.0001)
    im = ax.imshow(weights, vmin=0., vmax=1.)

    ax.set_xticks(np.arange(n_agents), labels=[f"agent_{i}" for i in range(n_agents)], size=14)
    ax.set_yticks(np.arange(n_agents), labels=[f"agent_{i}" for i in range(n_agents)], size=14)

    # Loop over data dimensions and create text annotations.
    for i in range(n_agents):
        for j in range(n_agents):
            text = ax.text(j, i, "%.2f" % weights[i, j], size=16,
                           ha="center", va="center", color="w")

    ax.set_title("Importance of agent's communicated observation", fontsize=14)


if __name__ == "__main__":
    # --------- #
    # Arguments #
    # --------- #
    parser = argparse.ArgumentParser(description="Visualize an episode of a trained policy.")

    parser.add_argument("--experiment_state", type=str, required=True,
                        help="Json file of experiment state that will be visualized")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--weights", action='store_true', default=False,
                        help="When weights flag provided shows the weights of GraphPPO communication")
    args = parser.parse_args()
    # determine used algorithm based on path of experiment state
    algo = "IPPO" if "IPPO_LBF" in args.experiment_state else "GAPPO"

    # ------- #
    # Seeding #
    # ------- #

    random.seed(args.seed)
    # set Numpy seed
    np.random.seed(args.seed)
    # set Pytorch seed
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode(True)

    # ----------- #
    # Environment #
    # ----------- #
    # register the environment for RLlib
    if algo == "IPPO":
        register_env("lbf", lambda c: RllibLBF(c))
    else:
        register_env("lbf", lambda c: RllibLBFComm(c))

    # ----- #
    # Model #
    # ----- #
    # register the models
    ModelCatalog.register_custom_model("GAPPO", GAPPO)
    ModelCatalog.register_custom_model("IPPO", IPPO)
    # retrieve the experiment
    experiment = ExperimentAnalysis(
        f"{args.experiment_state}",
        default_metric="episode_reward_mean",
        default_mode="max"
    )
    # retrieve the name of the experiment, should be equal to args.experiment_state
    name_run = experiment.best_logdir
    # retrieve the config file of the experiment
    config = experiment.best_config
    # retrieve the last checkpoint (saved model) of the experiment
    checkpoint_path = experiment.best_checkpoint
    # only set one worker for the visualization
    config["num_workers"] = 1
    # save the weights of the Graph-Attention layer for the visualization
    config['model']['custom_model_config']['weights_dict'] = True
    # create a trainer based on the retrieved config file
    trainer = PPO(config=config)
    # restore the model based on the last checkpoint (saved model)
    trainer.restore(checkpoint_path=checkpoint_path)

    # ----------- #
    # Visualizing #
    # ----------- #
    # Create new environment to visualize
    config['env_config']['seed'] = args.seed
    rl_env = RllibLBF(config['env_config']) if algo == "IPPO" else RllibLBFComm(config['env_config'])
    obs, _ = rl_env.reset(1, {})
    # render the original state of the environment
    rl_env.render()
    time.sleep(0.5)
    # keep track of the reward in the episode
    episode_reward = 0
    # dictionary to keep track of the actions of all agents
    actions = dict()
    n_agents = config['env_config']['players']
    # config['env_config']['grid_size'] = 15
    n_food = sum(list(map(lambda f: f[0], config['env_config']['food'].values())))

    # Put matplotlib in interactive mode to visualize the weights throughout the episode
    ax = None
    if algo == "GAPPO":
        plt.ion()
        plt.show()
        fig, ax = plt.subplots()

    # episodes in the LBF environment last 'max_steps'
    for step in range(config['env_config']['max_episode_steps']):
        # # render the environment
        rl_env.render()
        time.sleep(0.5)

        # Visualize the weights
        if algo == "GAPPO":
            if step != 0:
                create_importance_matrix(config['env_config']['players'], WEIGHTS_VIS_DICT['weights'], ax)
            WEIGHTS_VIS_DICT.clear()
            WEIGHTS_VIS_DICT['weights'] = np.zeros((config['env_config']['players'], config['env_config']['players']))

        #-- Actions --#
        if algo == "GAPPO":
            for i in range(config['env_config']['players']):
                a = trainer.compute_single_action(
                    observation=obs[f"agent_{i}"],
                    prev_state={},
                    policy_id=f"shared_policy"
                )
                # save chosen action of each agent
                actions[f"agent_{i}"] = a

        if algo == "IPPO":
            for i in range(config['env_config']['players']):
                a = trainer.compute_single_action(
                    observation=obs[f"agent_{i}"],
                    prev_state={},
                    policy_id=f"agent_{i}"
                )
                # save chosen action of each agent
                actions[f"agent_{i}"] = a
        # execute all actions in the environment and update observations
        obs, rewards, dones, truncateds, infos = rl_env.step(actions)

        episode_reward += sum(list(rewards.values()))
        print(f"Step [{step}]")
        input()

    print(f"Final episode reward: {episode_reward}")
