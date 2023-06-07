# Necessary libraries
import argparse
import json
import random
import numpy as np
import os
# Ray
import ray
# PyTorch
import torch
from ray import air
# Ray Tune
from ray import tune
# Ray RLlib
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

# Custom imports
# environment
from env.lbf import RllibLBF, RllibLBFComm
# models
from models.ippo import IPPO
from models.gappo import GAPPO


def il_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # each agent has its own policy under its ID
    return agent_id


def shared_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # each agent has its own policy under its ID
    return "shared_policy"


def get_config(algo: str, exp_config, seed, debug: bool):
    """
    Generate a PPO config
    """
    # get default PPO configuration
    algo_config = PPOConfig()
    # set framework to PyTorch
    algo_config.framework("torch")
    # set resources for faster training
    algo_config.resources(num_cpus_per_worker=1)
    # if seed is provided use for reproducibility
    if seed:
        algo_config.debugging(seed=seed)

    # ----------- #
    # Environment #
    # ----------- #
    # retrieve the environment config from the experiment config
    env_config = exp_config['env']
    # restrict the level of the players to 1 to have agents with all the same skill set
    env_config['max_player_level'] = 1
    # disable coop only mode where only food of maximum level spawns
    env_config['force_coop'] = False
    # set provided seed
    env_config['seed'] = seed
    # convert the provided environment field size
    env_config['field_size'] = (exp_config['env']['field_size'], exp_config['env']['field_size'])
    # configure the environment
    algo_config.environment(env="lbf", env_config=env_config)

    # Create test environments to extract observation space
    test_il_env = RllibLBF(env_config)
    test_comm_env = RllibLBFComm(env_config)

    # -------- #
    # Policies #
    # -------- #
    # create a test environment to extract observation and action space

    if algo == "IPPO":
        # # create independent policies to compute actions for the agents (IPPO)
        il_policies = {}
        # Independent learning, set up one policy per agent using the ID of the agent
        for idx in range(exp_config["env"]["players"]):
            il_policies[f"agent_{idx}"] = PolicySpec(
                policy_class=None,  # use default policy
                observation_space=test_il_env.observation_space,
                action_space=test_il_env.action_space)
        # IPPO
        algo_config.multi_agent(policies=il_policies, policy_mapping_fn=il_policy_mapping_fn, count_steps_by="env_steps")

    if algo == "GAPPO":
        shared_policy = {
            "shared_policy": PolicySpec(
                policy_class=None,
                observation_space=test_comm_env.observation_space,
                action_space=test_comm_env.action_space
            )
        }
        algo_config.multi_agent(policies=shared_policy, policy_mapping_fn=shared_policy_mapping_fn,
                                count_steps_by="env_steps")

    # custom models
    if algo == "IPPO":
        algo_config.training(model={"custom_model": "IPPO",
                                    "custom_model_config": {
                                        "n_agents": exp_config["env"]["players"],
                                        "food": exp_config["env"]["food"],
                                        "input_features": test_il_env.observation_space['obs'].shape[0],
                                        "hidden_dim": 128
                                    }
    })
    if algo == "GAPPO":
        algo_config.training(model={"custom_model": "GAPPO",
                                    "custom_model_config": {
                                        "n_agents": exp_config["env"]["players"],
                                        "food": exp_config["env"]["food"],
                                        "input_features": test_il_env.observation_space['obs'].shape[0],
                                        "hidden_dim": 128
                                    }})

    return algo_config


if __name__ == "__main__":
    # --------- #
    # Arguments #
    # --------- #
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--algo", type=str, default="IPPO", help="Algorithm to train, either IPPO or GAPPO")
    parser.add_argument("--config", type=str, default="", help="Configuration file for experiment")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--num_cpus", type=int, default=2, help="Seed for reproducibility")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="When debug flag provided only runs one training iteration")
    args = parser.parse_args()

    # ------- #
    # Seeding #
    # ------- #
    # use seed to allow reproducibility of experiments
    if args.seed:
        print(f"Using seed: {args.seed}")
        # set Python random seed
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

    # ------------- #
    # Configuration #
    # ------------- #
    # read in provided configuration file
    exp_config_file = open(args.config)
    # convert json file into Python dict
    exp_config = json.load(exp_config_file)
    # get the name of the current experiment which will be used as a directory to hold the runs of the experiment
    exp_name = exp_config["experiment_name"]

    # ----------- #
    # Environment #
    # ----------- #
    # register the level-based foraging environment
    if args.algo == "IPPO":
        register_env("lbf", lambda c: RllibLBF(c))
    elif args.algo == "GAPPO":
        register_env("lbf", lambda c: RllibLBFComm(c))

    # ----- #
    # Model #
    # ----- #
    ModelCatalog.register_custom_model("GAPPO", GAPPO)
    ModelCatalog.register_custom_model("IPPO", IPPO)

    # -------f- #
    # Training #
    # -------- #
    # get PPO configuration file
    config = get_config(args.algo, exp_config, args.seed, args.debug)

    # set amount of workers based on available CPUs
    config.rollouts(batch_mode="complete_episodes",
                    num_rollout_workers=1 if args.debug else args.num_cpus-1,
                    enable_connectors=False)

    if args.debug:
        algo = config.build()
        algo.train()
    else:
        # initialize ray
        print(f"[GAPPO_SSD]: Initializing Ray")
        ray.init(num_cpus=args.num_cpus, include_dashboard=False)
        print(f"[GAPPO_SSD]: Ray initialized with {args.num_cpus} cores")

        # stopping conditions for tuning
        stop = {"episodes_total": exp_config["n_episodes"]}
        # train
        results = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=air.RunConfig(stop=stop,
                                     verbose=1,
                                     name="IPPO_LBF" if args.algo == "IPPO" else f"GAPPO_LBF",
                                     checkpoint_config=air.CheckpointConfig(
                                         num_to_keep=1,
                                         checkpoint_frequency=1),
                                     local_dir=f"./exp_results/{exp_name}"),
        ).fit()
        assert results.num_errors == 0
