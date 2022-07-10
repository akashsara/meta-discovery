# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
"""
used for training a simple RL agent.
So small state vs random/max_damage/smart max_damage
"""
import json
import os

import gym
import numpy as np
import torch
import torch.nn as nn

import utils
from models import simple_models
from rl.agents.ppo_gym_compatible import PPOAgent
from rl.memory import PPOMemory

if __name__ == "__main__":
    # Config - Versioning
    experiment_name = f"TestLanderPPO"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 64,
        "log_interval": 1000,
        "num_training_epochs": 10,
        "gamma": 0.99,  # Discount Factor
        "lambda_": 0.95,  # GAE Parameter
        "clip_param": 0.2,  # Surrogate Clipping Parameter
        "c1": 0.5,  # Loss constant 1
        "c2": 0.002,  # Loss constant 2
        "normalize_advantages": False,
    }

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    NB_TRAINING_STEPS = 1000000
    STEPS_PER_ROLLOUT = 2500  # Steps to gather before running PPO (train interval)
    VALIDATE_EVERY = 200000  # Run intermediate evaluation every N steps
    NB_VALIDATION_EPISODES = 100  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 1000  # Final Evaluation

    # Config = Model Setup
    MODEL = simple_models.SimpleActorCriticModel
    MODEL_KWARGS = {}
    memory_config = {"batch_size": training_config["batch_size"]}

    # Config - Optimizer Setup
    OPTIMIZER = torch.optim.Adam
    OPTIMIZER_KWARGS = {"lr": 3e-4}

    # Set random seed
    np.random.seed(RANDOM_SEED)
    _ = torch.manual_seed(RANDOM_SEED)

    # Create Output Path
    output_dir = os.path.join(model_dir, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup player
    env = gym.make("LunarLander-v2")

    # Grab some values from the environment to setup our model
    print(env.action_space, env.observation_space)
    MODEL_KWARGS["n_actions"] = env.action_space.n
    MODEL_KWARGS["n_obs"] = 8

    # Setup memory
    memory = PPOMemory(**memory_config)
    # Defining our DQN
    ppo = PPOAgent(
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
        optimizer=OPTIMIZER,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        memory=memory,
        **training_config,
    )

    evaluation_results = {}

    evaluation_results = utils.gym_env_validate_model(
        evaluation_results,
        env,
        ppo,
        NB_VALIDATION_EPISODES,
        f"initial",
    )

    last_validated = 0
    num_epochs = max(VALIDATE_EVERY // STEPS_PER_ROLLOUT, 1)
    while ppo.iterations < NB_TRAINING_STEPS:
        # Train Model
        ppo.fit(env, steps_per_rollout=STEPS_PER_ROLLOUT, num_epochs=num_epochs)

        # Save Model
        ppo.save(output_dir, reset_trackers=True, create_plots=False)

        # Evaluate Model
        # Works only if NB_VALIDATION_EPISODES is set
        # And this isn't the last "epoch" [Since we do a full eval after this]
        if (
            NB_VALIDATION_EPISODES > 0
            and (ppo.iterations - last_validated) >= VALIDATE_EVERY
        ):
            evaluation_results = utils.gym_env_validate_model(
                evaluation_results,
                env,
                ppo,
                NB_VALIDATION_EPISODES,
                f"validation_{ppo.iterations}",
            )

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        evaluation_results = utils.gym_env_validate_model(
            evaluation_results, env, ppo, NB_EVALUATION_EPISODES, "final"
        )

    with open(os.path.join(output_dir, "results.json"), "w") as fp:
        json.dump(evaluation_results, fp)

    utils.load_trackers_to_ppo_model(output_dir, ppo)
    ppo.plot_and_save_metrics(
        output_dir, is_cumulative=True, reset_trackers=True, create_plots=True
    )
