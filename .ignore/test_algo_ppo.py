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
from rl.agents.sanity_check.ppo_gym_compatible import PPOAgent

if __name__ == "__main__":
    # Config - Versioning
    # experiment_name = f"TestLanderPPO"
    experiment_name = f"TestCartpolePPO"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 32,
        "num_training_epochs": 10,
        "gamma": 0.99,  # Discount Factor
        "gae_lambda": 0.95,  # GAE Parameter
        "clip_param": 0.2,  # Surrogate Clipping Parameter
        "value_clip_param": 0.2,  # Value Function Clipping Parameter
        "c1": 0.5,  # Loss constant 1
        "c2": 0.002,  # Loss constant 2
        "normalize_advantages": False,
        "use_action_mask": False,
    }

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    STEPS_PER_ROLLOUT = 2048  # Steps gathered before training (train interval)
    VALIDATE_EVERY = 204800  # Run intermediate evaluation every N steps
    NB_TRAINING_STEPS = 1024000  # Total training steps
    NB_VALIDATION_EPISODES = 100  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 1000  # Final Evaluation

    # Config = Model Setup
    MODEL = simple_models.SimpleActorCriticModel
    MODEL_KWARGS = {}

    # Config - Optimizer Setup
    OPTIMIZER = torch.optim.Adam
    OPTIMIZER_KWARGS = {"lr": 3e-4, "eps": 1e-5}

    # Set random seed
    np.random.seed(RANDOM_SEED)
    _ = torch.manual_seed(RANDOM_SEED)

    # Create Output Path
    output_dir = os.path.join(model_dir, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup player
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")

    # Grab some values from the environment to setup our model
    print(env.action_space, env.observation_space)
    MODEL_KWARGS["n_actions"] = env.action_space.n
    # MODEL_KWARGS["n_obs"] = 8
    MODEL_KWARGS["n_obs"] = 4

    # Defining our DQN
    ppo = PPOAgent(
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
        optimizer=OPTIMIZER,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        steps_per_rollout=STEPS_PER_ROLLOUT,
        state_size=MODEL_KWARGS["n_obs"],
        n_actions=MODEL_KWARGS["n_actions"],
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

    num_epochs = max(NB_TRAINING_STEPS // VALIDATE_EVERY, 1)
    for i in range(num_epochs):
        # Train Model
        ppo.fit(env, total_steps=VALIDATE_EVERY)

        # Save Model
        ppo.save(output_dir, reset_trackers=True, create_plots=False)

        # Evaluate Model
        # Works only if NB_VALIDATION_EPISODES is set
        # And this isn't the last "epoch" [Since we do a full eval after this]
        if NB_VALIDATION_EPISODES > 0 and i + 1 != num_epochs:
            evaluation_results = utils.gym_env_validate_model(
                evaluation_results,
                env,
                ppo,
                NB_VALIDATION_EPISODES,
                f"validation_{i+1}",
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
