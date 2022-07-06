# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
"""
used for training a simple RL agent.
So small state vs random/max_damage/smart max_damage
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import gym

from models import simple_models
from rl.agents.dqn_gym_compatible import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import (
    ExponentialDecayEpsilonGreedyPolicy,
    LinearDecayEpsilonGreedyPolicy,
)

if __name__ == "__main__":
    # Config - Versioning
    training_opponent = "max"  # random, max, smart
    experiment_name = f"TestTaxiDQN"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 32,
        "gamma": 0.9,
        "tau": 0.001,  # AKA Target Model Update
        "train_interval": 1,
        "log_interval": 1000,
        "warmup_steps": 1000,
    }

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    NB_TRAINING_STEPS = 100000
    VALIDATE_EVERY = 50000  # Run intermediate evaluation every N steps
    NB_VALIDATION_EPISODES = 100  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 1000  # Final Evaluation

    # Config = Model Setup
    MODEL = simple_models.SimpleModel
    MODEL_KWARGS = {}
    memory_config = {"capacity": 10000}

    # Config - Policy Setup
    # POLICY = LinearDecayEpsilonGreedyPolicy
    POLICY = ExponentialDecayEpsilonGreedyPolicy
    policy_config = {
        "max_epsilon": 0.95,
        "min_epsilon": 0.05,
        "epsilon_decay": NB_TRAINING_STEPS // 10,
        # "max_steps": NB_TRAINING_STEPS,
    }

    # Config - Optimizer Setup
    OPTIMIZER = torch.optim.Adam
    OPTIMIZER_KWARGS = {"lr": 0.00025}

    # Config - Loss Setup
    LOSS = nn.SmoothL1Loss
    LOSS_KWARGS = {
        "beta": 0.01,
    }

    # Set random seed
    np.random.seed(RANDOM_SEED)
    _ = torch.manual_seed(RANDOM_SEED)

    # Create Output Path
    output_dir = os.path.join(model_dir, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup player
    env = gym.make("Taxi-v3")

    # Grab some values from the environment to setup our model
    print(env.action_space, env.observation_space)
    MODEL_KWARGS["n_actions"] = env.action_space.n
    MODEL_KWARGS["n_obs"] = 1

    # Setup memory
    memory = SequentialMemory(**memory_config)

    # Simple Epsilon Greedy Policy
    policy = POLICY(**policy_config)

    # Defining our DQN
    dqn = DQNAgent(
        policy=policy,
        memory=memory,
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
        optimizer=OPTIMIZER,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        loss=LOSS,
        loss_kwargs=LOSS_KWARGS,
        **training_config,
    )

    evaluation_results = {}

    evaluation_results["initial"] = {
        "n_battles": NB_VALIDATION_EPISODES,
    }
    rewards = dqn.test(env, NB_VALIDATION_EPISODES)
    evaluation_results["initial"]["rewards"] = rewards

    print(f"INITIAL REWARD: {rewards}")

    epochs = max(NB_TRAINING_STEPS // VALIDATE_EVERY, 1)
    for i in range(epochs):
        # Train Model
        dqn.fit(env, num_training_steps=VALIDATE_EVERY)

        # Save Model
        dqn.save(output_dir, reset_trackers=True, create_plots=False)

        # Evaluate Model
        # Works only if NB_VALIDATION_EPISODES is set
        # And this isn't the last "epoch" [Since we do a full eval after this]
        if NB_VALIDATION_EPISODES > 0 and i + 1 != epochs:
            evaluation_results[f"validation_set_{i+1}"] = {
                "n_battles": NB_VALIDATION_EPISODES,
            }
            rewards = dqn.test(env, NB_VALIDATION_EPISODES)
            evaluation_results[f"validation_set_{i+1}"]["rewards"] = rewards
            print(f"VALIDATION {i+1} REWARD: {rewards}")

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        evaluation_results["final"] = {
            "n_battles": NB_EVALUATION_EPISODES,
        }
        rewards = dqn.test(env, NB_EVALUATION_EPISODES)
        evaluation_results["final"]["rewards"] = rewards
        print(f"FINAL REWARD: {rewards}")

    with open(os.path.join(output_dir, "results.json"), "w") as fp:
        json.dump(evaluation_results, fp)

    # Load back all the trackers to draw the final plots
    all_losses = []
    all_rewards = []
    all_battle_lengths = []
    for file in os.listdir(output_dir):
        if "statistics_" in file:
            x = torch.load(os.path.join(output_dir, file), map_location=dqn.device)
            all_losses.append(x["loss"])
            all_rewards.append(x["reward"])
            all_battle_lengths.append(x["battle_length"])
    all_losses = torch.cat(all_losses).flatten().cpu().numpy()
    all_rewards = torch.cat(all_rewards).flatten().cpu().numpy()
    all_battle_lengths = torch.cat(all_battle_lengths).flatten().cpu().numpy()
    dqn.losses = all_losses
    dqn.rewards = all_rewards
    dqn.battle_lengths = all_battle_lengths
    dqn.plot_and_save_metrics(
        output_dir, is_cumulative=True, reset_trackers=True, create_plots=True
    )