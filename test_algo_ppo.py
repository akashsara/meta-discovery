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
from rl.agents.ppo_gym_compatible import PPOAgent
from rl.memory import PPOMemory

if __name__ == "__main__":
    # Config - Versioning
    training_opponent = "max"  # random, max, smart
    experiment_name = f"TestTaxiPPO"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 32,
        "log_interval": 1000,
        "gamma": 0.9,  # Discount Factor
        "lambda_": 0.95,  # GAE Parameter
        "clip_param": 0.2,  # Surrogate Clipping Parameter
        "c1": 0.5,  # Loss constant 1
        "c2": 0.001,  # Loss constant 2
    }

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    NB_TRAINING_STEPS = 100000
    STEPS_PER_EPOCH = 5000  # Steps to gather before running PPO (train interval)
    VALIDATE_EVERY = 50000  # Run intermediate evaluation every N steps
    NB_VALIDATION_EPISODES = 100  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 1000  # Final Evaluation

    # Config = Model Setup
    MODEL = simple_models.SimpleActorCriticModel
    MODEL_KWARGS = {}
    memory_config = {"batch_size": training_config["batch_size"]}

    # Config - Optimizer Setup
    OPTIMIZER = torch.optim.Adam
    OPTIMIZER_KWARGS = {"lr": 0.00025}

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

    evaluation_results["initial"] = {
        "n_battles": NB_VALIDATION_EPISODES,
    }
    rewards = ppo.test(env, NB_VALIDATION_EPISODES)
    evaluation_results["initial"]["rewards"] = rewards

    print(f"INITIAL REWARD: {rewards}")

    last_validated = 0
    num_epochs = max(VALIDATE_EVERY // STEPS_PER_EPOCH, 1)
    while ppo.iterations < NB_TRAINING_STEPS:
        # Train Model
        ppo.fit(env, steps_per_epoch=STEPS_PER_EPOCH, num_epochs=num_epochs)

        # Save Model
        ppo.save(output_dir, reset_trackers=True, create_plots=False)

        # Evaluate Model
        # Works only if NB_VALIDATION_EPISODES is set
        # And this isn't the last "epoch" [Since we do a full eval after this]
        if (
            NB_VALIDATION_EPISODES > 0
            and (ppo.iterations - last_validated) >= VALIDATE_EVERY
        ):
            evaluation_results[f"validation_{ppo.iterations}"] = {
                "n_battles": NB_VALIDATION_EPISODES,
            }
            rewards = ppo.test(env, NB_VALIDATION_EPISODES)
            evaluation_results[f"validation_{ppo.iterations}"]["rewards"] = rewards
            print(f"VALIDATION {ppo.iterations} REWARD: {rewards}")

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        evaluation_results["final"] = {
            "n_battles": NB_EVALUATION_EPISODES,
        }
        rewards = ppo.test(env, NB_EVALUATION_EPISODES)
        evaluation_results["final"]["rewards"] = rewards
        print(f"FINAL REWARD: {rewards}")

    with open(os.path.join(output_dir, "results.json"), "w") as fp:
        json.dump(evaluation_results, fp)

    # Load back all the trackers to draw the final plots
    all_rewards = []
    all_episode_lengths = []
    all_actor_losses = []
    all_critic_losses = []
    all_entropy = []
    all_losses = []
    # Sort files by iteration for proper graphing
    files_to_read = sorted([int(file.split(".pt")[0].split("_")[1]) for file in os.listdir(output_dir) if "statistics_" in file])
    for file in files_to_read:
        x = torch.load(os.path.join(output_dir, f"statistics_{file}.pt"), map_location=ppo.device)
        all_rewards.append(x["reward"])
        all_episode_lengths.append(x["episode_lengths"])
        all_actor_losses.append(x["actor_loss"])
        all_critic_losses.append(x["critic_loss"])
        all_entropy.append(x["entropy"])
        all_losses.append(x["total_loss"])
    all_rewards = torch.cat(all_rewards).flatten().cpu().numpy()
    all_episode_lengths = torch.cat(all_episode_lengths).flatten().cpu().numpy()
    all_actor_losses = torch.cat(all_actor_losses).flatten().cpu().numpy()
    all_critic_losses = torch.cat(all_critic_losses).flatten().cpu().numpy()
    all_entropy = torch.cat(all_entropy).flatten().cpu().numpy()
    all_losses = torch.cat(all_losses).flatten().cpu().numpy()
    ppo.rewards = all_rewards
    ppo.episode_lengths = all_episode_lengths
    ppo.actor_losses = all_actor_losses
    ppo.critic_losses = all_critic_losses
    ppo.entropy = all_entropy
    ppo.total_losses = all_losses
    ppo.plot_and_save_metrics(
        output_dir, is_cumulative=True, reset_trackers=True, create_plots=True
    )