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
from stable_baselines3 import PPO
from tqdm import tqdm
from stable_baselines3.common.evaluation import evaluate_policy

def test_model(model, environment, num_episodes):
    all_rewards = []
    episode_rewards = []
    for i in tqdm(range(num_episodes)):
        state = environment.reset()
        done = False
        episode_reward = 0
        while not done:
            action = model.predict(state, deterministic=True)
            state, reward, done, _ = environment.step(action[0])
            all_rewards.append(reward)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.mean(all_rewards), np.mean(episode_rewards)


if __name__ == "__main__":
    # Config - Versioning
    # experiment_name = f"SB3TestLanderPPO"
    experiment_name = f"SB3TestCartpolePPO"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 128,
        "log_interval": 1000,
        "gamma": 0.99,  # Discount Factor
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
    OPTIMIZER_KWARGS = {"lr": 3e-4}

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
    ppo = PPO("MlpPolicy", env, verbose=1)

    mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=NB_VALIDATION_EPISODES, deterministic=True)
    print(f"INITIAL REWARD: {mean_reward}+/-{std_reward}")

    last_validated = 0
    num_epochs = max(NB_TRAINING_STEPS // VALIDATE_EVERY, 1)
    for i in range(num_epochs):
        # Train Model
        ppo.learn(total_timesteps=VALIDATE_EVERY)

        # Save Model
        ppo.save(os.path.join(output_dir, f"model_{i+1}"))

        # Evaluate Model
        # Works only if NB_VALIDATION_EPISODES is set
        # And this isn't the last "epoch" [Since we do a full eval after this]
        if (
            NB_VALIDATION_EPISODES > 0
            and i+1 != num_epochs
        ):
            mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=NB_VALIDATION_EPISODES, deterministic=True)
            print(f"VALIDATION {i+1} REWARD: {mean_reward}+/-{std_reward}")

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=NB_VALIDATION_EPISODES, deterministic=True)
        print(f"FINAL REWARD: {mean_reward}+/-{std_reward}")