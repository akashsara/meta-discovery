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
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration

import utils
from agents.simple_agent import SimpleRLPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from sb3_contrib import MaskablePPO
from tqdm import tqdm
from stable_baselines3.common.logger import configure

def model_training(environment, model, **kwargs):
    model.learn(**kwargs)
    environment.complete_current_battle()


def model_evaluation(environment, model, num_episodes):
    # Reset battle statistics
    environment.reset_battles()
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
    average_reward = np.mean(all_rewards)
    episodic_average_reward = np.mean(episode_rewards)
    print(
        f"Evaluation: {environment.n_won_battles} victories out of {num_episodes} episodes. Average Reward: {average_reward:.4f}. Average Episode Reward: {episodic_average_reward:.4f}"
    )
    return average_reward, episodic_average_reward


if __name__ == "__main__":
    # Config - Versioning
    training_opponent = "random"  # random, max, smart
    experiment_name = f"SB3_Simple_PPO_Base_v2"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    NB_TRAINING_STEPS = 1000000  # Total training steps
    VALIDATE_EVERY = 250000  # Run intermediate evaluation every N steps
    NB_VALIDATION_EPISODES = 100  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 1000  # Final Evaluation

    # Config - Model Hyperparameters
    model_config = {"seed": RANDOM_SEED}

    # Set random seed
    np.random.seed(RANDOM_SEED)
    _ = torch.manual_seed(RANDOM_SEED)

    # Setup agent usernames for connecting to local showdown
    # This lets us train multiple agents while connecting to the same server
    training_agent = PlayerConfiguration(hash_name + "_P1", None)
    rand_player = PlayerConfiguration(hash_name + "_Rand", None)
    max_player = PlayerConfiguration(hash_name + "_Max", None)
    smax_player = PlayerConfiguration(hash_name + "_SMax", None)

    # Create Output Path
    output_dir = os.path.join(model_dir, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup player
    env_player = SimpleRLPlayer(
        battle_format="gen8randombattle",
        player_configuration=training_agent,
        log_level=30,
    )

    # Setup opponents
    random_agent = RandomPlayer(
        battle_format="gen8randombattle", player_configuration=rand_player
    )
    max_damage_agent = MaxDamagePlayer(
        battle_format="gen8randombattle", player_configuration=max_player
    )
    smart_max_damage_agent = SmartMaxDamagePlayer(
        battle_format="gen8randombattle", player_configuration=smax_player
    )
    if training_opponent == "random":
        training_opponent = random_agent
    elif training_opponent == "max":
        training_opponent = max_damage_agent
    elif training_opponent == "smart":
        training_opponent = smart_max_damage_agent
    else:
        raise ValueError("Unknown training opponent.")

    # Defining our DQN
    ppo = MaskablePPO("MlpPolicy", env_player, verbose=1, **model_config)

    # Setup Logging
    logger = configure(os.path.join(output_dir, "logs"), ["stdout", "json"])
    ppo.set_logger(logger)

    # Setup evaluation dict
    evaluation_results = {}

    # Initial Evaluation
    evaluation_results = utils.poke_env_validate_model(
        env_player,
        model_evaluation,
        ppo,
        NB_VALIDATION_EPISODES,
        random_agent,
        max_damage_agent,
        smart_max_damage_agent,
        f"initial",
        evaluation_results,
    )

    num_epochs = max(NB_TRAINING_STEPS // VALIDATE_EVERY, 1)
    for i in range(num_epochs):
        # Train Model for VALIDATE_EVERY_STEPS
        env_player.play_against(
            env_algorithm=model_training,
            opponent=training_opponent,
            env_algorithm_kwargs={"model": ppo, "total_timesteps": VALIDATE_EVERY},
        )

        # Save Model
        ppo.save(os.path.join(output_dir, f"model_{i+1}"))

        # Evaluate Model
        # Works only if NB_VALIDATION_EPISODES is set
        # And this isn't the last "epoch" [Since we do a full eval after this]
        if NB_VALIDATION_EPISODES > 0 and i + 1 != num_epochs:
            evaluation_results = utils.poke_env_validate_model(
                env_player,
                model_evaluation,
                ppo,
                NB_VALIDATION_EPISODES,
                random_agent,
                max_damage_agent,
                smart_max_damage_agent,
                f"validation_{i+1}",
                evaluation_results,
            )

    # Save final model
    ppo.save(os.path.join(output_dir, f"model_final"))

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        evaluation_results = utils.poke_env_validate_model(
            env_player,
            model_evaluation,
            ppo,
            NB_EVALUATION_EPISODES,
            random_agent,
            max_damage_agent,
            smart_max_damage_agent,
            f"final",
            evaluation_results,
        )

    with open(os.path.join(output_dir, "results.json"), "w") as fp:
        json.dump(evaluation_results, fp)

    # TODO: Plot Graphs
