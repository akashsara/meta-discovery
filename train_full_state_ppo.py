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

from models import full_state_models
from agents.dqn_full_state_agent import FullStatePlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from rl.agents.ppo import PPOAgent
from rl.memory import PPOMemory


def model_training(player, model, **kwargs):
    model.fit(player, **kwargs)
    player.complete_current_battle()


def model_evaluation(player, model, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    average_reward, episodic_average_reward = model.test(player, num_episodes=nb_episodes)

    print(
        f"Evaluation: {player.n_won_battles} victories out of {nb_episodes} episodes. Average Reward: {average_reward}. Average Episode Reward: {episodic_average_reward}"
    )


if __name__ == "__main__":
    # Config - Versioning
    training_opponent = "random"  # random, max, smart
    experiment_name = f"FullState_PPO_Base_v1"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 32,
        "log_interval": 1000,
        "gamma": 0.99,  # Discount Factor
        "lambda_": 0.95,  # GAE Parameter
        "clip_param": 0.2,  # Surrogate Clipping Parameter
        "c1": 0.5,  # Loss constant 1
        "c2": 0.001,  # Loss constant 2
    }

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    NB_TRAINING_STEPS = 100000  # Total training steps
    STEPS_PER_EPOCH = 5000  # Steps to gather before running PPO (train interval)
    VALIDATE_EVERY = 50000  # Run intermediate evaluation every N steps
    NB_VALIDATION_EPISODES = 100  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 1000  # Final Evaluation

    # Config = Model Setup
    MODEL = full_state_models.ActorCriticBattleModel
    MODEL_KWARGS = {
        "pokemon_embedding_dim": 128,
        "team_embedding_dim": 128,
    }
    memory_config = {"batch_size": training_config["batch_size"]}

    # Config - Optimizer Setup
    OPTIMIZER = torch.optim.Adam
    OPTIMIZER_KWARGS = {"lr": 1e-4}

    # Config - Model Save Directory/Config Directory + json info files
    config = {
        "create": True,
        "pokemon_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json",
        "moves_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/moves.json",
        "items_json": "https://raw.githubusercontent.com/akashsara/showdown-data/main/dist/data/items.json",
        "lookup_filename": "player_lookup_dicts.joblib",
    }

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
    config["lookup_filename"] = os.path.join(output_dir, config["lookup_filename"])

    # Create Player
    env_player = FullStatePlayer(
        config,
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent,
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

    # Grab some values from the environment to setup our model
    MODEL_KWARGS["n_actions"] = len(env_player.action_space)
    MODEL_KWARGS["state_length_dict"] = env_player.get_state_lengths()
    MODEL_KWARGS["max_values_dict"] = env_player.lookup["max_values"]

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
    last_validated = 0
    num_epochs = max(VALIDATE_EVERY // STEPS_PER_EPOCH, 1)
    while ppo.iterations < NB_TRAINING_STEPS:
        # Train Model
        # We train for VALIDATE_EVERY steps
        # That is, VALIDATE_EVERY//STEPS_PER_EPOCH epochs
        # Each of STEPS_PER_EPOCH steps
        env_player.play_against(
            env_algorithm=model_training,
            opponent=training_opponent,
            env_algorithm_kwargs={
                "model": ppo,
                "steps_per_epoch": STEPS_PER_EPOCH,
                "num_epochs": num_epochs,
            },
        )

        # Evaluate Model
        # Works only if NB_VALIDATION_EPISODES is set
        # if STEPS_PER_EPOCH >= VALIDATE_EVERY,
        # we evaluate every STEPS_PER_EPOCH
        # Else
        # we evaluate every ceil(VALIDATE_EVERY / STEPS_PER_EPOCH) steps
        if (
            NB_VALIDATION_EPISODES > 0
            and (ppo.iterations - last_validated) >= VALIDATE_EVERY
        ):
            # Save model
            ppo.save(output_dir, reset_trackers=True, create_plots=False)
            # Validation
            last_validated = ppo.iterations
            evaluation_results[f"validation_{ppo.iterations}"] = {
                "n_battles": NB_VALIDATION_EPISODES,
            }

            print("Results against random player:")
            env_player.play_against(
                env_algorithm=model_evaluation,
                opponent=random_agent,
                env_algorithm_kwargs={
                    "model": ppo,
                    "nb_episodes": NB_VALIDATION_EPISODES,
                },
            )
            evaluation_results[f"validation_{ppo.iterations}"][
                "vs_random"
            ] = env_player.n_won_battles

            print("\nResults against max player:")
            env_player.play_against(
                env_algorithm=model_evaluation,
                opponent=max_damage_agent,
                env_algorithm_kwargs={
                    "model": ppo,
                    "nb_episodes": NB_VALIDATION_EPISODES,
                },
            )
            evaluation_results[f"validation_{ppo.iterations}"][
                "vs_max"
            ] = env_player.n_won_battles

            print("\nResults against smart max player:")
            env_player.play_against(
                env_algorithm=model_evaluation,
                opponent=smart_max_damage_agent,
                env_algorithm_kwargs={
                    "model": ppo,
                    "nb_episodes": NB_VALIDATION_EPISODES,
                },
            )
            evaluation_results[f"validation_{ppo.iterations}"][
                "vs_smax"
            ] = env_player.n_won_battles

    # Save final model
    ppo.save(output_dir, reset_trackers=True, create_plots=False)

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        evaluation_results["final"] = {
            "n_battles": NB_EVALUATION_EPISODES,
        }

        print("Results against random player:")
        env_player.play_against(
            env_algorithm=model_evaluation,
            opponent=random_agent,
            env_algorithm_kwargs={"model": ppo, "nb_episodes": NB_EVALUATION_EPISODES},
        )
        evaluation_results["final"]["vs_random"] = env_player.n_won_battles

        print("\nResults against max player:")
        env_player.play_against(
            env_algorithm=model_evaluation,
            opponent=max_damage_agent,
            env_algorithm_kwargs={"model": ppo, "nb_episodes": NB_EVALUATION_EPISODES},
        )
        evaluation_results["final"]["vs_max"] = env_player.n_won_battles

        print("\nResults against smart max player:")
        env_player.play_against(
            env_algorithm=model_evaluation,
            opponent=smart_max_damage_agent,
            env_algorithm_kwargs={"model": ppo, "nb_episodes": NB_EVALUATION_EPISODES},
        )
        evaluation_results["final"]["vs_smax"] = env_player.n_won_battles

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