# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py

import asyncio
import json
import os
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.utils import to_id_str

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
    average_reward = model.test(player, num_episodes=nb_episodes)

    print(
        f"Evaluation: {player.n_won_battles} victories out of {nb_episodes} episodes. Average Reward: {average_reward}"
    )


async def launch_battles(player, opponent):
    battles_coroutine = asyncio.gather(
        player.send_challenges(
            opponent=to_id_str(opponent.username),
            n_challenges=1,
            to_wait=opponent.logged_in,
        ),
        opponent.accept_challenges(opponent=to_id_str(player.username), n_challenges=1),
    )
    await battles_coroutine


def env_algorithm_wrapper(env_algorithm, player, kwargs):
    env_algorithm(player, **kwargs)

    player._start_new_battle = False
    while True:
        try:
            player.complete_current_battle()
            player.reset()
        except OSError:
            break


if __name__ == "__main__":
    # Config - Versioning
    experiment_name = f"FullState_PPO_SelfPlay_v1"
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
    training_agent1 = PlayerConfiguration(hash_name + "_P1", None)
    training_agent2 = PlayerConfiguration(hash_name + "_P2", None)
    rand_player = PlayerConfiguration(hash_name + "_Rand", None)
    max_player = PlayerConfiguration(hash_name + "_Max", None)
    smax_player = PlayerConfiguration(hash_name + "_SMax", None)

    # Create Output Path
    output_dir = os.path.join(model_dir, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup player
    player1 = FullStatePlayer(
        config,
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent1,
    )
    config["create"] = False
    player2 = FullStatePlayer(
        config,
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent2,
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

    # Grab some values from the environment to setup our model
    MODEL_KWARGS["n_actions"] = len(player1.action_space)
    MODEL_KWARGS["state_length_dict"] = player1.get_state_lengths()
    MODEL_KWARGS["max_values_dict"] = player1.lookup["max_values"]

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
        # Setup arguments to pass to the training function
        p1_env_kwargs = {
            "model": ppo,
            "steps_per_epoch": STEPS_PER_EPOCH // 2,
            "num_epochs": num_epochs,
            "do_training": True,
        }
        p2_env_kwargs = {
            "model": ppo,
            "steps_per_epoch": STEPS_PER_EPOCH // 2,
            "num_epochs": num_epochs,
            "do_training": False,
        }

        # Train Model
        # Make Two Threads And Play vs Each Other
        player1._start_new_battle = True
        player2._start_new_battle = True

        loop = asyncio.get_event_loop()

        t1 = Thread(
            target=lambda: env_algorithm_wrapper(model_training, player1, p1_env_kwargs)
        )
        t1.start()

        t2 = Thread(
            target=lambda: env_algorithm_wrapper(model_training, player2, p2_env_kwargs)
        )
        t2.start()

        while player1._start_new_battle:
            loop.run_until_complete(launch_battles(player1, player2))
        t1.join()
        t2.join()

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
            player1.play_against(
                env_algorithm=model_evaluation,
                opponent=random_agent,
                env_algorithm_kwargs={
                    "model": ppo,
                    "nb_episodes": NB_VALIDATION_EPISODES,
                },
            )
            evaluation_results[f"validation_{ppo.iterations}"][
                "vs_random"
            ] = player1.n_won_battles

            print("\nResults against max player:")
            player1.play_against(
                env_algorithm=model_evaluation,
                opponent=max_damage_agent,
                env_algorithm_kwargs={
                    "model": ppo,
                    "nb_episodes": NB_VALIDATION_EPISODES,
                },
            )
            evaluation_results[f"validation_{ppo.iterations}"][
                "vs_max"
            ] = player1.n_won_battles

            print("\nResults against smart max player:")
            player1.play_against(
                env_algorithm=model_evaluation,
                opponent=smart_max_damage_agent,
                env_algorithm_kwargs={
                    "model": ppo,
                    "nb_episodes": NB_VALIDATION_EPISODES,
                },
            )
            evaluation_results[f"validation_{ppo.iterations}"][
                "vs_smax"
            ] = player1.n_won_battles

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        evaluation_results["final"] = {
            "n_battles": NB_EVALUATION_EPISODES,
        }

        print("Results against random player:")
        player1.play_against(
            env_algorithm=model_evaluation,
            opponent=random_agent,
            env_algorithm_kwargs={"model": ppo, "nb_episodes": NB_EVALUATION_EPISODES},
        )
        evaluation_results["final"]["vs_random"] = player1.n_won_battles

        print("\nResults against max player:")
        player1.play_against(
            env_algorithm=model_evaluation,
            opponent=max_damage_agent,
            env_algorithm_kwargs={"model": ppo, "nb_episodes": NB_EVALUATION_EPISODES},
        )
        evaluation_results["final"]["vs_max"] = player1.n_won_battles

        print("\nResults against smart max player:")
        player1.play_against(
            env_algorithm=model_evaluation,
            opponent=smart_max_damage_agent,
            env_algorithm_kwargs={"model": ppo, "nb_episodes": NB_EVALUATION_EPISODES},
        )
        evaluation_results["final"]["vs_smax"] = player1.n_won_battles

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