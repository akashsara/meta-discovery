# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py

import os
import numpy as np
from poke_env.player.random_player import RandomPlayer

from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from agents.dqn_full_state_agent import FullStatePlayer

from poke_env.player_configuration import PlayerConfiguration

from rl.agents.dqn import DQNAgent
from rl.policy import (
    ExponentialDecayEpsilonGreedyPolicy,
    LinearDecayEpsilonGreedyPolicy,
)
from rl.memory import SequentialMemory

import models
import torch
import torch.nn as nn

# This is the function that will be used to train the dqn
def model_training(player, model, nb_steps):
    model.fit(player, num_training_steps=nb_steps)
    player.complete_current_battle()


def model_evaluation(player, model, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    model.test(player, num_episodes=nb_episodes)

    print(
        "Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


if __name__ == "__main__":
    # Config - Hyperparameters
    RANDOM_SEED = 42
    NB_TRAINING_STEPS = 100000
    NB_EVALUATION_EPISODES = 100

    MODEL = models.BattleModel
    MODEL_KWARGS = {
        "pokemon_embedding_dim": 32,
        "team_embedding_dim": 64,
    }
    memory_config = {
        "capacity": 10000
    }

    OPTIMIZER = torch.optim.Adam
    OPTIMIZER_KWARGS = {"lr": 0.00025}

    POLICY = LinearDecayEpsilonGreedyPolicy
    # POLICY = ExponentialDecayEpsilonGreedyPolicy
    policy_config = {
        "max_epsilon": 0.95,
        "min_epsilon": 0.05,
        # "epsilon_decay": 1000,
        "max_steps": NB_TRAINING_STEPS
    }

    LOSS = nn.SmoothL1Loss
    LOSS_KWARGS = {
        "beta": 0.01,
    }

    training_config = {
        "batch_size": 32,
        "gamma": 0.9,
        "use_soft_update": False,
        "tau": 1000, # AKA Target Model Update
        "train_interval": 1,
        "log_interval": 1000,
        "warmup_steps": 1000
    }

    # Config - Versioning
    training_opponent = "max" # random, max, smart
    experiment_name = f"New_FullState_DQN_Base_v1"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Set random seed
    np.random.seed(RANDOM_SEED)
    _ = torch.manual_seed(RANDOM_SEED)

    # Config - Model Save Directory/Config Directory + json info files
    model_dir = "models"
    config = {
        "create": True,
        "pokemon_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json",
        "moves_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/moves.json",
        "items_json": "https://raw.githubusercontent.com/akashsara/showdown-data/main/dist/data/items.json",
        "lookup_filename": "player_lookup_dicts.joblib",
    }

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
    config["lookup_filename"] = os.path.join(
        output_dir, config["lookup_filename"]
    )

    # Create Player
    env_player = FullStatePlayer(config, battle_format="gen8randombattle", log_level=50)

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
        **training_config
    )

    # Train Model
    env_player.play_against(
        env_algorithm=model_training,
        opponent=training_opponent,
        env_algorithm_kwargs={"model": dqn, "nb_steps": NB_TRAINING_STEPS},
    )
    dqn.save(output_dir)

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        print("Results against random player:")
        env_player.play_against(
            env_algorithm=model_evaluation,
            opponent=random_agent,
            env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES}
        )

        print("\nResults against max player:")
        env_player.play_against(
            env_algorithm=model_evaluation,
            opponent=max_damage_agent,
            env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES}
        )

        print("\nResults against smart max player:")
        env_player.play_against(
            env_algorithm=model_evaluation,
            opponent=smart_max_damage_agent,
            env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES}
        )
