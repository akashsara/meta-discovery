# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
"""
used for training a simple RL agent.
So small state vs random/max_damage/smart max_damage
"""
import os
import numpy as np

import torch
import torch.nn as nn

from agents.dqn_agent import SimpleRLPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration

from rl.agents.dqn import DQNAgent
from rl.policy import ExponentialDecayEpsilonGreedyPolicy, LinearDecayEpsilonGreedyPolicy
from rl.memory import SequentialMemory

import models

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
    NB_TRAINING_STEPS = 50000
    NB_EVALUATION_EPISODES = 100

    MODEL = models.SimpleModel
    MODEL_KWARGS = {}
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
    experiment_name = f"AANew_Simple_DQN_Base_v1"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

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
    env_player = SimpleRLPlayer(battle_format="gen8randombattle", player_configuration=training_agent)
    
    # Setup opponents
    random_agent = RandomPlayer(battle_format="gen8randombattle", player_configuration=rand_player)
    max_damage_agent = MaxDamagePlayer(battle_format="gen8randombattle", player_configuration=max_player)
    smart_max_damage_agent = SmartMaxDamagePlayer(battle_format="gen8randombattle", player_configuration=smax_player)
    if training_opponent == "random":
        training_opponent = random_agent
    elif training_opponent == "max":
        training_opponent = max_damage_agent
    elif training_opponent == "smart":
        training_opponent = smart_max_damage_agent
    else:
        raise ValueError("Unknown training opponent.")

    # Output dimension
    n_actions = len(env_player.action_space)
    MODEL_KWARGS["n_actions"] = n_actions

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
