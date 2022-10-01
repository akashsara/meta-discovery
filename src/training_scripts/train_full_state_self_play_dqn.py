# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py

import json
import os
import sys
import time

sys.path.append("./")

import numpy as np
import torch
import torch.nn as nn
import training_utils as utils
import battle_handler
from agents.full_state_agent import FullStatePlayer
from agents.max_damage_agent import MaxDamagePlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from models import full_state_models
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import (
    ExponentialDecayEpsilonGreedyPolicy,
    LinearDecayEpsilonGreedyPolicy,
)

if __name__ == "__main__":
    # Config - Versioning
    training_opponent = "random"  # random, max, smart
    experiment_name = f"New_FullState_DQN_SelfPlay_v1"
    server_port = 8000
    hash_name = str(hash(experiment_name))[2:12]
    expt_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"Experiment: {experiment_name}\t Time: {expt_time}")
    start_time = time.time()

    # Choose whether to use the flattened model or the full model
    use_flattened_model = False

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    NB_TRAINING_STEPS = 100000  # Total training steps
    VALIDATE_EVERY = 50000  # Run intermediate evaluation every N steps
    NB_VALIDATION_EPISODES = 100  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 1000  # Final Evaluation

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 32,
        "gamma": 0.95,  # Discount Factor
        "tau": 1,  # AKA Target Model Update
        "train_interval": 1,
        "log_interval": 1000,
        "warmup_steps": 1000,
    }

    # Config = Model Setup
    if use_flattened_model:
        MODEL = full_state_models.FlattenedBattleModel
        MODEL_KWARGS = {}
    else:
        MODEL = full_state_models.BattleModel
        MODEL_KWARGS = {
            "pokemon_embedding_dim": 128,
            "team_embedding_dim": 128,
        }

    # Config - Memory Setup
    MEMORY = SequentialMemory
    MEMORY_KWARGS = {"capacity": 10000}

    # Config - Policy Setup
    POLICY = LinearDecayEpsilonGreedyPolicy
    # POLICY = ExponentialDecayEpsilonGreedyPolicy
    POLICY_KWARGS = {
        "max_epsilon": 0.95,
        "min_epsilon": 0.05,
        # "epsilon_decay": 1000,
        "max_steps": NB_TRAINING_STEPS,
    }

    # Config - Optimizer Setup
    OPTIMIZER = torch.optim.Adam
    OPTIMIZER_KWARGS = {"lr": 0.00025}

    # Config - Loss Setup
    LOSS = nn.SmoothL1Loss
    LOSS_KWARGS = {
        "beta": 0.01,
    }

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

    # Setup server configuration
    # Maintain servers on different ports to avoid Compute Canada errors
    server_config = utils.generate_server_configuration(server_port)

    # Setup agent usernames for connecting to local showdown
    # This lets us train multiple agents while connecting to the same server
    training_agent1 = PlayerConfiguration(hash_name + "_P1", None)
    training_agent2 = PlayerConfiguration(hash_name + "_P2", None)
    test_agent = PlayerConfiguration(hash_name + "_Test", None)
    rand_player = PlayerConfiguration(hash_name + "_Rand", None)
    max_player = PlayerConfiguration(hash_name + "_Max", None)
    smax_player = PlayerConfiguration(hash_name + "_SMax", None)

    # Create Output Path
    output_dir = os.path.join(model_dir, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        evaluation_results = {}
    else:
        with open(os.path.join(output_dir, "results.json"), "r") as fp:
            evaluation_results = json.load(fp)
        iterations = max(
            [
                int(file.split(".")[0].split("_")[-1])
                for file in os.listdir(output_dir)
                if "model" in file
            ]
        )
        training_config["load_dict_path"] = os.path.join(
            output_dir, f"model_{iterations}.pt"
        )
        config["create"] = False
    config["lookup_filename"] = os.path.join(output_dir, config["lookup_filename"])

    # Setup opponents
    random_agent = RandomPlayer(
        battle_format="gen8randombattle",
        player_configuration=rand_player,
        server_configuration=server_config,
        start_timer_on_battle_start=True,
    )
    max_damage_agent = MaxDamagePlayer(
        battle_format="gen8randombattle",
        player_configuration=max_player,
        server_configuration=server_config,
        start_timer_on_battle_start=True,
    )
    smart_max_damage_agent = SimpleHeuristicsPlayer(
        battle_format="gen8randombattle",
        player_configuration=smax_player,
        server_configuration=server_config,
        start_timer_on_battle_start=True,
    )

    # Setup player
    player1 = FullStatePlayer(
        config,
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent1,
        server_configuration=server_config,
        opponent=None,
        start_challenging=False,
        start_timer_on_battle_start=True,
    )
    config["create"] = False
    player2 = FullStatePlayer(
        config,
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent2,
        server_configuration=server_config,
        opponent=None,
        start_challenging=False,
        start_timer_on_battle_start=True,
    )
    # Setup independent player for testing
    test_player = FullStatePlayer(
        config,
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=test_agent,
        server_configuration=server_config,
        opponent=None,
        start_challenging=False,
        start_timer_on_battle_start=True,
    )

    # Grab some values from the environment to setup our model
    MODEL_KWARGS["n_actions"] = player1.action_space.n
    if use_flattened_model:
        sample = player1.create_empty_state_vector()
        sample = player1.state_to_machine_readable_state(sample)
        MODEL_KWARGS["n_obs"] = sample.shape[0]
    else:
        MODEL_KWARGS["state_length_dict"] = player1.get_state_lengths()
        MODEL_KWARGS["max_values_dict"] = player1.lookup["max_values"]

    # Setup memory
    memory = MEMORY(**MEMORY_KWARGS)

    # Simple Epsilon Greedy Policy
    policy = POLICY(**POLICY_KWARGS)

    # Setup loss function
    loss = LOSS(**LOSS_KWARGS)

    # Defining our DQN
    dqn = DQNAgent(
        policy=policy,
        memory=memory,
        loss_function=loss,
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
        optimizer=OPTIMIZER,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        **training_config,
    )

    # Train Model
    battle_handler.run_selfplay(
        rl_model=dqn,
        player1=player1,
        player2=player2,
        test_player=test_player,
        random_agent=random_agent,
        max_damage_agent=max_damage_agent,
        smart_max_damage_agent=smart_max_damage_agent,
        nb_training_steps=NB_TRAINING_STEPS,
        validate_every=VALIDATE_EVERY,
        nb_validation_episodes=NB_VALIDATION_EPISODES,
        nb_evaluation_episodes=NB_EVALUATION_EPISODES,
        evaluation_results=evaluation_results,
        output_dir=output_dir,
    )

    # Load all statistics & make plots
    utils.load_trackers_to_dqn_model(output_dir, dqn)
    dqn.plot_and_save_metrics(
        output_dir, is_cumulative=True, reset_trackers=True, create_plots=True
    )

    end_time = time.time()
    print(f"Running Time: {end_time - start_time}")
