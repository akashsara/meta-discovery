# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py
"""
used for training a "full state" RL agent.
This is essentially a large state representation that uses nearly every aspect
of the poke-env API
"""
import json
import os
import sys
import time

sys.path.append("./")

import numpy as np
import torch
import torch.nn as nn
import training_utils as utils
from agents.full_state_agent import FullStatePlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
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
    experiment_name = f"New_FullState_DQN_Base_v1"
    server_port = 8000
    hash_name = str(hash(experiment_name))[2:12]
    expt_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"Experiment: {experiment_name}\t Time: {expt_time}")
    start_time = time.time()

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    NB_TRAINING_STEPS = 100000  # Total training steps
    VALIDATE_EVERY = 50000  # Run intermediate evaluation every N steps
    NB_VALIDATION_EPISODES = 10  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 100  # Final Evaluation

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 32,
        "gamma": 0.95,  # Discount Factor
        "tau": 0.001,  # AKA Target Model Update
        "train_interval": 1,
        "log_interval": 1000,
        "warmup_steps": 1000,
    }

    # Config = Model Setup
    MODEL = full_state_models.BattleModel
    MODEL_KWARGS = {
        "pokemon_embedding_dim": 32,
        "team_embedding_dim": 64,
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
    training_agent = PlayerConfiguration(hash_name + "_P1", None)
    test_agent = PlayerConfiguration(hash_name + "_Test", None)
    rand_player = PlayerConfiguration(hash_name + "_Rand", None)
    max_player = PlayerConfiguration(hash_name + "_Max", None)
    smax_player = PlayerConfiguration(hash_name + "_SMax", None)

    # Create Output Path
    output_dir = os.path.join(model_dir, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
    smart_max_damage_agent = SmartMaxDamagePlayer(
        battle_format="gen8randombattle",
        player_configuration=smax_player,
        server_configuration=server_config,
        start_timer_on_battle_start=True,
    )
    if training_opponent == "random":
        training_opponent = random_agent
    elif training_opponent == "max":
        training_opponent = max_damage_agent
    elif training_opponent == "smart":
        training_opponent = smart_max_damage_agent
    else:
        raise ValueError("Unknown training opponent.")

    # Setup player
    env_player = FullStatePlayer(
        config,
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent,
        server_configuration=server_config,
        opponent=training_opponent,
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
    MODEL_KWARGS["n_actions"] = env_player.action_space.n
    MODEL_KWARGS["state_length_dict"] = env_player.get_state_lengths()
    MODEL_KWARGS["max_values_dict"] = env_player.lookup["max_values"]

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

    evaluation_results = {}
    if NB_VALIDATION_EPISODES > 0:
        evaluation_results = utils.poke_env_validate_model(
            test_player,
            dqn,
            NB_VALIDATION_EPISODES,
            random_agent,
            max_damage_agent,
            smart_max_damage_agent,
            f"initial",
            evaluation_results,
        )
    num_epochs = max(NB_TRAINING_STEPS // VALIDATE_EVERY, 1)

    for i in range(num_epochs):
        # Train Model
        env_player.start_challenging()
        dqn.fit(env_player, VALIDATE_EVERY, do_training=True)
        # Shutdown training agent
        env_player.close(purge=False)

        # Evaluate Model
        if NB_VALIDATION_EPISODES > 0 and i + 1 != num_epochs:
            # Save model
            dqn.save(output_dir, reset_trackers=True, create_plots=False)
            # Validation
            evaluation_results = utils.poke_env_validate_model(
                test_player,
                dqn,
                NB_VALIDATION_EPISODES,
                random_agent,
                max_damage_agent,
                smart_max_damage_agent,
                f"validation_{i+1}",
                evaluation_results,
            )

    # Save final model
    dqn.save(output_dir, reset_trackers=True, create_plots=False)

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        evaluation_results = utils.poke_env_validate_model(
            test_player,
            dqn,
            NB_EVALUATION_EPISODES,
            random_agent,
            max_damage_agent,
            smart_max_damage_agent,
            f"final",
            evaluation_results,
        )

    with open(os.path.join(output_dir, "results.json"), "w") as fp:
        json.dump(evaluation_results, fp)

    utils.load_trackers_to_dqn_model(output_dir, dqn)
    dqn.plot_and_save_metrics(
        output_dir, is_cumulative=True, reset_trackers=True, create_plots=True
    )
    end_time = time.time()
    print(f"Running Time: {end_time - start_time}")

    dqn.plot_and_save_metrics(
        output_dir, is_cumulative=True, reset_trackers=True, create_plots=True
    )
