# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py

import asyncio
import json
import os
import sys
import time
from threading import Thread

sys.path.append("./")

import numpy as np
import torch
import torch.nn as nn
import training_utils as utils
from agents.max_damage_agent import MaxDamagePlayer
from agents.simple_agent import SimpleRLPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from models import simple_models
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import (
    ExponentialDecayEpsilonGreedyPolicy,
    LinearDecayEpsilonGreedyPolicy,
)


async def battle_handler(player1, player2, num_challenges):
    await asyncio.gather(
        player1.agent.accept_challenges(player2.username, num_challenges),
        player2.agent.send_challenges(player1.username, num_challenges),
    )


def training_function(player, opponent, model, model_kwargs):
    # Fit (train) model as necessary.
    model.fit(player, **model_kwargs)
    player.done_training = True
    # Play out the remaining battles so both fit() functions complete
    # We use 99 to give the agent an invalid option so it's forced
    # to take a random legal action
    while not opponent.done_training:
        _, _, done, _ = player.step(99)
        if done and not opponent.done_training:
            _ = player.reset()
            done = False

    # Forfeit any ongoing battles
    while player.current_battle and not player.current_battle.finished:
        _ = player.step(-1)


if __name__ == "__main__":
    # Config - Versioning
    training_opponent = "random"  # random, max, smart
    experiment_name = f"New_Simple_DQN_SelfPlay_v1"
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
    VALIDATE_EVERY = 20000  # Run intermediate evaluation every N steps
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
    MODEL = simple_models.SimpleModel
    MODEL_KWARGS = {}

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
        iterations = max([int(file.split(".")[0].split("_")[-1]) for file in os.listdir(output_dir) if "model" in file])
        training_config["load_dict_path"] = os.path.join(output_dir, f"model_{iterations}.pt")

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
    player1 = SimpleRLPlayer(
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent1,
        server_configuration=server_config,
        opponent=None,
        start_challenging=False,
        start_timer_on_battle_start=True,
    )
    player2 = SimpleRLPlayer(
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent2,
        server_configuration=server_config,
        opponent=None,
        start_challenging=False,
        start_timer_on_battle_start=True,
    )
    # Setup independent player for testing
    test_player = SimpleRLPlayer(
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=test_agent,
        server_configuration=server_config,
        opponent=None,
        start_challenging=False,
        start_timer_on_battle_start=True,
    )

    # Grab some values from the environment to setup our model
    n_actions = player1.action_space.n
    MODEL_KWARGS["n_actions"] = n_actions

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
        # Setup arguments to pass to the training function
        p1_env_kwargs = {
            "num_training_steps": VALIDATE_EVERY,
            "do_training": True,
        }
        p2_env_kwargs = {
            "num_training_steps": VALIDATE_EVERY,
            "do_training": False,
        }

        # Self-Play bits
        player1.done_training = False
        player2.done_training = False
        # 1. Get event loop
        loop = asyncio.get_event_loop()
        # Make Two Threads; one per player and run model.fit()
        t1 = Thread(
            target=lambda: training_function(player1, player2, dqn, p1_env_kwargs)
        )
        t1.start()

        t2 = Thread(
            target=lambda: training_function(player2, player1, dqn, p2_env_kwargs)
        )
        t2.start()
        # On the network side, keep sending & accepting battles
        while not player1.done_training or not player2.done_training:
            loop.run_until_complete(battle_handler(player1, player2, 1))
        # Wait for thread completion
        t1.join()
        t2.join()

        player1.close(purge=False)
        player2.close(purge=False)

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
