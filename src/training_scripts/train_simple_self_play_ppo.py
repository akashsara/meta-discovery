# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py

import json
import os
import sys
import time

sys.path.append("./")

import numpy as np
import torch
import training_utils as utils
import battle_handler
from agents.max_damage_agent import MaxDamagePlayer
from agents.simple_agent import SimpleRLPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from models import simple_models
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from rl.agents.ppo import PPOAgent


if __name__ == "__main__":
    # Config - Versioning
    experiment_name = f"Simple_PPO_SelfPlay_v1"
    server_port = 8000
    hash_name = str(hash(experiment_name))[2:12]
    expt_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"Experiment: {experiment_name}\t Time: {expt_time}")
    start_time = time.time()

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    STEPS_PER_ROLLOUT = 2048  # Steps gathered before training (train interval)
    VALIDATE_EVERY = 51200  # Run intermediate evaluation every N steps
    NB_TRAINING_STEPS = 102400  # Total training steps
    NB_VALIDATION_EPISODES = 100  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 1000  # Final Evaluation

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 32,
        "num_training_epochs": 10,
        "gamma": 0.99,  # Discount Factor
        "gae_lambda": 0.95,  # GAE Parameter
        "clip_param": 0.2,  # Surrogate Clipping Parameter
        "value_clip_param": 0.2,  # Value Function Clipping Parameter
        "c1": 0.5,  # Loss constant 1
        "c2": 0.002,  # Loss constant 2
        "normalize_advantages": False,
        "use_action_mask": True,
        "memory_size": STEPS_PER_ROLLOUT * 2,  # Since selfplay
    }

    # Config = Model Setup
    MODEL = simple_models.SimpleActorCriticModel
    MODEL_KWARGS = {}

    # Config - Optimizer Setup
    OPTIMIZER = torch.optim.Adam
    OPTIMIZER_KWARGS = {"lr": 3e-4, "eps": 1e-5}

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
    state_size = 10  # Hard-coded for the simple model
    n_actions = player1.action_space.n
    MODEL_KWARGS["n_actions"] = n_actions

    # Defining our DQN
    ppo = PPOAgent(
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
        optimizer=OPTIMIZER,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        steps_per_rollout=STEPS_PER_ROLLOUT,
        state_size=state_size,
        n_actions=n_actions,
        **training_config,
    )

    # Train Model
    battle_handler.run_selfplay(
        rl_model=ppo,
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
    utils.load_trackers_to_ppo_model(output_dir, ppo)
    ppo.plot_and_save_metrics(
        output_dir, is_cumulative=True, reset_trackers=True, create_plots=True
    )

    end_time = time.time()
    print(f"Running Time: {end_time - start_time}")
