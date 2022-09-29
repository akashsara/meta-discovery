# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_new_open_ai_gym_wrapper.py

import json
import os
import sys
import time

sys.path.append("./")

import numpy as np
import torch
import training_utils as utils
from agents.max_damage_agent import MaxDamagePlayer
from agents.simple_agent import SimpleRLPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from models import simple_models
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from rl.agents.ppo import PPOAgent

if __name__ == "__main__":
    # Config - Versioning
    training_opponent = "random"  # random, max, smart
    experiment_name = f"Simple_PPO_Base_v1"
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
        "memory_size": STEPS_PER_ROLLOUT,
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
    training_agent = PlayerConfiguration(hash_name + "_P1", None)
    test_agent = PlayerConfiguration(hash_name + "_Test", None)
    rand_player = PlayerConfiguration(hash_name + "_Rand", None)
    max_player = PlayerConfiguration(hash_name + "_Max", None)
    smax_player = PlayerConfiguration(hash_name + "_SMax", None)

    # Create Output Path
    output_dir = os.path.join(model_dir, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup opponents
    random_agent = RandomPlayer(
        battle_format="gen8randombattle",
        player_configuration=rand_player,
        server_configuration=server_config,
        start_timer_on_battle_start=True
    )
    max_damage_agent = MaxDamagePlayer(
        battle_format="gen8randombattle",
        player_configuration=max_player,
        server_configuration=server_config,
        start_timer_on_battle_start=True
    )
    smart_max_damage_agent = SimpleHeuristicsPlayer(
        battle_format="gen8randombattle",
        player_configuration=smax_player,
        server_configuration=server_config,
        start_timer_on_battle_start=True
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
    env_player = SimpleRLPlayer(
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent,
        server_configuration=server_config,
        opponent=training_opponent,
        start_challenging=False,
        start_timer_on_battle_start=True
    )
    # Setup independent player for testing
    test_player = SimpleRLPlayer(
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=test_agent,
        server_configuration=server_config,
        opponent=None,
        start_challenging=False,
        start_timer_on_battle_start=True
    )

    # Grab some values from the environment to setup our model
    state_size = 10  # Hard-coded for the simple model
    n_actions = env_player.action_space.n
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

    evaluation_results = {}
    if NB_VALIDATION_EPISODES > 0:
        evaluation_results = utils.poke_env_validate_model(
            test_player,
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
        # Train Model
        env_player.start_challenging()
        ppo.fit(env_player, VALIDATE_EVERY, do_training=True)
        # Shutdown training agent
        env_player.close(purge=False)

        # Evaluate Model
        if NB_VALIDATION_EPISODES > 0 and i + 1 != num_epochs:
            # Save model
            ppo.save(output_dir, reset_trackers=True, create_plots=False)
            # Validation
            evaluation_results = utils.poke_env_validate_model(
                test_player,
                ppo,
                NB_VALIDATION_EPISODES,
                random_agent,
                max_damage_agent,
                smart_max_damage_agent,
                f"validation_{i+1}",
                evaluation_results,
            )

    # Save final model
    ppo.save(output_dir, reset_trackers=True, create_plots=False)

    # Evaluation
    if NB_EVALUATION_EPISODES > 0:
        evaluation_results = utils.poke_env_validate_model(
            test_player,
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

    utils.load_trackers_to_ppo_model(output_dir, ppo)
    ppo.plot_and_save_metrics(
        output_dir, is_cumulative=True, reset_trackers=True, create_plots=True
    )
    end_time = time.time()
    print(f"Running Time: {end_time - start_time}")
