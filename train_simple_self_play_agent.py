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

import models
from agents.dqn_agent import SimpleRLPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import (
    ExponentialDecayEpsilonGreedyPolicy,
    LinearDecayEpsilonGreedyPolicy,
)


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
    experiment_name = f"New_Simple_DQN_SelfPlay_v2"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Config - Model Hyperparameters
    training_config = {
        "batch_size": 32,
        "gamma": 0.9,
        "tau": 1000,  # AKA Target Model Update
        "train_interval": 1,
        "log_interval": 1000,
        "warmup_steps": 1000,
    }

    # Config - Training Hyperparameters
    RANDOM_SEED = 42
    NB_TRAINING_STEPS = 10000
    VALIDATE_EVERY = 5000  # Run intermediate evaluation every N steps
    NB_VALIDATION_EPISODES = 100  # Intermediate Evaluation
    NB_EVALUATION_EPISODES = 1000  # Final Evaluation

    # Config = Model Setup
    MODEL = models.SimpleModel
    MODEL_KWARGS = {}
    memory_config = {"capacity": 10000}

    # Config - Policy Setup
    POLICY = LinearDecayEpsilonGreedyPolicy
    # POLICY = ExponentialDecayEpsilonGreedyPolicy
    policy_config = {
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
    player1 = SimpleRLPlayer(
        battle_format="gen8randombattle",
        log_level=30,
        player_configuration=training_agent1,
    )
    player2 = SimpleRLPlayer(
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
    n_actions = len(player1.action_space)
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
        **training_config,
    )

    evaluation_results = {}
    epochs = NB_TRAINING_STEPS // VALIDATE_EVERY
    for i in range(epochs):
        # Setup arguments to pass to the training function
        p1_env_kwargs = {"model": dqn, "nb_steps": VALIDATE_EVERY}
        p2_env_kwargs = {"model": dqn, "nb_steps": VALIDATE_EVERY}

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

        # Save model
        dqn.save(output_dir)

        # Evaluate Model
        # Works only if NB_VALIDATION_EPISODES is set
        # And this isn't the last "epoch" [Since we do a full eval after this]
        if NB_VALIDATION_EPISODES > 0 and i + 1 != epochs:
            evaluation_results[f"validation_set_{i+1}"] = {
                "n_battles": NB_VALIDATION_EPISODES,
            }

            print("Results against random player:")
            player1.play_against(
                env_algorithm=model_evaluation,
                opponent=random_agent,
                env_algorithm_kwargs={
                    "model": dqn,
                    "nb_episodes": NB_VALIDATION_EPISODES,
                },
            )
            evaluation_results[f"validation_set_{i+1}"][
                "vs_random"
            ] = player1.n_won_battles

            print("\nResults against max player:")
            player1.play_against(
                env_algorithm=model_evaluation,
                opponent=max_damage_agent,
                env_algorithm_kwargs={
                    "model": dqn,
                    "nb_episodes": NB_VALIDATION_EPISODES,
                },
            )
            evaluation_results[f"validation_set_{i+1}"][
                "vs_max"
            ] = player1.n_won_battles

            print("\nResults against smart max player:")
            player1.play_against(
                env_algorithm=model_evaluation,
                opponent=smart_max_damage_agent,
                env_algorithm_kwargs={
                    "model": dqn,
                    "nb_episodes": NB_VALIDATION_EPISODES,
                },
            )
            evaluation_results[f"validation_set_{i+1}"][
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
            env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
        )
        evaluation_results["final"]["vs_random"] = player1.n_won_battles

        print("\nResults against max player:")
        player1.play_against(
            env_algorithm=model_evaluation,
            opponent=max_damage_agent,
            env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
        )
        evaluation_results["final"]["vs_max"] = player1.n_won_battles

        print("\nResults against smart max player:")
        player1.play_against(
            env_algorithm=model_evaluation,
            opponent=smart_max_damage_agent,
            env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
        )
        evaluation_results["final"]["vs_smax"] = player1.n_won_battles

    with open(os.path.join(output_dir, "results.json"), "w") as fp:
        json.dump(evaluation_results, fp)
