# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py

import asyncio
import os
import numpy as np
from poke_env.player.random_player import RandomPlayer
from threading import Thread

from poke_env.utils import to_id_str
from poke_env.player_configuration import PlayerConfiguration
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from agents.dqn_agent import SimpleRLPlayer
import models
import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

def env_algorithm_wrapper(env_algorithm, player, kwargs):
    env_algorithm(player, **kwargs)

    player._start_new_battle = False
    while True:
        try:
            player.complete_current_battle()
            player.reset()
        except OSError:
            break


# This is the function that will be used to train the model
def model_training(player, model, nb_steps, kwargs):
    model.fit(player, nb_steps=nb_steps, **kwargs)
    player.complete_current_battle()


def model_evaluation(player, model, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    model.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
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


if __name__ == "__main__":
    # Config - Hyperparameters
    NB_TRAINING_STEPS = 10000 # N/2 steps from each agent's perspective.
    NB_EVALUATION_EPISODES = 0
    MEMORY_SIZE = 10000
    LOG_INTERVAL = 1000
    TRAIN_INTERVAL = 10
    TARGET_MODEL_UPDATE = 1000
    RANDOM_SEED = 42

    # Config - Logging stuff 
    p1_log_interval = LOG_INTERVAL
    p2_log_interval = LOG_INTERVAL
    p1_verbose = 1
    p2_verbose = 0
    
    # Config - Versioning
    experiment_name = f"Simple_SelfPlay_DQN_Base_v1"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Set Random Seed
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Setup agent usernames for connecting to local showdown 
    # This lets us train multiple agents while connecting to the same server
    training_agent1 = PlayerConfiguration(hash_name + "_P1", None)
    training_agent2 = PlayerConfiguration(hash_name + "_P2", None)
    eval_agent = PlayerConfiguration(hash_name + "_Eval", None)
    rand_player = PlayerConfiguration(hash_name + "_Rand", None)
    max_player = PlayerConfiguration(hash_name + "_Max", None)
    smax_player = PlayerConfiguration(hash_name + "_SMax", None)

    # Create Output Path
    model_parent_dir = os.path.join(model_dir, experiment_name)
    model_output_dir = os.path.join(model_parent_dir, experiment_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Create Players
    player1 = SimpleRLPlayer(battle_format="gen8randombattle", log_level=50)
    player2 = SimpleRLPlayer(battle_format="gen8randombattle", log_level=50)
    n_actions = len(player1.action_space)

    # Create RL and Model Configs
    player1_kwargs = {"verbose": p1_verbose, "log_interval": p1_log_interval}
    player2_kwargs = {"verbose": p2_verbose, "log_interval": p2_log_interval}

    # Create Model
    model = models.SimpleModel(n_action=n_actions)
    print(model.summary())

    # Define Memory
    memory = SequentialMemory(limit=MEMORY_SIZE, window_length=1)

    # Define Policy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )

    # Create RL Network
    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        delta_clip=0.01,
        enable_double_dqn=True,
        target_model_update=TARGET_MODEL_UPDATE,
        train_interval=TRAIN_INTERVAL
    )

    # Compile Network
    dqn.compile(Adam(learning_rate=0.0005), metrics=["mae"])

    # Create Environment Configs
    p1_env_kwargs = {"model": dqn, "nb_steps": NB_TRAINING_STEPS, "kwargs": player1_kwargs}
    p2_env_kwargs = {"model": dqn, "nb_steps": NB_TRAINING_STEPS, "kwargs": player2_kwargs}

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

    # Save
    model.save(model_output_dir)

    # Evaluation
    test_player = SimpleRLPlayer(
        battle_format="gen8randombattle", log_level=50, player_configuration=eval_agent
    )
    
    print("Results against random player:")
    random_agent = RandomPlayer(
        battle_format="gen8randombattle", log_level=50, player_configuration=rand_player
    )
    test_player.play_against(
        env_algorithm=model_evaluation,
        opponent=random_agent,
        env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    max_damage_agent = MaxDamagePlayer(
        battle_format="gen8randombattle", log_level=50, player_configuration=max_player
    )
    test_player.play_against(
        env_algorithm=model_evaluation,
        opponent=max_damage_agent,
        env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against smart max player:")
    smart_max_damage_agent = SmartMaxDamagePlayer(
        battle_format="gen8randombattle", log_level=50, layer_configuration=smax_player
    )
    test_player.play_against(
        env_algorithm=model_evaluation,
        opponent=smart_max_damage_agent,
        env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )
