# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py

# self-play training is a planned feature for poke-env
# This script illustrates a very rough approach that can currently be used to train using self-play
# Don't hesitate to open an issue if things seem not to be working

import asyncio
import sys
import numpy as np
from poke_env.player.random_player import RandomPlayer
from tabulate import tabulate
from threading import Thread

from poke_env.utils import to_id_str
from poke_env.player.utils import cross_evaluate
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from agents.dqn_agent import SimpleRLPlayer

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Goal:
# Create a single RL model.
# Pass this model in as the agent for both players.
# Play N battles, storing the results for future use
# Once N battles are done, train the agent on the 2N battle data.


def env_algorithm_wrapper(env_algorithm, player, kwargs):
    env_algorithm(player, **kwargs)

    player._start_new_battle = False
    while True:
        try:
            player.complete_current_battle()
            player.reset()
        except OSError:
            break


# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps, kwargs):
    dqn.fit(player, nb_steps=nb_steps, **kwargs)
    player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=2)

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


def create_rl_network(n_action, memory_size, training_steps, **kwargs):
    # Define Model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Define Memory
    memory = SequentialMemory(limit=memory_size, window_length=1)

    # Define Policy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=training_steps,
    )

    # Define Agent
    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
        **kwargs
    )

    # Compile Network
    dqn.compile(Adam(learning_rate=0.0005), metrics=["mae"])

    # Return
    return dqn, model


if __name__ == "__main__":
    random_seed = 42
    training_steps = 100000 # N/2 steps from each agent's perspective
    memory_size = 10000
    evaluation_episodes = 100
    train_interval = 1
    p1_log_interval = 10000
    p2_log_interval = 10000
    p1_verbose = 1
    p2_verbose = 0
    model_name = "self_play_dqn_100K_steps"

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    player1 = SimpleRLPlayer(battle_format="gen8randombattle", log_level=25)
    player2 = SimpleRLPlayer(battle_format="gen8randombattle", log_level=25)
    n_action = len(player1.action_space)

    dqn_kwargs = {"train_interval": train_interval}
    player1_kwargs = {"verbose": p1_verbose, "log_interval": p1_log_interval}
    player2_kwargs = {"verbose": p2_verbose, "log_interval": p2_log_interval}

    dqn, model = create_rl_network(n_action, memory_size, training_steps)
    
    p1_env_kwargs = {"dqn": dqn, "nb_steps": training_steps, "kwargs": player1_kwargs}
    p2_env_kwargs = {"dqn": dqn, "nb_steps": training_steps, "kwargs": player2_kwargs}

    # Make Two Threads And Play vs Each Other
    player1._start_new_battle = True
    player2._start_new_battle = True

    loop = asyncio.get_event_loop()

    t1 = Thread(target=lambda: env_algorithm_wrapper(dqn_training, player1, p1_env_kwargs))
    t1.start()
    
    t2 = Thread(target=lambda: env_algorithm_wrapper(dqn_training, player2, p2_env_kwargs))
    t2.start()

    while player1._start_new_battle:
        loop.run_until_complete(launch_battles(player1, player2))
    t1.join()
    t2.join()

    # Save
    model.save(model_name)

    # Evaluation
    test_player = SimpleRLPlayer(battle_format="gen8randombattle", log_level=25)
    random_agent = RandomPlayer(battle_format="gen8randombattle", log_level=25)
    max_damage_agent = MaxDamagePlayer(battle_format="gen8randombattle", log_level=25)
    smart_max_damage_agent = SmartMaxDamagePlayer(battle_format="gen8randombattle")
    print("Results against random player:")
    test_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=random_agent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": evaluation_episodes},
    )

    print("\nResults against max player:")
    test_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=max_damage_agent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": evaluation_episodes},
    )

    print("\nResults against smart max player:")
    test_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=smart_max_damage_agent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": evaluation_episodes},
    )
