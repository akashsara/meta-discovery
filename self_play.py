# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/experimental-self-play.py

import asyncio
import os
import numpy as np
from poke_env.player.random_player import RandomPlayer
from tabulate import tabulate
from threading import Thread

from poke_env.utils import to_id_str
from poke_env.player.utils import cross_evaluate
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from agents.dqn_agent import SimpleRLPlayer
from agents.dqn_full_state_agent import FullStatePlayer
import models

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

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
    model.test(player, nb_episodes=nb_episodes, visualize=False, verbose=2)

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


def create_model(mode, n_actions, **kwargs):
    if mode == "FullState":
        # Create Model
        state = kwargs["player"].create_empty_state_vector()
        state = kwargs["player"].state_to_machine_readable_state(state)
        max_values = kwargs["player"].lookup["max_values"]
        embedding_dim = kwargs["embedding_dim"]
        model = models.FullStateModel(n_actions, state, embedding_dim, max_values)
    elif mode == "SmallState":
        # Define Model
        model = Sequential()
        model.add(Dense(128, activation="elu", input_shape=(1, 10)))
        model.add(Flatten())
        model.add(Dense(64, activation="elu"))
        model.add(Dense(n_actions, activation="linear"))
    return model


def create_rl_network(model, n_action, memory_size, training_steps, **kwargs):
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
    return dqn


if __name__ == "__main__":
    random_seed = 42
    training_steps = 100000  # N/2 steps from each agent's perspective
    memory_size = 10000
    evaluation_episodes = 100
    train_interval = 1
    p1_log_interval = 10000
    p2_log_interval = 10000
    p1_verbose = 1
    p2_verbose = 0
    model_dir = "models"
    model_name = "self_play_full_state_dqn_100K_steps"
    mode = "FullState"
    embedding_dim = 128
    config = {
        "create": True,
        "pokemon_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json",
        "moves_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/moves.json",
        "items_json": "https://raw.githubusercontent.com/itsjavi/showdown-data/main/dist/data/items.json",
        "lookup_filename": "player_lookup_dicts.joblib",
    }

    # Create Output Path
    model_parent_dir = os.path.join(model_dir, model_name)
    model_output_dir = os.path.join(model_dir, model_name, model_name)
    config["lookup_filename"] = os.path.join(
        model_parent_dir, config["lookup_filename"]
    )
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    # Set Random Seed
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    # Create Players
    if mode == "FullState":
        player1 = FullStatePlayer(
            config, battle_format="gen8randombattle", log_level=25
        )
        config["create"] = False
        player2 = FullStatePlayer(
            config, battle_format="gen8randombattle", log_level=25
        )
    elif mode == "SmallState":
        player1 = SimpleRLPlayer(battle_format="gen8randombattle", log_level=25)
        player2 = SimpleRLPlayer(battle_format="gen8randombattle", log_level=25)
    n_actions = len(player1.action_space)
    # Create RL and Model Configs
    dqn_kwargs = {"train_interval": train_interval}
    player1_kwargs = {"verbose": p1_verbose, "log_interval": p1_log_interval}
    player2_kwargs = {"verbose": p2_verbose, "log_interval": p2_log_interval}
    # Create Model
    model = create_model(mode, n_actions, player=player1, embedding_dim=embedding_dim)
    # create RL Network
    dqn = create_rl_network(model, n_actions, memory_size, training_steps)
    # Create Environment Configs
    p1_env_kwargs = {"model": dqn, "nb_steps": training_steps, "kwargs": player1_kwargs}
    p2_env_kwargs = {"model": dqn, "nb_steps": training_steps, "kwargs": player2_kwargs}
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
    test_player = SimpleRLPlayer(battle_format="gen8randombattle", log_level=25)
    random_agent = RandomPlayer(battle_format="gen8randombattle", log_level=25)
    max_damage_agent = MaxDamagePlayer(battle_format="gen8randombattle", log_level=25)
    smart_max_damage_agent = SmartMaxDamagePlayer(battle_format="gen8randombattle")
    print("Results against random player:")
    test_player.play_against(
        env_algorithm=model_evaluation,
        opponent=random_agent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": evaluation_episodes},
    )

    print("\nResults against max player:")
    test_player.play_against(
        env_algorithm=model_evaluation,
        opponent=max_damage_agent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": evaluation_episodes},
    )

    print("\nResults against smart max player:")
    test_player.play_against(
        env_algorithm=model_evaluation,
        opponent=smart_max_damage_agent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": evaluation_episodes},
    )
