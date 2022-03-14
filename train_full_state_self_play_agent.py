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
from rl.core import Processor

class DQNAgentModified(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        # state -> state[0] due to our complex input
        q_values = self.compute_q_values(state[0])
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action


class MultiInputProcessor(Processor):
    def __init__(self):
        Processor.__init__(self)

    def process_state_batch(self, state_batch):
        final_state = []
        if len(state_batch) == 1:
            for items in zip(*state_batch):
                final_state.append(items[0].reshape(1, -1))
        else:
            state_batch = np.squeeze(np.array(state_batch))
            for items in zip(*state_batch):
                final_state.append(np.stack(items, axis=0))
        return final_state


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


def create_model(n_actions, player, embedding_dim):
    # Create Model
    state = player.create_empty_state_vector()
    state = player.state_to_machine_readable_state(state)
    action_mask = np.array([0.0 for i in range(22)], dtype="float32")
    state += [action_mask]
    max_values = player.lookup["max_values"]
    model = models.FullStateModel(n_actions, state, embedding_dim, max_values)
    processor = MultiInputProcessor()
    return model, processor


if __name__ == "__main__":
    # Config - Hyperparameters
    NB_TRAINING_STEPS = 10000 # N/2 steps from each agent's perspective.
    NB_EVALUATION_EPISODES = 100
    MEMORY_SIZE = 10000
    LOG_INTERVAL = 1000
    TRAIN_INTERVAL = 1
    TARGET_MODEL_UPDATE = 1000
    RANDOM_SEED = 42
    embedding_dim = 128

    # Config - Logging stuff 
    p1_log_interval = LOG_INTERVAL
    p2_log_interval = LOG_INTERVAL
    p1_verbose = 1
    p2_verbose = 0
    
    # Config - Versioning
    version = 1
    model_name = "Random"
    experiment_name = f"FullState_SelfPlay_DQN_{model_name}_v{version}"

    # Config - Model Save Directory
    model_dir = "models"
    config = {
        "create": True,
        "pokemon_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json",
        "moves_json": "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/moves.json",
        "items_json": "https://raw.githubusercontent.com/itsjavi/showdown-data/main/dist/data/items.json",
        "lookup_filename": "player_lookup_dicts.joblib",
    }

    # Set Random Seed
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Create Output Path
    model_parent_dir = os.path.join(model_dir, experiment_name)
    model_output_dir = os.path.join(model_parent_dir, experiment_name)
    config["lookup_filename"] = os.path.join(
        model_parent_dir, config["lookup_filename"]
    )
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Create Players
    player1 = FullStatePlayer(
        config, battle_format="gen8randombattle", log_level=50
    )
    config["create"] = False
    player2 = FullStatePlayer(
        config, battle_format="gen8randombattle", log_level=50
    )
    n_actions = len(player1.action_space)

    # Create RL and Model Configs
    dqn_kwargs = {"train_interval": TRAIN_INTERVAL}
    player1_kwargs = {"verbose": p1_verbose, "log_interval": p1_log_interval}
    player2_kwargs = {"verbose": p2_verbose, "log_interval": p2_log_interval}

    # Create Model
    model, processor = create_model(
        n_actions, player=player1, embedding_dim=embedding_dim
    )
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

    # Define Agent
    dqn = DQNAgentModified(
        model=model,
        processor=processor,
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
    test_player = FullStatePlayer(
        config, battle_format="gen8randombattle", log_level=50
    )

    print("Results against random player:")
    random_agent = RandomPlayer(battle_format="gen8randombattle", log_level=50)
    test_player.play_against(
        env_algorithm=model_evaluation,
        opponent=random_agent,
        env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    max_damage_agent = MaxDamagePlayer(battle_format="gen8randombattle", log_level=50)
    test_player.play_against(
        env_algorithm=model_evaluation,
        opponent=max_damage_agent,
        env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against smart max player:")
    smart_max_damage_agent = SmartMaxDamagePlayer(
        battle_format="gen8randombattle", log_level=50
    )
    test_player.play_against(
        env_algorithm=model_evaluation,
        opponent=smart_max_damage_agent,
        env_algorithm_kwargs={"model": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )
