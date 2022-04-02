# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
"""
used for training a simple RL agent.
So small state vs random/max_damage/smart max_damage
"""
import os
import numpy as np
import tensorflow as tf

from agents.dqn_agent import SimpleRLPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam
import models

# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps=nb_steps, verbose=1, log_interval=LOG_INTERVAL)
    player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )

if __name__ == "__main__":
    # Config - Hyperparameters
    NB_TRAINING_STEPS = 10000 
    NB_EVALUATION_EPISODES = 100
    MEMORY_SIZE = 10000
    LOG_INTERVAL = 1000
    TRAIN_INTERVAL = 1
    TARGET_MODEL_UPDATE = 1000
    RANDOM_SEED = 42

    # Config - Versioning
    training_opponent = "smart" # random, max, smart
    experiment_name = f"Simple_DQN_Base_v1"
    hash_name = str(hash(experiment_name))[2:12]

    # Config - Model Save Directory
    model_dir = "models"

    # Set random seed
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Setup agent usernames for connecting to local showdown 
    # This lets us train multiple agents while connecting to the same server
    training_agent = PlayerConfiguration(hash_name + "_P1", None)
    rand_player = PlayerConfiguration(hash_name + "_Rand", None)
    max_player = PlayerConfiguration(hash_name + "_Max", None)
    smax_player = PlayerConfiguration(hash_name + "_SMax", None)

    # Create Output Path
    model_parent_dir = os.path.join(model_dir, experiment_name)
    model_output_dir = os.path.join(model_parent_dir, experiment_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Setup player
    env_player = SimpleRLPlayer(battle_format="gen8randombattle", player_configuration=training_agent)
    
    # Setup opponents
    random_agent = RandomPlayer(battle_format="gen8randombattle", player_configuration=rand_player)
    max_damage_agent = MaxDamagePlayer(battle_format="gen8randombattle", player_configuration=max_player)
    smart_max_damage_agent = SmartMaxDamagePlayer(battle_format="gen8randombattle", player_configuration=smax_player)
    if training_opponent == "random":
        training_opponent = random_agent
    elif training_opponent == "max":
        training_opponent = max_damage_agent
    elif training_opponent == "smart":
        training_opponent = smart_max_damage_agent
    else:
        raise ValueError("Unknown training opponent.")

    # Output dimension
    n_actions = len(env_player.action_space)

    # Define Model
    model = models.SimpleModel(n_action=n_actions)

    # Setup memory
    memory = SequentialMemory(limit=MEMORY_SIZE, window_length=1)

    # Simple Epsilon Greedy Policy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )

    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        delta_clip=0.01,
        enable_double_dqn=True,
        target_model_update=TARGET_MODEL_UPDATE,
        train_interval=TRAIN_INTERVAL
    )

    # Compile Model
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Train Model
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=training_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS},
    )
    model.save(model_output_dir)

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=random_agent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=max_damage_agent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against smart max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=smart_max_damage_agent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )
