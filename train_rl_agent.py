# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py
import numpy as np
import tensorflow as tf

from agents.dqn_agent import SimpleRLPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from poke_env.player.random_player import RandomPlayer

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


NB_TRAINING_STEPS = 100000
NB_EVALUATION_EPISODES = 100
SEED = 42
MEMORY_SIZE = 10000
LOG_INTERVAL = 10000
TRAIN_INTERVAL = 1
TARGET_MODEL_UPDATE = 1000

tf.random.set_seed(SEED)
np.random.seed(SEED)

if __name__ == "__main__":
    experiment_name = f"dqn_SmartMaxDamage_train{TRAIN_INTERVAL}_targetmodel{TARGET_MODEL_UPDATE}"

    env_player = SimpleRLPlayer(battle_format="gen8randombattle")
    random_agent = RandomPlayer(battle_format="gen8randombattle")
    max_damage_agent = MaxDamagePlayer(battle_format="gen8randombattle")
    smart_max_damage_agent = SmartMaxDamagePlayer(battle_format="gen8randombattle")
    training_opponent = smart_max_damage_agent

    # Output dimension
    n_action = len(env_player.action_space)

    model = models.SimpleModel(n_action=n_action)
    memory = SequentialMemory(limit=MEMORY_SIZE, window_length=1)

    # Simple Epsilon Greedy Policy
    """
    EpsGreedyQPolicy: Standard epsilon greedy policy
    LinearAnnealedPolicy: Linearly anneals the given attribute
    This effectively just reduces the value of epsilon over time.
    It starts from max and slowly goes to min. 
    At testing time epsilon is set to 0.
    """
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )

    # Defining our DQN
    """
    Model, policy, memory and nb_actions are self-explanatory.
    Warm Up: 
    We use a very low LR for a fixed number of steps.
    Then, we increase it to the actual LR. This is done to both
    reduce overall variance and also combat overfitting.
    Especially in a case where the first few batches might be 
    highly correlated data which might skew the model.
    Gamma = Discount Factor

    target_model_update = Update model every X steps
    On other steps it does a "soft update" 
    Refer: https://github.com/keras-rl/keras-rl/issues/55

    delta_clip = Delta for clipping gradients.
    Refer: https://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/

    enable_double_dqn = Have a secondary target network
    According to the paper "Deep Reinforcement Learning with Double Q-learning"
    (van Hasselt et al., 2015), in Double DQN, the online network predicts the 
    actions while the target network is used to estimate the Q value.
    This model receives no updates, but instead clones the original's weights.

    train_interval = After how many steps do we perform a training update?
    This is different from the target model update. This updates our neural network. Target Model Update updates our secondary target network. 
    memory_interval = After how many steps do we add something to the memory.
    """
    dqn = DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=TARGET_MODEL_UPDATE,
        delta_clip=0.01,
        enable_double_dqn=True,
        train_interval=TRAIN_INTERVAL
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Training
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=training_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS},
    )
    model.save(experiment_name)

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
