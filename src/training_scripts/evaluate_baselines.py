# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_new_open_ai_gym_wrapper.py

import sys
import time
import asyncio

sys.path.append("./")

import numpy as np
import torch
import training_utils as utils
from poke_env.utils import to_id_str
from agents.max_damage_agent import MaxDamagePlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.player.random_player import RandomPlayer


async def evaluate(player1, player2, n_battles):
    await asyncio.gather(
        player1.send_challenges(
            opponent=to_id_str(player2.username),
            n_challenges=n_battles,
            to_wait=player2.logged_in,
        ),
        player2.accept_challenges(
            opponent=to_id_str(player1.username), n_challenges=n_battles
        ),
    )


if __name__ == "__main__":
    server_port = 8000
    start_time = time.time()

    # Config - Training Hyperparameters
    AGENT = "smart"  # random, max, smart
    RANDOM_SEED = 42
    n_battles = 1000  # Final Evaluation
    server_config = utils.generate_server_configuration(server_port)

    # Set random seed
    np.random.seed(RANDOM_SEED)
    _ = torch.manual_seed(RANDOM_SEED)

    # Setup evaluation agents
    random_agent = RandomPlayer(
        battle_format="gen8randombattle",
        server_configuration=server_config,
        start_timer_on_battle_start=True,
        max_concurrent_battles=10,
    )
    max_damage_agent = MaxDamagePlayer(
        battle_format="gen8randombattle",
        server_configuration=server_config,
        start_timer_on_battle_start=True,
        max_concurrent_battles=10,
    )
    smart_max_damage_agent = SimpleHeuristicsPlayer(
        battle_format="gen8randombattle",
        server_configuration=server_config,
        start_timer_on_battle_start=True,
        max_concurrent_battles=10,
    )

    if AGENT == "random":
        player = random_agent
    elif AGENT == "max":
        player = max_damage_agent
    elif AGENT == "smart":
        player = smart_max_damage_agent

    # Setup opponents
    random_opp = RandomPlayer(
        battle_format="gen8randombattle",
        server_configuration=server_config,
        start_timer_on_battle_start=True,
        max_concurrent_battles=10,
    )
    max_damage_opp = MaxDamagePlayer(
        battle_format="gen8randombattle",
        server_configuration=server_config,
        start_timer_on_battle_start=True,
        max_concurrent_battles=10,
    )
    smart_max_damage_opp = SimpleHeuristicsPlayer(
        battle_format="gen8randombattle",
        server_configuration=server_config,
        start_timer_on_battle_start=True,
        max_concurrent_battles=10,
    )

    asyncio.get_event_loop().run_until_complete(
        evaluate(player, random_opp, n_battles)
    )
    print(f"Random: {player.n_won_battles}/{n_battles}.")
    player.reset_battles()

    asyncio.get_event_loop().run_until_complete(
        evaluate(player, max_damage_opp, n_battles)
    )
    print(f"Max: {player.n_won_battles}/{n_battles}.")
    player.reset_battles()

    asyncio.get_event_loop().run_until_complete(
        evaluate(player, smart_max_damage_opp, n_battles)
    )
    print(f"Smart: {player.n_won_battles}/{n_battles}.")
    player.reset_battles()

    end_time = time.time()
    print(f"Running Time: {end_time - start_time}")
