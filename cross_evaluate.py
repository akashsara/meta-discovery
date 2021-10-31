# -*- coding: utf-8 -*-
# https://github.com/hsahovic/poke-env/blob/master/examples/cross_evaluate_random_players.py
import asyncio

from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from tabulate import tabulate
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer


async def main():
    # We create three random players
    players = [
        SmartMaxDamagePlayer(battle_format="gen8randombattle"),
        MaxDamagePlayer(battle_format="gen8randombattle")
    ]

    # Now, we can cross evaluate them: every player will player 20 games against every
    # other player.
    cross_evaluation = await cross_evaluate(players, n_challenges=100)

    # Defines a header for displaying results
    table = [["-"] + [p.username for p in players]]

    # Adds one line per player with corresponding results
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Displays results in a nicely formatted table.
    print(tabulate(table))

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())