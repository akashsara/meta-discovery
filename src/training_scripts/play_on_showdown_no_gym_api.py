# -*- coding: utf-8 -*-
# https://poke-env.readthedocs.io/en/stable/connecting_to_showdown_and_challenging_humans.html
import asyncio
import sys

sys.path.append("./")

import torch
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from pokemon_showdown_accounts import id_dict


# Choose Agent:
#   Random Agent, Max Damage Agent, Smart Max Damage Agent
AGENT = "Smart Max Damage Agent"
# Choose Mode:
#   LADDER = Play 100 Matches on Ladder
#   CHALLENGE = Accept a single challenge from any user on Showdown
MODE = "CHALLENGE"
NUM_GAMES = 1
USERNAME = id_dict[AGENT]["username"]
PASSWORD = id_dict[AGENT]["password"]

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")


async def main():
    print("Loading model...")
    # RandomPlayer
    if AGENT == "Random Agent":
        player = RandomPlayer(
            player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
        )
    # MaxDamagePlayer
    elif AGENT == "Max Damage Agent":
        player = MaxDamagePlayer(
            player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
        )
    # SmartMaxDamagePlayer
    elif AGENT == "Smart Max Damage Agent":
        player = SmartMaxDamagePlayer(
            player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
        )

    print("Connecting to Pokemon Showdown...")
    # Playing games on the ladder
    if MODE == "LADDER":
        await player.ladder(NUM_GAMES)
    # Accepting challenges from any user
    elif MODE == "CHALLENGE":
        print(f"/challenge {USERNAME}")
        await player.accept_challenges(None, NUM_GAMES)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
