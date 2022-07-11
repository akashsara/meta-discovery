# -*- coding: utf-8 -*-
# https://poke-env.readthedocs.io/en/stable/connecting_to_showdown_and_challenging_humans.html
import asyncio

from poke_env.player.random_player import RandomPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from agents.simple_agent import SimpleRLPlayerTesting
from agents.full_state_agent import FullStatePlayerTesting

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
import joblib
from tensorflow import keras
from pokemon_showdown_accounts import id_dict

# Choose Agent:
#   Random Agent, Max Damage Agent, Smart Max Damage Agent
#   Simple State DQN Agent, Full State DQN Agent
AGENT = "Full State DQN Agent"
# Choose Mode: 
#   LADDER = Play 100 Matches on Ladder
#   CHALLENGE = Accept a single challenge from any user on Showdown
MODE = "CHALLENGE"
NUM_GAMES = 1
USERNAME = id_dict[AGENT]["username"]
PASSWORD = id_dict[AGENT]["password"]

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
    # DQN Agent - Simple State
    elif AGENT == "Simple State DQN Agent":
        model_loc = id_dict[AGENT]["model_loc"]
        model = keras.models.load_model(model_loc)
        player = SimpleRLPlayerTesting(
            model=model,
            player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
        )
    elif AGENT == "Full State DQN Agent":
        model_loc = id_dict[AGENT]["model_loc"]
        model= keras.models.load_model(model_loc)
        config = {
            "create": False,
            "lookup_filename": id_dict[AGENT]["config_loc"],
        }
        player = FullStatePlayerTesting(
            model=model,
            player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
            config=config,
        )

    print("Connecting to Pokemon Showdown...")
    # Playing games on the ladder
    if MODE == "LADDER":
        await player.ladder(NUM_GAMES)
        # Print the rating of the player and its opponent after each battle
        for battle in player.battles.values():
            print(battle.rating, battle.opponent_rating)
    # Accepting challenges from any user
    elif MODE == "CHALLENGE":
        await player.accept_challenges(None, NUM_GAMES)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
