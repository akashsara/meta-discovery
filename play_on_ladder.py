# -*- coding: utf-8 -*-
# https://poke-env.readthedocs.io/en/stable/connecting_to_showdown_and_challenging_humans.html
import asyncio

from poke_env.player.random_player import RandomPlayer
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from agents.dqn_agent import SimpleRLPlayerTesting

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
import models
from tensorflow import keras

# Random Agent: UAB GL RA: TheThirdLawOfRobotics
# Max Damage: UAB GL MD: TheThirdLawOfRobotics
# Smart Max Damage: UAB GL Smart MD: TheThirdLawOfRobotics
USERNAME = "UAB GL Smart MD"
PASSWORD = "TheThirdLawOfRobotics"


async def main():
    # RandomPlayer
    # player = RandomPlayer(
    #     player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
    #     server_configuration=ShowdownServerConfiguration,
    #     start_timer_on_battle_start=True,
    # )

    # MaxDamagePlayer
    # player = MaxDamagePlayer(
    #     player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
    #     server_configuration=ShowdownServerConfiguration,
    #     start_timer_on_battle_start=True,
    # )

    # SmartMaxDamagePlayer
    # player = SmartMaxDamagePlayer(
    #     player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
    #     server_configuration=ShowdownServerConfiguration,
    #     start_timer_on_battle_start=True,
    # )

    # DQN Agent - Simple
    model_loc = "models/dqn_SmartMaxDamage/"
    model = keras.models.load_model(model_loc)
    player = SimpleRLPlayerTesting(
        model=model,
        player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
        server_configuration=ShowdownServerConfiguration,
        start_timer_on_battle_start=True,
    )

    # Sending challenges to 'your_username'
    # await player.send_challenges("username", n_challenges=1)

    # Accepting one challenge from any user
    await player.accept_challenges(None, 1)

    # Accepting three challenges from 'your_username'
    # await player.accept_challenges('your_username', 3)

    # Playing games on the ladder
    # await player.ladder(50)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
