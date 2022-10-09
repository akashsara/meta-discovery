# -*- coding: utf-8 -*-
# https://poke-env.readthedocs.io/en/stable/connecting_to_showdown_and_challenging_humans.html
import asyncio
import sys

sys.path.append("./")

import joblib
import numpy as np
import torch
from agents import full_state_agent, simple_agent
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from models import full_state_models, simple_models
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from pokemon_showdown_accounts import id_dict

# Choose Agent:
#   Random Agent, Max Damage Agent, Smart Max Damage Agent
#   Simple State DQN Agent, Full State DQN Agent
#   Simple State SelfPlay DQN Agent, Full State SelfPlay DQN Agent
#   Simple State PPO Agent, Full State PPO Agent
#   Simple State SelfPlay PPO Agent, Full State SelfPlay PPO Agent
AGENT = "Full State DQN Agent"
# Choose Mode:
#   LADDER = Play 100 Matches on Ladder
#   CHALLENGE = Accept a single challenge from any user on Showdown
MODE = "CHALLENGE"
NUM_GAMES = 1
OPPONENT = "DarkeKnight"  # Only used in CHALLENGE mode
USERNAME = id_dict[AGENT]["username"]
PASSWORD = id_dict[AGENT]["password"]

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")


async def main():
    print("Loading model...")
    if AGENT in [
        "Simple State DQN Agent",
        "Simple State SelfPlay DQN Agent",
    ]:
        model_path = id_dict[AGENT]["model_path"]
        model_kwargs = id_dict[AGENT]["model_kwargs"]
        player_kwargs = id_dict[AGENT]["player_kwargs"]
        # Create model
        model = simple_models.SimpleModel(**model_kwargs)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Eval mode
        model.eval()
        # Setup player
        player = simple_agent.GeneralAPISimpleAgent(
            model=model,
            player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
            device=device,
            **player_kwargs,
        )
    # Simple State PPO Agent
    elif AGENT in [
        "Simple State PPO Agent",
        "Simple State SelfPlay PPO Agent",
    ]:
        model_path = id_dict[AGENT]["model_path"]
        model_kwargs = id_dict[AGENT]["model_kwargs"]
        player_kwargs = id_dict[AGENT]["player_kwargs"]
        # Create model
        model = simple_models.SimpleActorCriticModel(**model_kwargs)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Eval mode
        model.eval()
        # Setup player
        player = simple_agent.GeneralAPISimpleAgent(
            model=model,
            player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
            device=device,
            **player_kwargs,
        )
    # Full State DQN Agent
    elif AGENT in [
        "Full State DQN Agent",
        "Full State SelfPlay DQN Agent",
    ]:
        model_path = id_dict[AGENT]["model_path"]
        model_kwargs = id_dict[AGENT]["model_kwargs"]
        player_kwargs = id_dict[AGENT]["player_kwargs"]
        # Setup temporary player to get some values
        temp_player = full_state_agent.FullStatePlayer(
            opponent=None,
            config={
                "create": False,
                "lookup_filename": player_kwargs["lookup_filename"],
            },
        )
        model_kwargs["state_length_dict"] = temp_player.get_state_lengths()
        model_kwargs["max_values_dict"] = temp_player.lookup["max_values"]
        del temp_player
        # Create model
        model = full_state_models.BattleModel(**model_kwargs)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Eval mode
        model.eval()
        # Setup player
        player = full_state_agent.GeneralAPIFullStateAgent(
            model=model,
            player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
            device=device,
            **player_kwargs,
        )
    # Full State PPO Agent
    elif AGENT in [
        "Full State PPO Agent",
        "Full State SelfPlay PPO Agent",
    ]:
        model_path = id_dict[AGENT]["model_path"]
        model_kwargs = id_dict[AGENT]["model_kwargs"]
        player_kwargs = id_dict[AGENT]["player_kwargs"]
        # Setup temporary player to get some values
        temp_player = full_state_agent.FullStatePlayer(
            opponent=None,
            config={
                "create": False,
                "lookup_filename": player_kwargs["lookup_filename"],
            },
        )
        model_kwargs["state_length_dict"] = temp_player.get_state_lengths()
        model_kwargs["max_values_dict"] = temp_player.lookup["max_values"]
        del temp_player
        # Create model
        model = full_state_models.ActorCriticBattleModel(**model_kwargs)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Eval mode
        model.eval()
        # Setup player
        player = full_state_agent.GeneralAPIFullStateAgent(
            model=model,
            player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
            device=device,
            **player_kwargs,
        )

    print("Connecting to Pokemon Showdown...")
    # Playing games on the ladder
    if MODE == "LADDER":
        print(f"{USERNAME} LADDERING")
        await player.ladder(NUM_GAMES)
    # Accepting challenges from any user
    elif MODE == "CHALLENGE":
        print(f"{USERNAME} CHALLENGING")
        print(f"/challenge {USERNAME}")
        await player.accept_challenges(None, NUM_GAMES)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
