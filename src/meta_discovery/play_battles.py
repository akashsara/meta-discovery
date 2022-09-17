"""

On metagame vs playable_tier:
    For our experiment's sake we have two separate variables. 
    playable_tier determines the actual rules we're using. 
    metagame determines the rules we use while battling. 
    exclusions allows us to ignore certain rules.

    For instance if we set playable_tier = "ou", metagame = "ubers"
    and exclusions = ["power construct"]:
        Then we will essentially be playing only OU.
        Except Power Construct is unbanned.
        Since Showdown won't let us do that legally in OU, we instead
        play on ubers where Power Construct is naturally unbanned.

    tl;dr:
    playable_tier = tier you want to use
    exclusions = things you want to unban
        pokemon = pokemon to unban
        moveset = items/abilities/moves to unban
    metagame = lowest tier where all exclusions are legal
"""
import asyncio
import os
import random
import sys

sys.path.append("./")
import time

import joblib
import numpy as np
import torch
from agents import full_state_agent, simple_agent
from agents.max_damage_agent import MaxDamagePlayer
from agents.smart_max_damage_agent import SmartMaxDamagePlayer
from models import full_state_models, simple_models
from poke_env.player.random_player import RandomPlayer
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration

import utils
from meta_discovery import MetaDiscoveryDatabase, TeamBuilder
from policy import LinearDecayEpsilon

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")


async def play_battles(player1, player2, n_battles):
    await player1.battle_against(player2, n_battles=n_battles)


def setup_and_load_model(model, model_kwargs, model_path):
    model = model(**model_kwargs)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    moveset_db_path = "meta_discovery/data/moveset_database.joblib"
    meta_discovery_db_path = "meta_discovery/data/test.joblib"
    tier_list_path = "meta_discovery/data/tier_data.joblib"
    # Used to enforce species clause
    pokedex_json_path = "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json"
    # Total num. battles to simulate
    num_battles_to_simulate = 20000
    # Num. battles between generating new teams
    team_generation_interval = 1000
    # Num. teams generated
    num_teams_to_generate = 2500
    # Exploration Factor - Epsilon
    # Prob. of using inverse pickrate over winrate
    epsilon_min = 0.001
    epsilon_max = 1.0
    epsilon_decay = 20000
    # Set local port that Showdown is running on
    server_port = 8000
    # Metagame / Ban List Selection [read comment at the top]
    metagame = "gen8ubers"
    playable_tier = "ou"
    banlist_pokemon_exclusions = ["zygarde"]
    banlist_moveset_exclusions = []
    # Number of battles to run simultaneously
    max_concurrent_battles = 25
    # Set random seed for reproducible results
    random_seed = 42

    # Setup player information
    player1_class = SimpleHeuristicsPlayer  # simple_agent.GeneralAPISimpleAgent
    player1_config = PlayerConfiguration(f"{server_port}_BattleAgent1", None)
    # This is used only if it's a model-based agent
    player1_model_class = simple_models.SimpleActorCriticModel
    player1_model_path = "outputs/Simple_PPO_Base_v2.1/model_1024000.pt"
    player1_model_kwargs = {
        "n_actions": 22,
        "n_obs": 10,
    }

    player2_class = SimpleHeuristicsPlayer  # simple_agent.GeneralAPISimpleAgent
    player2_config = PlayerConfiguration(f"{server_port}_BattleAgent2", None)
    # This is used only if it's a model-based agent
    player2_model_class = simple_models.SimpleActorCriticModel
    player2_model_path = "outputs/Simple_PPO_Base_v2.1/model_1024000.pt"
    player2_model_kwargs = {
        "n_actions": 22,
        "n_obs": 10,
    }

    # Use random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    _ = torch.manual_seed(random_seed)

    # Special case for zygarde we need to unban a pokemon + ability
    if "zygarde" in banlist_pokemon_exclusions:
        banlist_moveset_exclusions.append("powerconstruct")

    # Setup banlist
    print("---" * 40)
    print(f"Tier Selected: {playable_tier}")
    ban_list = utils.get_ban_list(playable_tier, tier_list_path, banlist_pokemon_exclusions)
    print("---" * 30)
    print("Ban List in Effect:")
    print(ban_list)

    # Load moveset DB
    moveset_database = joblib.load(moveset_db_path)
    # Remove illegal moves/items/abilities based on tier
    # Also remove Pokemon that have no movesets due to the above
    print("---" * 30)
    moveset_database, ban_list = utils.legality_checker(
        moveset_database, playable_tier, ban_list, banlist_moveset_exclusions
    )
    print("Final Banlist:")
    print(ban_list)
    # Setup meta discovery database & load existing one if possible
    print("---" * 30)
    print("Setting up Meta Discovery database.")
    meta_discovery_database = MetaDiscoveryDatabase(moveset_database)
    if os.path.exists(meta_discovery_db_path):
        print("Found existing database. Loading...")
        meta_discovery_database.load(meta_discovery_db_path)
        print(f"Load complete. {meta_discovery_database.num_battles} Battles Complete")

    exploration_control = LinearDecayEpsilon(epsilon_max, epsilon_min, epsilon_decay)

    all_pokemon = list(meta_discovery_database.key2pokemon.values())
    team_builder = TeamBuilder(
        exploration_control, moveset_database, all_pokemon, pokedex_json_path, ban_list
    )

    # Setup kwargs & load trained models to use for each agent if necessary
    player1_kwargs = {}
    player2_kwargs = {}
    if "agent" in str(player1_class):
        player1_kwargs["model"] = setup_and_load_model(
            player1_model_class, player1_model_kwargs, player1_model_path
        )
        player1_kwargs["device"] = device
    if "agent" in str(player2_class):
        player2_kwargs["model"] = setup_and_load_model(
            player2_model_class, player2_model_kwargs, player2_model_path
        )
        player2_kwargs["device"] = device

    # Setup server configuration
    # Maintain servers on different ports to avoid Compute Canada errors
    server_config = utils.generate_server_configuration(server_port)

    # Create our battle agents
    player1 = player1_class(
        battle_format=metagame,
        max_concurrent_battles=max_concurrent_battles,
        player_configuration=player1_config,
        server_configuration=server_config,
        start_timer_on_battle_start=True,
        **player1_kwargs,
    )
    player2 = player2_class(
        battle_format=metagame,
        max_concurrent_battles=max_concurrent_battles,
        player_configuration=player2_config,
        server_configuration=server_config,
        start_timer_on_battle_start=True,
        **player1_kwargs,
    )

    # Meta Discovery Loop Starts Here
    print("---" * 30)
    start_time = time.time()
    for i in range(num_battles_to_simulate // team_generation_interval):
        print(f"Epoch {i+1}")

        # Generate new teams
        print("Generating New Teams")
        start = time.time()
        team_builder.generate_teams(meta_discovery_database, num_teams_to_generate)
        end = time.time()
        print(f"Time Taken: {end - start}s")

        # Make agents use the generated teams
        player1._team = team_builder
        player2._team = team_builder

        # Play battles
        print("Battling")
        start = time.time()
        asyncio.run(
            play_battles(
                player1=player1,
                player2=player2,
                n_battles=team_generation_interval,
            )
        )
        end = time.time()
        print(f"Battles Completed: {team_generation_interval}")
        print(f"Time Taken: {end - start}s")

        # Extract stats from the battles played
        # We get battle IDs from p1 since both players are in every battle.
        # This needs to be changed if we have multiple players
        player1_all_wins = []
        player1_all_losses = []
        player2_all_wins = []
        player2_all_losses = []
        for battle in player1.battles:
            p1_team = [
                pokemon.species for _, pokemon in player1.battles[battle].team.items()
            ]
            p2_team = [
                pokemon.species for _, pokemon in player2.battles[battle].team.items()
            ]
            if player1.battles[battle].won:
                player1_all_wins.extend(p1_team)
                player2_all_losses.extend(p2_team)
            else:
                player2_all_wins.extend(p2_team)
                player1_all_losses.extend(p1_team)

        # Reset trackers so we don't count battles twice
        player1.reset_battles()
        player2.reset_battles()

        # Update overall statistics
        meta_discovery_database.update_battle_statistics(
            player1_all_wins + player2_all_wins,
            player1_all_losses + player2_all_losses,
            team_generation_interval,
        )
        # Save DB
        meta_discovery_database.save(meta_discovery_db_path)
    end_time = time.time()
    print(f"Simulation Time Taken: {end_time - start_time:.4f}")

    # Meta Discovery
    print("Fin.")
