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
    meta_discovery_db_path = "meta_discovery/data/meta_discovery_database.joblib"
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
    epsilon_min = 0.1
    epsilon_max = 1.0
    epsilon_decay = 20000
    # Set local port that Showdown is running on
    server_port = 8000
    # Choose metagame to play in
    metagame = "gen8ubers"
    # Set random seed for reproducible results
    random_seed = 42
    # Setup player information
    player1_model_class = simple_models.SimpleActorCriticModel
    player1_model_path = "outputs/Simple_PPO_Base_v2.1/model_1024000.pt"
    player1_model_kwargs = {
        "n_actions": 22,
        "n_obs": 10,
    }
    player1_config = PlayerConfiguration("Battle_Agent_1", None)
    player2_model_class = simple_models.SimpleActorCriticModel
    player2_model_path = "outputs/Simple_PPO_SelfPlay_v2.0/model_2047756.pt"
    player2_model_kwargs = {
        "n_actions": 22,
        "n_obs": 10,
    }
    player2_config = PlayerConfiguration("Battle_Agent_2", None)

    # Use random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    _ = torch.manual_seed(random_seed)

    # Setup banlist
    current_tier = metagame.split("gen8")[1]
    print("---" * 40)
    print(f"Tier Selected: {current_tier}")
    ban_list = utils.get_ban_list(current_tier, tier_list_path)
    print("---" * 30)
    print("Ban List in Effect:")
    print(ban_list)

    # Load moveset DB
    moveset_database = joblib.load(moveset_db_path)
    # Remove illegal moves/items/abilities based on tier
    # Also remove Pokemon that have no movesets due to the above
    print("---" * 30)
    moveset_database, ban_list = utils.legality_checker(
        moveset_database, current_tier, ban_list
    )
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

    # Load trained models to use for each agent
    player1_model = setup_and_load_model(
        player1_model_class, player1_model_kwargs, player1_model_path
    )
    player2_model = setup_and_load_model(
        player2_model_class, player2_model_kwargs, player2_model_path
    )

    # Setup server configuration
    # Maintain servers on different ports to avoid Compute Canada errors
    server_config = utils.generate_server_configuration(server_port)

    # Create our battle agents
    player1_kwargs = {}
    player2_kwargs = {}

    player1 = simple_agent.GeneralAPISimpleAgent(
        battle_format=metagame,
        max_concurrent_battles=25,
        model=player1_model,
        device=device,
        start_timer_on_battle_start=True,
        player_configuration=player1_config,
        server_configuration=server_config,
        **player1_kwargs,
    )

    player2 = simple_agent.GeneralAPISimpleAgent(
        battle_format=metagame,
        max_concurrent_battles=25,
        model=player2_model,
        device=device,
        start_timer_on_battle_start=True,
        player_configuration=player2_config,
        server_configuration=server_config,
        **player2_kwargs,
    )

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
    end_time = time.time()
    print(f"Simulation Time Taken: {end_time - start_time:.4f}")

    meta_discovery_database.save(meta_discovery_db_path)

    # Meta Discovery
    print("Fin.")
