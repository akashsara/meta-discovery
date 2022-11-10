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
from models import full_state_models, simple_models
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration

import utils
from meta_discovery import MetaDiscoveryDatabase
from policy import LinearDecayEpsilon
from team_builder import TeamBuilder

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


def get_player_class(model_type):
    if model_type == "simple":
        return simple_agent.GeneralAPISimpleAgent
    elif model_type == "heuristic":
        return SimpleHeuristicsPlayer
    else:
        return full_state_agent.GeneralAPIFullStateAgent


def make_model(
    model_type,
    model_kwargs,
    model_path,
    lookup_path=None,
):
    if model_type == "simple":
        model_class = simple_models.SimpleActorCriticModel
        model_kwargs["n_actions"] = 22
        model_kwargs["n_obs"] = 10
    elif model_type in ["full", "flatten"]:
        # Create a temporary player to get sample inputs/outputs to build
        # the model with
        temp = full_state_agent.FullStatePlayer(
            config={"create": False, "lookup_filename": lookup_path}, opponent=None
        )
        state = temp.create_empty_state_vector()
        state = temp.state_to_machine_readable_state(state)
        state_length_dict = temp.get_state_lengths()
        max_values_dict = temp.lookup["max_values"]
        n_actions = temp.action_space.n
        temp.close(purge=True)

        if model_type == "flatten":
            model_kwargs["n_actions"] = n_actions
            model_kwargs["n_obs"] = state.shape[0]
            model_class = full_state_models.ActorCriticFlattenedBattleModel
        else:
            model_kwargs["n_actions"] = n_actions
            model_kwargs["state_length_dict"] = state_length_dict
            model_kwargs["max_values_dict"] = max_values_dict
            model_class = full_state_models.ActorCriticBattleModel
    model = setup_and_load_model(model_class, model_kwargs, model_path)
    return model


if __name__ == "__main__":
    moveset_db_path = "meta_discovery/data/moveset_database.joblib"
    meta_discovery_db_path = "meta_discovery/data/test.joblib"
    tier_list_path = "meta_discovery/data/tier_data.joblib"

    # Total num. battles to simulate
    num_battles_to_simulate = 20000
    # Num. battles between generating new teams
    team_generation_interval = 1000
    # Num. teams generated
    num_teams_to_generate = 2500

    # Team generation parameters
    use_pickrates = True
    use_popularity_score = True
    use_type_score = True
    use_meta_type_score = True
    use_bst_score = True
    ban_lc = True

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
    banlist_pokemon_exclusions = []
    banlist_moveset_exclusions = []

    # Number of battles to run simultaneously
    max_concurrent_battles = 25
    # Set random seed for reproducible results
    random_seed = 42

    # Setup player information - Determines what kind of agent we have
    player1_model_type = "simple"  # full, flatten, simple, heuristic
    player2_model_type = "simple"  # full, flatten, simple, heuristic

    player1_class = get_player_class(player1_model_type)
    player2_class = get_player_class(player2_model_type)

    player1_config = PlayerConfiguration(f"{server_port}_BattleAgent1", None)
    player2_config = PlayerConfiguration(f"{server_port}_BattleAgent2", None)

    player1_model_kwargs = {}
    # FullState specific model kwargs
    if player1_model_type == "full":
        player1_model_kwargs["pokemon_embedding_dim"] = 128
        player1_model_kwargs["team_embedding_dim"] = 128

    player2_model_kwargs = {}
    # FullState specific model kwargs
    if player2_model_type == "full":
        player2_model_kwargs["pokemon_embedding_dim"] = 128
        player2_model_kwargs["team_embedding_dim"] = 128

    player1_model_dir = "outputs/Simple_PPO_SelfPlay_v1.0/"
    player1_model_name = "model_2047627.pt"
    player1_model_path = os.path.join(player1_model_dir, player1_model_name)
    player1_lookup_path = None

    player2_model_dir = "outputs/Simple_PPO_SelfPlay_v1.0/"
    player2_model_name = "model_2047627.pt"
    player2_model_path = os.path.join(player2_model_dir, player2_model_name)
    player2_lookup_path = None

    # Setup player (battle agent) arguments
    player1_kwargs = {}
    player2_kwargs = {}

    # Device is a kwarg only if this is a model-based agent
    if player1_model_type != "heuristic":
        player1_kwargs["device"] = device

    if player2_model_type != "heuristic":
        player2_kwargs["device"] = device

    # FullState specific player kwargs
    if player1_model_type in ["flatten", "full"]:
        player1_lookup_path = os.path.join(
            player1_model_dir, "player_lookup_dicts.joblib"
        )
        player1_kwargs["lookup_filename"] = player1_lookup_path

    # FullState specific player kwargs
    if player2_model_type in ["flatten", "full"]:
        player2_lookup_path = os.path.join(
            player2_model_dir, "player_lookup_dicts.joblib"
        )
        player2_kwargs["lookup_filename"] = player2_lookup_path

    # Use random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    _ = torch.manual_seed(random_seed)

    # Create models if not using a heuristic agent
    if player1_model_type != "heuristic":
        player1_model = make_model(
            player1_model_type,
            player1_model_kwargs,
            player1_model_path,
            player1_lookup_path,
        )
        player1_kwargs["model"] = player1_model

    if player2_model_type != "heuristic":
        player2_model = make_model(
            player2_model_type,
            player2_model_kwargs,
            player2_model_path,
            player2_lookup_path,
        )
        player2_kwargs["model"] = player2_model

    # Special case for zygarde we need to unban a pokemon + ability
    if "zygarde" in banlist_pokemon_exclusions:
        banlist_moveset_exclusions.append("powerconstruct")

    # Setup banlist
    print("---" * 40)
    print(f"Tier Selected: {playable_tier}")
    ban_list = utils.get_ban_list(
        playable_tier, tier_list_path, banlist_pokemon_exclusions, ban_lc=ban_lc
    )
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
        epsilon=exploration_control,
        moveset_database=moveset_database,
        all_pokemon=all_pokemon,
        ban_list=ban_list,
        use_pickrates=use_pickrates,
        use_popularity_score=use_popularity_score,
        use_type_score=use_type_score,
        use_meta_type_score=use_meta_type_score,
        use_bst_score=use_bst_score,
    )

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
        **player2_kwargs,
    )

    # Meta Discovery Loop Starts Here
    print("---" * 30)
    start_time = time.time()
    for i in range(num_battles_to_simulate // team_generation_interval):
        print(f"Epoch {i+1}")

        # Generate new teams
        print("Generating New Teams.")
        start = time.time()
        team_builder.generate_teams(meta_discovery_database, num_teams_to_generate)
        end = time.time()
        print(f"Time Taken: {end - start}s")

        # Make agents use the generated teams
        player1._team = team_builder
        player2._team = team_builder

        # Play battles
        print("Battling.")
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

        # Update overall battle statistics
        print("Updating.")
        start = time.time()
        meta_discovery_database.update_battle_statistics(
            player1,
            player2,
            team_generation_interval,
        )
        end = time.time()
        print(f"Time Taken: {end - start}s")

        # Reset trackers so we don't count battles twice
        player1.reset_battles()
        player2.reset_battles()

        # Save DB
        meta_discovery_database.save(meta_discovery_db_path)
    end_time = time.time()
    print(f"Simulation Time Taken: {end_time - start_time:.4f}")

    # Meta Discovery
    print("Fin.")
