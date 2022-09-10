import os

import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate

import utils
from meta_discovery import MetaDiscoveryDatabase

if __name__ == "__main__":
    moveset_db_path = "meta_discovery/data/moveset_database.joblib"
    meta_discovery_db_path = "meta_discovery/data/meta_discovery_database.joblib"
    tier_list_path = "meta_discovery/data/tier_data.joblib"
    # Used to enforce species clause
    pokedex_json_path = "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json"
    top_n = 25  # No. Pokemon to consider
    metagame = "gen8ubers"
    # Set random seed for reproducible results
    random_seed = 42

    # Use random seeds
    np.random.seed(random_seed)

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
    print("Found existing database. Loading...")
    meta_discovery_database.load(meta_discovery_db_path)
    print(f"Load complete. {meta_discovery_database.num_battles} Battles Complete")
    print("---" * 30)

    # Print top N most picked Pokemon with their pickrates and winrates.
    top = []
    for pokemon in meta_discovery_database.pickrates.argsort()[::-1]:
        if meta_discovery_database.key2pokemon[pokemon] not in ban_list:
            top.append(pokemon)

    print(f"\n25 Most Popular Picks: ")
    data = []
    for idx in top[:top_n]:
        data.append(
            (
                meta_discovery_database.key2pokemon[idx],
                meta_discovery_database.picks[idx],
                f"{meta_discovery_database.pickrates[idx] * 100:.4f}%",
                meta_discovery_database.wins[idx],
                f"{meta_discovery_database.winrates[idx] * 100:.4f}%",
            )
        )
    print(
        tabulate(
            data,
            headers=["Name", "Picks", "Pickrate", "Wins", "Winrate"],
            tablefmt="orgtbl",
        )
    )

    # Print top N Win% Pokemon with their pickrates and winrates.
    top = []
    for pokemon in meta_discovery_database.winrates.argsort()[::-1]:
        if meta_discovery_database.key2pokemon[pokemon] not in ban_list:
            top.append(pokemon)
    print(f"\n25 Strongest Picks: ")
    data = []
    for idx in top[:top_n]:
        data.append(
            (
                meta_discovery_database.key2pokemon[idx],
                meta_discovery_database.picks[idx],
                f"{meta_discovery_database.pickrates[idx] * 100:.4f}%",
                meta_discovery_database.wins[idx],
                f"{meta_discovery_database.winrates[idx] * 100:.4f}%",
            )
        )
    print(
        tabulate(
            data,
            headers=["Name", "Picks", "Pickrate", "Wins", "Winrate"],
            tablefmt="orgtbl",
        )
    )
