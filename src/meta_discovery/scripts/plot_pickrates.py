import os
import sys

import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.append("./")
import graphics
from meta_discovery import utils
from meta_discovery.meta_discovery import MetaDiscoveryDatabase

def weights2probabilities(weights: np.ndarray) -> np.ndarray:
    """
    np.choice requires probabilities not weights
    """
    summation = weights.sum()
    if summation:
        return weights / summation
    else:
        return (weights + 1) / weights.shape[0]

def plot_and_save(data, title, save_dir):
    data = sorted(weights2probabilities(data))
    output_dir = os.path.join(save_dir, f"{title}.png")
    graphics.plot_pickrates(data, title, output_dir)
    print(title)

def sigmoid(x):
    return 1 / (1 + np.exp(x))

if __name__ == "__main__":
    moveset_db_path = "meta_discovery/data/moveset_database.joblib"
    meta_discovery_database_path = (
        "meta_discovery/data/V2-epsilon-0.001/full_ou_700k_v2.joblib"
    )
    tier_list_path = "meta_discovery/data/tier_data.joblib"

    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Used to enforce species clause
    pokedex_json_path = "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env/data/pokedex.json"
    metagame = "gen8ou"
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
    database = MetaDiscoveryDatabase(moveset_database)
    database.load(meta_discovery_database_path)
    print(f"Load complete. {database.num_battles} Battles Complete")

    statistic = database.winrates

    # Method 1 - Raw Winrates
    data = statistic.copy()
    plot_and_save(data, "Prob(winrates)", output_dir)

    # Method 2 - 2*Winrates
    data = statistic.copy() * 2
    plot_and_save(data, "Prob(winrates_x2)", output_dir)

    # Method 3 - 10*Winrates
    data = statistic.copy() * 10
    plot_and_save(data, "Prob(winrates_x10)", output_dir)

    # Method 4 = Winrates^2
    data = statistic.copy() ** 2
    plot_and_save(data, "Prob(winrates_^2)", output_dir)

    # Method 5 = Winrates^3
    data = statistic.copy() ** 3
    plot_and_save(data, "Prob(winrates_^3)", output_dir)

    # Method 6 = Winrates^4
    data = statistic.copy() ** 4
    plot_and_save(data, "Prob(winrates_^4)", output_dir)

    # Method 7 = Winrates^5
    data = statistic.copy() ** 5
    plot_and_save(data, "Prob(winrates_^5)", output_dir)

    # Method 8 = Winrates^6
    data = statistic.copy() ** 6
    plot_and_save(data, "Prob(winrates_^6)", output_dir)

    # Method 9 = Winrates-0.5
    data = statistic.copy() - 0.5
    plot_and_save(data, "Prob(winrates_-0.5)", output_dir)

    # Method 10 = sigmoid(Winrates)
    data = sigmoid(statistic.copy())
    plot_and_save(data, "Prob(sigmoid(winrates))", output_dir)

    # Method 10 = sigmoid(Winrates-0.5)
    data = sigmoid(statistic.copy() - 0.5)
    plot_and_save(data, "Prob(sigmoid(winrates_-0.5))", output_dir)

    # Method 11 = Winrates^12
    data = statistic.copy() ** 12
    plot_and_save(data, "Prob(winrates_^12)", output_dir)

    # Method 11 = Winrates^9
    data = statistic.copy() ** 9
    plot_and_save(data, "Prob(winrates_^9)", output_dir)


