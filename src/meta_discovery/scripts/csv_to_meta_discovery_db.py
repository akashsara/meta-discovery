import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append("./")
from meta_discovery import utils
from meta_discovery.meta_discovery import MetaDiscoveryDatabase


def fix_stats(month, prev_month):
    """
    Fixes the statistics in the file by removing the previous month stats.
    We don't do this while running the code since our team generation depends
    on the stats from the previous months.
    """
    month.picks = month.picks - prev_month.picks
    month.wins = month.wins - prev_month.wins
    month.num_battles = month.num_battles - prev_month.num_battles
    month.calc_winrates_pickrates()
    return month


if __name__ == "__main__":
    moveset_db_path = "meta_discovery/data/moveset_database.joblib"
    csv_path = "usage_stats/kyurem_preban.csv"
    output_file = "meta_discovery/data/standard_ou_kyurem_preban.joblib"

    # Set random seed for reproducible results
    random_seed = 42

    # Use random seeds
    np.random.seed(random_seed)

    # Load moveset DB
    moveset_database = joblib.load(moveset_db_path)
    database = MetaDiscoveryDatabase(moveset_database)

    df = pd.read_csv(csv_path)
    df_pokemon2key = {pokemon: key for key, pokemon in df.to_dict()['pokemon'].items()}

    # Store in dict
    invalid = []
    for pokemon, key in database.pokemon2key.items():
        if pokemon not in df_pokemon2key:
            invalid.append(pokemon)
            continue
        df_key = df_pokemon2key[pokemon]
        database.picks[key] = df.loc[df_key, "picks"]

    database.num_battles = database.picks.sum() // 12
    database.calc_winrates_pickrates()

    database.save(output_file)

    print("Illegal Pokemon:")
    print(invalid)