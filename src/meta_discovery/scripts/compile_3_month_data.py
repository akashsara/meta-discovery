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
    prior_month_data_path = ""
    month1_data_path = "meta_discovery/data/standard_ou_kyurem_v2/standard_ou_kyurem_preban_v2.joblib.month1.joblib"
    month2_data_path = "meta_discovery/data/standard_ou_kyurem_v2/standard_ou_kyurem_preban_v2.joblib.month2.joblib"
    month3_data_path = "meta_discovery/data/standard_ou_kyurem_v2/standard_ou_kyurem_preban_v2.joblib.month3.joblib"
    output_file = (
        "meta_discovery/data/standard_ou_kyurem_v2/standard_ou_kyurem_preban_v2.csv"
    )

    # Set random seed for reproducible results
    random_seed = 42

    # Use random seeds
    np.random.seed(random_seed)

    # Load moveset DB
    moveset_database = joblib.load(moveset_db_path)

    print("---" * 30)
    print("Setting up Meta Discovery database.")

    if prior_month_data_path:
        print("Loading Prior Months Data")
        prior_months = MetaDiscoveryDatabase(moveset_database)
        prior_months.load(prior_month_data_path)
        print(f"Load complete. {prior_months.num_battles} Battles Complete")

    print("Loading Month 1")
    month1 = MetaDiscoveryDatabase(moveset_database)
    month1.load(month1_data_path)
    print(f"Load complete. {month1.num_battles} Battles Complete")

    print("Loading Month 2")
    month2 = MetaDiscoveryDatabase(moveset_database)
    month2.load(month2_data_path)
    print(f"Load complete. {month2.num_battles} Battles Complete")

    print("Loading Month 3")
    month3 = MetaDiscoveryDatabase(moveset_database)
    month3.load(month3_data_path)
    print(f"Load complete. {month3.num_battles} Battles Complete")
    print("---" * 30)

    # Fix stats so we have monthly data
    month3 = fix_stats(month3, month2)  # Month 3 = Month 3 - Month 2
    month2 = fix_stats(month2, month1)  # Month 2 = Month 2 - Month 1
    if prior_month_data_path:
        month1 = fix_stats(month1, prior_months)

    # Calculate 3 Month Average
    sum_battles = month3.num_battles + month2.num_battles + month1.num_battles
    num_battles = sum_battles
    sum_picks = month3.picks + month2.picks + month1.picks
    sum_wins = month3.wins + month2.wins + month1.wins
    pickrates = sum_picks / (2 * sum_battles)

    # Store in dict
    output = []
    for key, pokemon in month3.key2pokemon.items():
        if sum_picks[key] == 0:
            continue
        output.append(
            {
                "pokemon": pokemon,
                "num_battles": num_battles,
                "picks": sum_picks[key],
                "wins": sum_wins[key],
                "average_pickrates": pickrates[key],
            }
        )

    # Convert to csv, sort, and save.
    df = pd.DataFrame(output).sort_values("average_pickrates", ascending=False)
    df.to_csv(output_file)
