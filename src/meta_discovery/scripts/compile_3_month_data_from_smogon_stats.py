import os
import sys

import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate
from poke_env.data import to_id_str

sys.path.append("./")
from meta_discovery import utils
from meta_discovery.meta_discovery import MetaDiscoveryDatabase


def get_value_from_df(df, pokemon, column):
    out = df[df["pokemon"] == pokemon]
    # Some Pokemon may not see usage in every month
    if out.shape[0] == 0:
        return 0
    return out[column].iloc[0]


if __name__ == "__main__":
    # Kyurem: Banned in 12-2021. Take the 3 months before and after.
    # Spectrier: Banned in 02-2021. Take the 3 months before and after.
    # Urshifu: Banned in 01-2021. Take the 3 months before and after.
    month1_data_path = "usage_stats/gen8ou-1695-2020-10.csv"
    month2_data_path = "usage_stats/gen8ou-1695-2020-11.csv"
    month3_data_path = "usage_stats/gen8ou-1695-2020-12.csv"
    output_file = "usage_stats/urshifu_preban.csv"

    # Set random seed for reproducible results
    random_seed = 42

    # Use random seeds
    np.random.seed(random_seed)

    # Load CSVs
    month1 = pd.read_csv(
        month1_data_path,
        usecols=[
            "pokemon",
            "weighted_usage_percentage",
            "raw_usage",
            "total_battles",
            "weightage",
        ],
    )
    month2 = pd.read_csv(
        month2_data_path,
        usecols=[
            "pokemon",
            "weighted_usage_percentage",
            "raw_usage",
            "total_battles",
            "weightage",
        ],
    )
    month3 = pd.read_csv(
        month3_data_path,
        usecols=[
            "pokemon",
            "weighted_usage_percentage",
            "raw_usage",
            "total_battles",
            "weightage",
        ],
    )

    # Apply to_id_str for naming consistency
    month1["pokemon"] = month1["pokemon"].apply(to_id_str)
    month2["pokemon"] = month2["pokemon"].apply(to_id_str)
    month3["pokemon"] = month3["pokemon"].apply(to_id_str)

    # Convert strings to percentages
    month1["weighted_usage_percentage"] = month1["weighted_usage_percentage"].apply(
        lambda x: float(x.replace("%", "")) / 100
    )
    month2["weighted_usage_percentage"] = month2["weighted_usage_percentage"].apply(
        lambda x: float(x.replace("%", "")) / 100
    )
    month3["weighted_usage_percentage"] = month3["weighted_usage_percentage"].apply(
        lambda x: float(x.replace("%", "")) / 100
    )

    # Compile all the Pokemon used through these months
    all_pokemon = set(month1["pokemon"].to_list() + month2["pokemon"].to_list() + month3["pokemon"].to_list())
    # Get average num. battles (and total!) and average. weightage
    sum_battles = (
        month1["total_battles"].iloc[0]
        + month2["total_battles"].iloc[0]
        + month3["total_battles"].iloc[0]
    )
    num_battles = sum_battles // 3
    weightage = (
        month1["weightage"].iloc[0]
        + month2["weightage"].iloc[0]
        + month3["weightage"].iloc[0]
    ) / 3

    output = []
    for pokemon in all_pokemon:
        sum_picks = (
            get_value_from_df(month1, pokemon, "raw_usage")
            + get_value_from_df(month2, pokemon, "raw_usage")
            + get_value_from_df(month3, pokemon, "raw_usage")
        )
        # Picks = average picks
        picks = sum_picks // 3
        # Pickrates = average pickrates
        pickrates = sum_picks / (2 * (sum_battles))
        # Calculating this just in case
        weighted_pickrates = (
            get_value_from_df(month1, pokemon, "weighted_usage_percentage")
            + get_value_from_df(month2, pokemon, "weighted_usage_percentage")
            + get_value_from_df(month3, pokemon, "weighted_usage_percentage")
        ) / 3

        output.append(
            {
                "pokemon": pokemon,
                "num_battles": num_battles,
                "weightage": weightage,
                "picks": picks,
                "pickrates": pickrates,
                "weighted_pickrates": weighted_pickrates,
            }
        )

    df = pd.DataFrame(output).sort_values("pickrates", ascending=False)
    df.to_csv(output_file)
