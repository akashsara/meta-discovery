"""
This script is for comparing only one meta to another.
So there are no bans or pre-ban vs post-ban stuff.
"""

import numpy as np
import pandas as pd


def calculate_overlap(baseline, generated, top_n):
    generated_position_lookup = {
        value: key for key, value in generated["pokemon"].items()
    }
    baseline_position_lookup = {
        value: key for key, value in baseline["pokemon"].items()
    }
    generated_meta = set(generated["pokemon"][:top_n].to_list())
    baseline_meta = set(baseline["pokemon"][:top_n].to_list())
    true_unique_meta_pokemon = [
        pokemon for pokemon in baseline_meta if pokemon not in generated_meta
    ]
    # For Pokemon in the existing meta but not in our meta, 
    # what is the distance in ranking from our meta?
    distances = [
        generated_position_lookup.get(pokemon, len(generated_position_lookup)) - top_n + 1
        for pokemon in true_unique_meta_pokemon
    ]
    d1 = np.mean(distances) if len(distances) > 0 else 0
    our_unique_meta_pokemon = [
        pokemon for pokemon in generated_meta if pokemon not in baseline_meta
    ]
    # For Pokemon in our meta but not in the existing meta, 
    # what is the distance in ranking from the existing existing?
    distances = [
        baseline_position_lookup.get(pokemon, len(baseline_position_lookup)) - top_n + 1
        for pokemon in our_unique_meta_pokemon
    ]
    d2 = np.mean(distances) if len(distances) > 0 else 0
    print(f"Average Distance from Meta: ({d1:.2f}, {d2:.2f})")
    return (top_n - len(true_unique_meta_pokemon)) / top_n


if __name__ == "__main__":
    true_meta = "usage_stats/current_meta.csv"
    version = "8.1"
    our_meta = f"meta_discovery/data/standard_ou_no_bans_v{version}/standard_ou_no_bans_v{version}.csv"
    top_n = 40  # No. Pokemon to consider

    true_meta = pd.read_csv(true_meta)
    our_meta = pd.read_csv(our_meta)

    postban_overlap = calculate_overlap(true_meta, our_meta, top_n)
    print(f"Overlap: {postban_overlap*100:.2f}%")
