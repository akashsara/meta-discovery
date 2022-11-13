import numpy as np
import pandas as pd
from poke_env.utils import to_id_str

def calculate_edit_distance(previous_meta, current_meta, top_n):
    superset = sorted(
        list(
            set(
                previous_meta["pokemon"][:top_n].to_list()
                + current_meta["pokemon"][:top_n].to_list()
            )
        )
    )
    all_edit_distances = []
    for pokemon in superset:
        if (
            pokemon in previous_meta["pokemon"].values
            and pokemon in current_meta["pokemon"].values
        ):
            previous = previous_meta[previous_meta["pokemon"] == pokemon].index.values[
                0
            ]
            current = current_meta[current_meta["pokemon"] == pokemon].index.values[0]
            edit_distance = np.absolute(previous - current)
            all_edit_distances.append(edit_distance)
        else:
            print(
                pokemon,
                pokemon in previous_meta["pokemon"].values,
                pokemon in current_meta["pokemon"].values,
            )
    return all_edit_distances


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
    print("Pokemon In Existing Meta & Not In Our Meta:")
    print(true_unique_meta_pokemon)
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
    print("Pokemon Not In Existing Meta & In Our Meta:")
    print(our_unique_meta_pokemon)
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
    smogon_preban_data = "usage_stats/kyurem_preban.csv"
    smogon_postban_data = "usage_stats/kyurem_postban.csv"
    our_preban_data = "usage_stats/kyurem_preban.csv"
    our_postban_data = f"meta_discovery/data/kyurem_postban_baseline.csv"
    top_n = 40  # No. Pokemon to consider

    # Set random seed for reproducible results
    random_seed = 42

    # Use random seeds
    np.random.seed(random_seed)

    smogon_preban = pd.read_csv(smogon_preban_data)
    smogon_postban = pd.read_csv(smogon_postban_data)
    our_preban = pd.read_csv(our_preban_data)
    our_postban = pd.read_csv(our_postban_data)

    # Preprocess to ensure they use the same name format
    smogon_preban["pokemon"] = smogon_preban["pokemon"].apply(to_id_str)
    smogon_postban["pokemon"] = smogon_postban["pokemon"].apply(to_id_str)
    our_preban["pokemon"] = our_preban["pokemon"].apply(to_id_str)
    our_postban["pokemon"] = our_postban["pokemon"].apply(to_id_str)

    # Version 1: Find Mean Edit Distance
    print("===" * 10)
    smogon_edit_distances = calculate_edit_distance(
        smogon_preban, smogon_postban, top_n
    )
    print("===" * 10)
    our_edit_distances = calculate_edit_distance(our_preban, our_postban, top_n)
    print("---" * 10)
    print(
        f"Smogon:\nEdit Distances: {np.mean(smogon_edit_distances):.2f}/{np.median(smogon_edit_distances):.2f}\nTotal Pokemon: {len(smogon_edit_distances)}"
    )
    print("---" * 10)
    print(
        f"Our:\nEdit Distances: {np.mean(our_edit_distances):.2f}/{np.median(our_edit_distances):.2f}\nTotal Pokemon: {len(our_edit_distances)}"
    )

    print("###" * 20)
    # Version 2: Find Raw Overlap
    print("PREBAN:")
    preban_overlap = calculate_overlap(smogon_preban, our_preban, top_n)
    print("POSTBAN: ")
    postban_overlap = calculate_overlap(smogon_postban, our_postban, top_n)
    print(
        f"Pre-Ban Overlap: {preban_overlap*100:.4f}%.\nPost-Ban Overlap: {postban_overlap*100:.4f}%"
    )
