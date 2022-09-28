import numpy as np
import pandas as pd


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
            edit_distance = previous - current
            all_edit_distances.append(edit_distance)
        else:
            print(
                pokemon,
                pokemon in previous_meta["pokemon"].values,
                pokemon in current_meta["pokemon"].values,
            )
    return all_edit_distances


def calculate_overlap(baseline, generated, top_n):
    generated = set(generated["pokemon"][:top_n].to_list())
    baseline = baseline["pokemon"][:top_n].to_list()
    return sum([1 if pokemon in generated else 0 for pokemon in baseline]) / top_n


if __name__ == "__main__":
    smogon_preban_data = "usage_stats/spectrier_preban.csv"
    smogon_postban_data = "usage_stats/spectrier_postban.csv"
    our_preban_data = (
        "meta_discovery/data/standard_ou_spectrier/standard_ou_spectrier_preban.csv"
    )
    our_postban_data = (
        "meta_discovery/data/standard_ou_spectrier/standard_ou_spectrier_postban.csv"
    )
    top_n = 40  # No. Pokemon to consider

    # Set random seed for reproducible results
    random_seed = 42

    # Use random seeds
    np.random.seed(random_seed)

    smogon_preban = pd.read_csv(smogon_preban_data)
    smogon_postban = pd.read_csv(smogon_postban_data)
    our_preban = pd.read_csv(our_preban_data)
    our_postban = pd.read_csv(our_postban_data)

    # Version 1: Find Mean Edit Distance
    print("===" * 10)
    smogon_edit_distances = calculate_edit_distance(
        smogon_preban, smogon_postban, top_n
    )
    print("===" * 10)
    our_edit_distances = calculate_edit_distance(our_preban, our_postban, top_n)
    print("---" * 10)
    print(
        f"Smogon:\nMean Edit Distance: {np.mean(smogon_edit_distances):.4f}\nMedian Edit Distance: {np.median(smogon_edit_distances)}\nTotal Pokemon: {len(smogon_edit_distances)}"
    )
    print("---" * 10)
    print(
        f"Our:\nMean Edit Distance: {np.mean(our_edit_distances):.4f}\nMedian Edit Distance: {np.median(our_edit_distances):.4f}\nTotal Pokemon: {len(our_edit_distances)}"
    )

    # Version 2: Find Raw Overlap
    preban_overlap = calculate_overlap(smogon_preban, our_preban, top_n)
    postban_overlap = calculate_overlap(smogon_postban, our_postban, top_n)
    print("###" * 10)
    print(f"Pre-Ban Overlap: {preban_overlap*100:.4f}%.\nPost-Ban Overlap: {postban_overlap*100:.4f}%")
