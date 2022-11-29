import numpy as np
import pandas as pd
from poke_env.utils import to_id_str
from scipy.stats import spearmanr

def calculate_edit_distance(previous_meta, current_meta, top_n):
    superset = sorted(
        list(
            set(
                previous_meta["pokemon"][:top_n].to_list()
                + current_meta["pokemon"][:top_n].to_list()
            )
        )
    )
    all_edit_distances = {}
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
            all_edit_distances[pokemon] = edit_distance
    return all_edit_distances


if __name__ == "__main__":
    top_n = 40  # No. Pokemon to consider
    versions = ["1", "1.1"]#, "1.1", "2", "2.1", "baseline"]
    tier = "pu"
    pokemon = "vanilluxe"

    smogon_preban_data = f"usage_stats/{pokemon}_preban.csv"
    smogon_postban_data = f"usage_stats/{pokemon}_postban.csv"

    # Set random seed for reproducible results
    random_seed = 42

    # Use random seeds
    np.random.seed(random_seed)

    for version in versions:
        if version == "baseline":
            our_postban_data = f"meta_discovery/data/{pokemon}_postban_baseline.csv"
        elif pokemon == "kyurem":
            if version == "2":
                our_postban_data = f"meta_discovery/data/smogon_{tier}_{pokemon}_ablation_v10/smogon_{tier}_{pokemon}_ablation_postban_v10.csv"
            elif version == "2.1":
                our_postban_data = f"meta_discovery/data/smogon_{tier}_{pokemon}_ablation_v10.1/smogon_{tier}_{pokemon}_ablation_postban_v10.1.csv"
            else:
                our_postban_data = f"meta_discovery/data/smogon_{tier}_{pokemon}_final_v{version}/smogon_{tier}_{pokemon}_final_v{version}.csv"
        else:
            our_postban_data = f"meta_discovery/data/smogon_{tier}_{pokemon}_v{version}/smogon_{tier}_{pokemon}_v{version}.csv"

        # Load data
        smogon_preban = pd.read_csv(smogon_preban_data)
        smogon_postban = pd.read_csv(smogon_postban_data)
        our_postban = pd.read_csv(our_postban_data)

        # Preprocess to ensure they use the same name format
        smogon_preban["pokemon"] = smogon_preban["pokemon"].apply(to_id_str)
        smogon_postban["pokemon"] = smogon_postban["pokemon"].apply(to_id_str)
        our_postban["pokemon"] = our_postban["pokemon"].apply(to_id_str)

        smogon_edit_distances = calculate_edit_distance(smogon_preban, smogon_postban, top_n)
        our_edit_distances = calculate_edit_distance(smogon_preban, our_postban, top_n)

        df = pd.DataFrame([smogon_edit_distances, our_edit_distances]).T.rename(columns={0: "True", 1: "Ours"})
        df.dropna(inplace=True)
        print(f"Version:{version}", spearmanr(df["True"], df["Ours"], nan_policy="raise"))
        df.to_csv(f"{pokemon}_simple.csv")