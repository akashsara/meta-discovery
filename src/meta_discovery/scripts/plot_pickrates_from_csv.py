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
    csv_path = "usage_stats/gen8ou-0-2022-08.csv"

    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set random seed for reproducible results
    random_seed = 42

    # Use random seeds
    np.random.seed(random_seed)

    df = pd.read_csv(csv_path)
    pickrates = sorted(
        df["raw_usage_percentage"].apply(lambda x: float(x.replace("%", "")))
    )
    pickrates = np.array(pickrates)

    # Method 1 - Raw Winrates
    data = pickrates.copy()
    plot_and_save(data, "Prob(winrates)", output_dir)

    # Method 2 - 2*Winrates
    data = pickrates.copy() * 2
    plot_and_save(data, "Prob(winrates_x2)", output_dir)

    # Method 3 - 10*Winrates
    data = pickrates.copy() * 10
    plot_and_save(data, "Prob(winrates_x10)", output_dir)

    # Method 4 = Winrates^2
    data = pickrates.copy() ** 2
    plot_and_save(data, "Prob(winrates_^2)", output_dir)

    # Method 5 = Winrates^3
    data = pickrates.copy() ** 3
    plot_and_save(data, "Prob(winrates_^3)", output_dir)

    # Method 6 = Winrates^4
    data = pickrates.copy() ** 4
    plot_and_save(data, "Prob(winrates_^4)", output_dir)

    # Method 7 = Winrates^5
    data = pickrates.copy() ** 5
    plot_and_save(data, "Prob(winrates_^5)", output_dir)

    # Method 8 = Winrates^6
    data = pickrates.copy() ** 6
    plot_and_save(data, "Prob(winrates_^6)", output_dir)

    # Method 9 = Winrates-0.5
    data = pickrates.copy() - 0.5
    plot_and_save(data, "Prob(winrates_-0.5)", output_dir)

    # Method 10 = sigmoid(Winrates)
    data = sigmoid(pickrates.copy())
    plot_and_save(data, "Prob(sigmoid(winrates))", output_dir)

    # Method 10 = sigmoid(Winrates-0.5)
    data = sigmoid(pickrates.copy() - 0.5)
    plot_and_save(data, "Prob(sigmoid(winrates_-0.5))", output_dir)
