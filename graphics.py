import matplotlib.pyplot as plt


def plot_and_save_loss(values, xlabel, ylabel, output_path):
    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot()
    plt.plot(values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
