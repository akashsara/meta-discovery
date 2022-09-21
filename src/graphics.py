import matplotlib.pyplot as plt


def plot_and_save_loss(values, xlabel, ylabel, output_path):
    plt.figure(dpi=300)
    ax = plt.subplot()
    plt.plot(values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pickrates(data, title, output_path):
    plt.figure(dpi=300)
    x = list(range(len(data)))
    plt.bar(x, data)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
