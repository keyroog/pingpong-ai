import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(metrics_dict, rolling_window=50, title="Metrics Over Episodes", xlabel="Episodes", ylabel="Value", save_path=None):
    """
    Plots multiple metrics on the same graph.

    :param metrics_dict: Dictionary where keys are labels (e.g., "Left Rewards") and values are lists of metrics.
    :param rolling_window: Window size for calculating rolling average. Set to 0 to disable.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param save_path: If provided, saves the plot to this path.
    """
    plt.figure(figsize=(12, 6))

    for label, values in metrics_dict.items():
        plt.plot(values, label=label, alpha=0.6)
        if rolling_window > 1:
            rolling_avg = np.convolve(values, np.ones(rolling_window) / rolling_window, mode="valid")
            plt.plot(
                range(rolling_window - 1, len(values)),
                rolling_avg,
                label=f"{label} (Rolling Avg, {rolling_window})",
                linestyle="--"
            )

    # Add labels, title, and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()