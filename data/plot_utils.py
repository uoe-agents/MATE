import os
from pathlib import Path

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


plt.style.use("./prettyplots.mplstyle")
# avoid type-3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


ALPHA = 0.2
COLOURS = sns.color_palette("muted")


def setup_figure(num_colours=None):
    # colouring
    sns.set_style("whitegrid")
    if num_colours:
        colours = sns.color_palette("icefire", num_colours)
    else:
        colours = sns.color_palette("colorblind")

    # patterns, linestyles, and markers
    patterns = ["/", "\\", "o", ".", "|", "*", "+", "-", "O", "x"]
    markers = ["D", "o", "P", "X", "^", "*", "v", "+", "s", ".", "x"]

    return colours, patterns, markers


def smooth_values(values, bin_size=2, min_vals=100):
    """
    Apply average smoothing with given bin_size
    :param values: list of Iterables to smooth
    :param bin_size: size of bins to average over
    :param min_vals: minimum number of (expected) values
    :return: list of smoothed Iterables
    """
    # average values together
    num_iterables = len(values)
    num_values = min([len(vals) for vals in values] + [100])
    num_bins = num_values // bin_size + 1 * int((num_values % bin_size) != 0)
    values_smoothed = [[[] for _ in range(num_bins)] for _ in range(num_iterables)]
    for i, vals in enumerate(zip(*values)):
        new_idx = i // bin_size
        if new_idx >= num_bins:
            break
        for vals_smoothed, val in zip(values_smoothed, vals):
            vals_smoothed[new_idx].append(val)

    values_smoothed = [np.array(vals).mean(1) for vals in values_smoothed]
    return values_smoothed


def decorate_axis(ax, wrect=10, hrect=10, labelsize="large", scale_ticks=True):
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    if scale_ticks:
        ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
    else:
        ax.tick_params(labelsize=labelsize)
    # Pablos' comment
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))


def annotate_and_decorate_axis(
    ax,
    xlabel,
    ylabel,
    ylabel_colour=None,
    wrect=10,
    hrect=10,
    grid_alpha=1,
    labelsize="large",
    ticklabelsize="large",
    scale_ticks=True,
):
    ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel_colour is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize, color=ylabel_colour)
    else:
        ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.grid(True, alpha=grid_alpha)
    decorate_axis(ax, wrect, hrect, ticklabelsize, scale_ticks)


def plot_results(
    result_dict,
    x_label,
    y_label,
    y_lims=None,
    title=None,
    plot_legend=True,
    legend_loc="best",
    legend_args={},
    smooth=False,
    save_path=None,
):
    """
    Plot given metrics for all set of runs within the dictionary

    :param group_dict: dictionary mapping algorithm name to aggregate data with (x_values, aggregate_scores, CIs)
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param y_lims: pair of (min,max) value for y-axis or None for no limits
    :param title: title for plot if None
    :param plot_legend: boolean if legend should be plotted
    :param legend_loc: location of legend for matplotlib plots
    :param legend_args: further legend arguments
    :param smooth: smooth values by bins to cut number of values in half
    :param save_path: if not None save figure as plot under this file name
    """
    fig = plt.figure()
    for colour, (alg_name, values) in zip(COLOURS, result_dict.items()):
        assert len(values) == 3
        x_values, y_aggregates, y_confidence_intervals = values
        y_confidence_lower = y_confidence_intervals[0]
        y_confidence_upper = y_confidence_intervals[1]

        # average values together
        if smooth:
            x_values, y_aggregates, y_confidence_lower, y_confidence_upper = (
                smooth_values(
                    [x_values, y_aggregates, y_confidence_lower, y_confidence_upper]
                )
            )

        plt.plot(x_values, y_aggregates, label=alg_name, color=colour)
        plt.fill_between(
            x_values,
            y_confidence_lower,
            y_confidence_upper,
            alpha=ALPHA,
            color=colour,
        )

    if y_lims is not None:
        plt.ylim(*y_lims)

    ax = fig.axes[0]
    annotate_and_decorate_axis(
        ax, x_label, y_label, labelsize="x-large", ticklabelsize="x-large"
    )
    if title is not None:
        plt.title(title)

    if plot_legend:
        plt.legend(fontsize="medium", loc=legend_loc)

    if save_path is not None:
        path_dirs = Path(save_path).parent.absolute()
        os.makedirs(path_dirs, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")


def generate_fake_legend(
    labels,
    save_path="legend",
    alpha=1.0,
    bbox_anchor=(0.6, 1.1),
    vertical=True,
    reverse=True,
):
    # avoid type-3 fonts
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    fig = plt.figure()
    colours, _, _ = setup_figure()

    if not vertical and reverse:
        colours = list(reversed([colours[i] for i in range(len(labels))]))
        labels = list(reversed(labels))
    else:
        colours = colours[: len(labels)]
    fake_patches = [mpatches.Patch(color=colour, alpha=alpha) for colour in colours]
    fig.legend(
        fake_patches,
        labels,
        loc="lower center",
        fancybox=True,
        ncol=1 if vertical else len(labels),
        fontsize="x-large",
        bbox_to_anchor=bbox_anchor,
    )
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
