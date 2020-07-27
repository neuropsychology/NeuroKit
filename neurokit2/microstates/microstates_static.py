# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.gridspec
from matplotlib import pyplot as plt
from ..misc import find_groups, as_vector


def microstates_static(microstates, sampling_rate=1000, show=False):
    """Static properties of microstates.

    The duration of each microstate is also referred to as the Ratio of Time Covered (RTT) in
    some microstates publications.

    Parameters
    ----------
    microstates : np.ndarray
        The topographic maps of the found unique microstates which has a shape of n_channels x n_states,
        generated from ``nk.microstates_segment()``.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Defaults to 1000.
    show : bool
        Returns a plot of microstate duration, proportion, and lifetime distribution if True.

    Returns
    -------
    DataFrame
        Values of microstates proportion, lifetime distribution and duration (median, mean, and their averages).

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> microstates = [0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0]
    >>> nk.microstates_static(microstates, sampling_rate=100)  #doctest: +ELLIPSIS
      Microstate_0_Proportion  ...  Microstate_Average_DurationMedian
    0 ...                      ...  ...

    [1 rows x 14 columns]

    """
    out = {}
    microstates = as_vector(microstates)

    out, lifetimes = _microstates_prevalence(microstates, out=out)
    out, durations, types = _microstates_duration(microstates, sampling_rate=sampling_rate, out=out)

    if show is True:
        fig = plt.figure(constrained_layout=False)
        spec = matplotlib.gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[1, 1], width_ratios=[1, 1])

        ax0 = fig.add_subplot(spec[1, :])
        ax1 = fig.add_subplot(spec[0, :-1])
        ax2 = fig.add_subplot(spec[0, 1])

        _microstates_duration_plot(durations, types, ax=ax0)
        _microstates_prevalence_plot(microstates, lifetimes, out, ax_prop=ax1, ax_distrib=ax2)

    df = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("Microstate_")

    return df



# =============================================================================
# Duration
# =============================================================================
def _microstates_duration(microstates, sampling_rate=1000, out=None):
    states = np.unique(microstates)

    if out is None:
        out = {}

    # Find durations of each state
    groups = find_groups(microstates)
    # Initialize empty containers for duration and type
    durations = np.full(len(groups), np.nan)
    types = np.full(len(groups), np.nan)
    for i, group in enumerate(groups):
        types[i] = group[0]
        durations[i] = len(group) / sampling_rate

    # Average duration
    for s in states:
        out[str(s) + "_DurationMean"] = np.mean(durations[types == s])
        out[str(s) + "_DurationMedian"] = np.median(durations[types == s])
    out["Average_DurationMean"] = np.mean(durations)
    out["Average_DurationMedian"] = np.median(durations)

    return out, durations, types


def _microstates_duration_plot(durations, types, ax=None):

    # Make data for violin
    states = np.unique(types)
    data = []
    for s in states:
        data.append(durations[types == s])

    # Plot
    if ax is None:
        fig, ax = plt.subplots(ncols=1)
    else:
        fig = None

    parts = ax.violinplot(data, vert=False, showmedians=True, showextrema=False)
    for component in parts:
        if isinstance(parts[component], list):
            for part in parts[component]:
                part.set_facecolor('#FF5722')
                part.set_edgecolor('white')
        else:
            parts[component].set_edgecolor('black')
    ax.set_xlabel("Duration (s)")
    ax.set_title("Duration")

    return fig


# =============================================================================
# Prevalence
# =============================================================================
def _microstates_prevalence(microstates, out=None):
    n = len(microstates)
    states = np.unique(microstates)

    if out is None:
        out = {}

    # Average proportion
    for s in states:
        out[str(s) + "_Proportion"] = np.sum(microstates == s) / n

    # Leftime distribution
    out, lifetimes = _microstates_lifetime(microstates, out=out)

    return out, lifetimes


def _microstates_prevalence_plot(microstates, lifetimes, out, ax_prop=None, ax_distrib=None):
    states = np.unique(microstates)

    # Plot
    if ax_prop is None and ax_distrib is None:
        fig, axes = plt.subplots(ncols=2)
        ax_prop = axes[0]
        ax_distrib = axes[1]
    else:
        fig = None

    for s in states:
        ax_prop.bar(s, out[str(s) + "_Proportion"])
        ax_distrib.plot(lifetimes[s], label=str(s))

    plt.legend()
    ax_prop.set_title("Proportion")
    ax_distrib.set_title("Lifetime Distribution")

    return fig


# Lifetime distribution
# ------------------------
def _microstates_lifetime(microstates, out=None):
    """Based on https://github.com/Frederic-vW/eeg_microstates

    Compute the lifetime distributions for each symbol in a symbolic sequence X with ns symbols.
    """
    n = len(microstates)
    states = np.unique(microstates)

    tau_dict = {s: [] for s in states}
    s = microstates[0]  # current symbol
    tau = 1.0  # current lifetime
    for i in range(n):
        if microstates[i] == s:
            tau += 1.0
        else:
            tau_dict[s].append(tau)
            s = microstates[i]
            tau = 1.0
    tau_dict[s].append(tau)  # last state

    # Initialize empty distributions with max lifetime for each symbol
    lifetimes = {}
    for s in states:
        lifetimes[s] = np.zeros(int(np.max(tau_dict[s])))

    # Lifetime distributions
    for s in states:
        for j in range(len(tau_dict[s])):
            tau = tau_dict[s][j]
            lifetimes[s][int(tau)-1] += 1.0

    # Get Area under curve (AUCs)
    if out is None:
        out = {}
    for s in states:
        out[str(s) + "_LifetimeDistribution"] = np.trapz(lifetimes[s])

    return out, lifetimes
