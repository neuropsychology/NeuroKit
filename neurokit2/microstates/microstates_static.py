# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from ..misc import find_groups


def microstates_static(microstates, sampling_rate=1000, show=False):
    """Static properties of microstates

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> microstates = [0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0]
    >>> nk.microstates_static(microstates, sampling_rate=100)
    """
    out = {}

    out, lifetimes = _microstates_prevalence(microstates, out=out)
    out, durations, types = _microstates_duration(microstates, sampling_rate=sampling_rate, out=out)

    if show is True:
        fig, axes = plt.subplots(nrows=2)
        axes[0] = _microstates_prevalence_plot(microstates, lifetimes, out)
        axes[1] = _microstates_duration_plot(durations, types)

    return out



# =============================================================================
# Duration
# =============================================================================
def _microstates_duration(microstates, sampling_rate=1000, out=None):
    """
    Examples
    --------
    >>> import numpy as np
    >>> microstates = np.random.randint(0, 5, 1000)
    >>> _microstates_basic(microstates, sampling_rate=100)
    """
    n = len(microstates)
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
        out["Microstate_" + str(s) + "_DurationMean"] = np.mean(durations[types == s])
        out["Microstate_" + str(s) + "_DurationMedian"] = np.median(durations[types == s])
    out["Microstate_Average_DurationMean"] = np.mean(durations)
    out["Microstate_Average_DurationMedian"] = np.median(durations)



    return out, durations, types



def _microstates_duration_plot(durations, types):
    """
    """
    # Make data for violin
    states = np.unique(types)
    data = []
    for s in states:
        data.append(durations[types == s])

    # Plot
    fig, ax = plt.subplots(ncols=1)
    parts = ax.violinplot(data, vert=False, showmedians=True, showextrema=False)
    for component in parts:
        if isinstance(parts[component], list):
            for part in parts[component]:
                part.set_facecolor('#FF5722')
                part.set_edgecolor('white')
        else:
            parts[component].set_edgecolor('black')
    plt.xlabel("Duration (s)")
    ax.set_title("Duration")



# =============================================================================
# Prevalence
# =============================================================================
def _microstates_prevalence(microstates, out=None):
    """
    Examples
    --------
    >>> import numpy as np
    >>> microstates = np.random.randint(0, 5, 1000)
    >>> _microstates_prevalence(microstates, sampling_rate=100)
    """
    n = len(microstates)
    states = np.unique(microstates)

    if out is None:
        out = {}

    # Average proportion
    for s in states:
        out["Microstate_" + str(s) + "_Proportion"] = np.sum(microstates == s) / n

    # Leftime distribution
    out, lifetimes = _microstates_lifetime(microstates, out=out)

    return out, lifetimes


def _microstates_prevalence_plot(microstates, lifetimes, out):
    """
    """
    states = np.unique(microstates)
    fig, axes = plt.subplots(ncols=2)
    for s in states:
        axes[0].bar(s, out["Microstate_" + str(s) + "_Proportion"])
        axes[1].plot(lifetimes[s], label=str(s))
    plt.legend()
    axes[0].set_title("Proportion")
    axes[1].set_title("Lifetime Distribution")


# Lifetime distribution
# ------------------------
def _microstates_lifetime(microstates, out=None):
    """Based on https://github.com/Frederic-vW/eeg_microstates

    Compute the lifetime distributions for each symbol in a symbolic sequence X with ns symbols.

    Examples
    --------
    >>> import numpy as np
    >>> microstates = np.random.randint(0, 5, 1000)
    >>> lifetimes, auc = _microstates_lifetimes(microstates, show=True)
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
        out["Microstate_" + str(s) + "_LifetimeDistribution"] = np.trapz(lifetimes[s])

    return out, lifetimes


