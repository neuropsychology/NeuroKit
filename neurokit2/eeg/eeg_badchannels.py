# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from ..signal import signal_zerocrossings
from ..stats import hdi, mad, standardize


def eeg_badchannels(eeg, bad_threshold=0.5, distance_threshold=0.99, show=False):
    """**Find bad channels**

    Find bad channels among the EEG channels.

    Parameters
    ----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    bad_threshold : float
        The proportion of indices (for instance, the mean, the SD, the skewness, the kurtosis, etc.)
        on which an observation is considered an outlier to be considered as bad. The default, 0.5,
        means that a channel must score as an outlier on half or more of the indices.
    distance_threshold : float
        The quantile that defines the absolute distance from the mean, i.e., the z-score for a
        value of a variable to be considered an outlier. For instance, .975 becomes
        ``scipy.stats.norm.ppf(.975) ~= 1.96``. The default value (.99) means that all observations
        beyond 2.33 SD from the mean will be classified as outliers.
    show : bool
        Visualize individual EEG channels with highlighted bad channels. Defaults to False

    Returns
    -------
    list
        List of bad channel names
    DataFrame
        Information of each channel, such as standard deviation (SD), mean, median absolute
        deviation (MAD), skewness, kurtosis, amplitude, highest density intervals, number of zero
        crossings.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      eeg = nk.mne_data("filt-0-40_raw")
      bads, info = nk.eeg_badchannels(eeg, distance_threshold=0.95, show=False)

    """
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        try:
            import mne
        except ImportError as e:
            raise ImportError(
                "NeuroKit error: eeg_badchannels(): the 'mne' module is required for this function"
                " to run. Please install it first (`pip install mne`).",
            ) from e
        selection = mne.pick_types(eeg.info, eeg=True)
        ch_names = np.array(eeg.ch_names)[selection]
        eeg, _ = eeg[selection]
    else:
        ch_names = np.arange(len(eeg))

    results = []
    for i in range(len(eeg)):
        channel = eeg[i, :]

        hdi_values = hdi(channel, ci=0.90)
        info = {
            "Channel": [i],
            "SD": [np.nanstd(channel, ddof=1)],
            "Mean": [np.nanmean(channel)],
            "MAD": [mad(channel)],
            "Median": [np.nanmedian(channel)],
            "Skewness": [scipy.stats.skew(channel)],
            "Kurtosis": [scipy.stats.kurtosis(channel)],
            "Amplitude": [np.max(channel) - np.min(channel)],
            "CI_low": [hdi_values[0]],
            "CI_high": [hdi_values[1]],
            "n_ZeroCrossings": [len(signal_zerocrossings(channel - np.nanmean(channel)))],
        }
        results.append(pd.DataFrame(info))
    results = pd.concat(results, axis=0)
    results = results.set_index("Channel")

    z = standardize(results)
    results["Bad"] = (z.abs() > scipy.stats.norm.ppf(distance_threshold)).sum(axis=1) / len(
        results.columns
    )
    bads = ch_names[np.where(results["Bad"] >= bad_threshold)[0]]

    if show:
        _plot_eeg_badchannels(eeg, bads, ch_names)

    return list(bads), results


def _plot_eeg_badchannels(eeg, bads, ch_names):

    # Prepare plot
    fig, ax = plt.subplots()
    fig.suptitle("Individual EEG channels")
    ax.set_ylabel("Voltage (V)")
    ax.set_xlabel("Samples")

    bads_list = []
    for bad in bads:
        channel_index = np.where(ch_names == bad)[0]
        bads_list.append(channel_index[0])

    # Prepare colors for plotting
    colors_good = plt.cm.Greys(np.linspace(0, 1, len(eeg)))
    colors_bad = plt.cm.autumn(np.linspace(0, 1, len(bads)))

    # Plot good channels
    for i in range(len(eeg)):
        if i not in bads_list:
            channel = eeg[i, :]
            ax.plot(np.arange(1, len(channel) + 1), channel, c=colors_good[i])

    # Plot bad channels
    for i, bad in enumerate(bads_list):
        channel = eeg[bad, :]
        ax.plot(np.arange(1, len(channel) + 1), channel, c=colors_bad[i], label=ch_names[i])

    ax.legend(loc="upper right")

    return fig
