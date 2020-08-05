# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats

from ..stats import standardize, mad, hdi
from ..signal import signal_zerocrossings


def eeg_badchannels(eeg, bad_threshold=0.5, distance_threshold=0.99):
    """Find bad channels.

    Parameters
    ----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    bad_threshold : float
        The proportion of indices (for instance, the mean, the SD, the skewness, the kurtosis, etc.)
        on which an observation is considered an outlier to be considered as bad. The default, 0.5,
        means that a channel must score as an outlier on half or more of the indices.
    distance_threshold : float
        The quantile that desfines the absolute distance from the mean, i.e., the z-score for a
        value of a variable to be considered an outlier. For instance, .975 becomes
        ``scipy.stats.norm.ppf(.975) ~= 1.96``. The default value (.99) means that all observations
        beyond 2.33 SD from the mean will be classified as outliers.

    Returns
    -------
    list
        List of bad channel names
    DataFrame
        Information of each channel, such as standard deviation (SD), mean, median absolute deviation (MAD),
        skewness, kurtosis, amplitude, highest density intervals, number of zero crossings.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw")
    >>> bads, info = nk.eeg_badchannels(eeg)

    """
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        try:
            import mne
        except ImportError:
            raise ImportError(
                "NeuroKit error: eeg_badchannels(): the 'mne' module is required for this function"
                " to run. Please install it first (`pip install mne`).",
            )
        selection = mne.pick_types(eeg.info, eeg=True)
        ch_names = np.array(eeg.ch_names)[selection]
        eeg, _ = eeg[selection]
    else:
        ch_names = np.arange(len(eeg))

    results = []
    for i in range(len(eeg)):
        channel = eeg[i, :]

        hdi_values = hdi(channel, ci=0.90)
        info = {"Channel": [i],
                "SD": [np.nanstd(channel, ddof=1)],
                "Mean": [np.nanmean(channel)],
                "MAD": [mad(channel)],
                "Median": [np.nanmedian(channel)],
                "Skewness": [scipy.stats.skew(channel)],
                "Kurtosis": [scipy.stats.kurtosis(channel)],
                "Amplitude": [np.max(channel) - np.min(channel)],
                "CI_low": [hdi_values[0]],
                "CI_high": [hdi_values[1]],
                "n_ZeroCrossings": [len(signal_zerocrossings(channel - np.nanmean(channel)))]}
        results.append(pd.DataFrame(info))
    results = pd.concat(results, axis=0)
    results = results.set_index("Channel")

    z = standardize(results)
    results["Bad"] = (z.abs() > scipy.stats.norm.ppf(distance_threshold)).sum(axis=1) / len(results.columns)
    bads = ch_names[np.where(results["Bad"] >= bad_threshold)[0]]

    return list(bads), results
