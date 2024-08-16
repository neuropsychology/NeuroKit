# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..epochs.eventrelated_utils import (
    _eventrelated_addinfo,
    _eventrelated_sanitizeinput,
    _eventrelated_sanitizeoutput,
)
from ..misc import NeuroKitWarning


def emg_eventrelated(epochs, silent=False):
    """**Event-related EMG Analysis**

    Performs event-related EMG analysis on epochs.

    Parameters
    ----------
    epochs : Union[dict, pd.DataFrame]
        A dict containing one DataFrame per event/trial, usually obtained via ``epochs_create()``,
        or a DataFrame containing all epochs, usually obtained via ``epochs_to_df()``.
    silent : bool
        If ``True``, silence possible warnings.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed EMG features for each epoch, with each epoch indicated
        by the `Label` column (if not present, by the `Index` column). The analyzed features consist
        of the following:

        .. codebookadd::
            EMG_Activation|Indication of whether there is muscular activation following.
            EMG_Amplitude_Mean|The mean amplitude of the activity.
            EMG_Amplitude_Max|The maximum amplitude of the activity.
            EMG_Amplitude_SD|The standard deviation of the activity amplitude.
            EMG_Amplitude_Max_Time|The time of maximum amplitude.
            EMG_Bursts|The number of activations, or bursts of activity, within each epoch.

    See Also
    --------
    emg_simulate, emg_process, .events_find, .epochs_create

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Example with simulated data
      emg = nk.emg_simulate(duration=20, sampling_rate=1000, burst_number=3)
      emg_signals, info = nk.emg_process(emg, sampling_rate=1000)
      epochs = nk.epochs_create(emg_signals, events=[3000, 6000, 9000], sampling_rate=1000,
                                epochs_start=-0.1,epochs_end=1.9)
      nk.emg_eventrelated(epochs)

    """
    # Sanity checks
    epochs = _eventrelated_sanitizeinput(epochs, what="emg", silent=silent)

    # Extract features and build dataframe
    data = {}  # Initialize an empty dict
    for i in epochs.keys():

        data[i] = {}  # Initialize an empty dict for the current epoch

        # Activation following event
        if "EMG_Onsets" not in epochs[i]:
            warn(
                "Input does not have an `EMG_Onsets` column." " Unable to process EMG features.",
                category=NeuroKitWarning,
            )
            data[i]["EMG_Activation"] = 0
        elif np.any(epochs[i]["EMG_Onsets"][epochs[i].index > 0] != 0):
            data[i]["EMG_Activation"] = 1
        else:
            data[i]["EMG_Activation"] = 0

        # Analyze features based on activation
        if data[i]["EMG_Activation"] == 1:
            data[i] = _emg_eventrelated_features(epochs[i], data[i])
        else:
            data[i]["EMG_Amplitude_Mean"] = np.nan
            data[i]["EMG_Amplitude_Max"] = np.nan
            data[i]["EMG_Amplitude_SD"] = np.nan
            data[i]["EMG_Amplitude_Max_Time"] = np.nan
            data[i]["EMG_Bursts"] = np.nan

        # Fill with more info
        data[i] = _eventrelated_addinfo(epochs[i], data[i])

    df = _eventrelated_sanitizeoutput(data)

    return df


# =============================================================================
# Internals
# =============================================================================
def _emg_eventrelated_features(epoch, output={}):

    # Sanitize input
    if "EMG_Activity" not in epoch or "EMG_Amplitude" not in epoch:
        warn(
            "Input does not have an `EMG_Activity` column or `EMG_Amplitude` column."
            " Will skip computation of EMG amplitudes.",
            category=NeuroKitWarning,
        )
        return output

    # Peak amplitude and Time of peak
    activations = len(np.where(epoch["EMG_Onsets"][epoch.index > 0] == 1)[0])
    activated_signal = np.where(epoch["EMG_Activity"][epoch.index > 0] == 1)
    mean = np.array(epoch["EMG_Amplitude"][epoch.index > 0].iloc[activated_signal]).mean()
    maximum = np.array(epoch["EMG_Amplitude"][epoch.index > 0].iloc[activated_signal]).max()

    index_time = np.where(epoch["EMG_Amplitude"][epoch.index > 0] == maximum)[0]
    time = np.array(epoch["EMG_Amplitude"][epoch.index > 0].index[index_time])[0]

    output["EMG_Amplitude_Mean"] = mean
    output["EMG_Amplitude_Max"] = maximum
    output["EMG_Amplitude_SD"] = np.std(
        epoch["EMG_Amplitude"][epoch.index > 0].iloc[activated_signal]
    )
    output["EMG_Amplitude_Max_Time"] = time
    output["EMG_Bursts"] = activations

    return output
