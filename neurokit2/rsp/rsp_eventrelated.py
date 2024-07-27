# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..epochs.eventrelated_utils import (
    _eventrelated_addinfo,
    _eventrelated_rate,
    _eventrelated_sanitizeinput,
    _eventrelated_sanitizeoutput,
)
from ..misc import NeuroKitWarning, find_closest


def rsp_eventrelated(epochs, silent=False):
    """**Performs event-related RSP analysis on epochs**

    Parameters
    ----------
    epochs : Union[dict, pd.DataFrame]
        A dict containing one DataFrame per event/trial, usually obtained via
        :func:`.epochs_create`, or a DataFrame containing all epochs, usually obtained
        via :func:`.epochs_to_df`.
    silent : bool
        If ``True``, silence possible warnings.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed RSP features for each epoch, with each epoch indicated
        by the `Label` column (if not present, by the `Index` column). The analyzed features
        consist of the following:

        .. codebookadd::
            RSP_Rate_Max|The maximum respiratory rate after stimulus onset.
            RSP_Rate_Min|The minimum respiratory rate after stimulus onset.
            RSP_Rate_Mean|The mean respiratory rate after stimulus onset.
            RSP_Rate_SD|The standard deviation of the respiratory rate after stimulus onset.
            RSP_Rate_Max_Time|The time at which maximum respiratory rate occurs.
            RSP_Rate_Min_Time|The time at which minimum respiratory rate occurs.
            RSP_Amplitude_Baseline|The respiratory amplitude at stimulus onset.
            RSP_Amplitude_Max|The change in maximum respiratory amplitude from before stimulus onset.
            RSP_Amplitude_Min|The change in minimum respiratory amplitude from before stimulus onset.
            RSP_Amplitude_Mean|The change in mean respiratory amplitude from before stimulus onset.
            RSP_Amplitude_SD|The standard deviation of the respiratory amplitude after stimulus onset.
            RSP_Phase|Indication of whether the onset of the event concurs with respiratory inspiration (1) or expiration (0).
            RSP_PhaseCompletion|Indication of the stage of the current respiration phase (0 to 1) at the onset of the event.

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Example with simulated data
      rsp, info = nk.rsp_process(nk.rsp_simulate(duration=120))
      epochs = nk.epochs_create(rsp, events=[5000, 10000, 15000], epochs_start=-0.1, epochs_end=1.9)

      # Analyze
      nk.rsp_eventrelated(epochs)

    .. ipython:: python

      # Example with real data
      data = nk.data("bio_eventrelated_100hz")

      # Process the data
      df, info = nk.bio_process(rsp=data["RSP"], sampling_rate=100)
      events = nk.events_find(data["Photosensor"], threshold_keep='below',
                             event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
      epochs = nk.epochs_create(df, events, sampling_rate=100, epochs_start=-0.1, epochs_end=2.9)

      # Analyze
      nk.rsp_eventrelated(epochs)


    """
    # Sanity checks
    epochs = _eventrelated_sanitizeinput(epochs, what="rsp", silent=silent)

    # Extract features and build dataframe
    data = {}  # Initialize an empty dict
    for i in epochs.keys():

        data[i] = {}  # Initialize empty container

        # Rate
        data[i] = _eventrelated_rate(epochs[i], data[i], var="RSP_Rate")

        # Amplitude
        data[i] = _rsp_eventrelated_amplitude(epochs[i], data[i])

        # Inspiration
        data[i] = _rsp_eventrelated_inspiration(epochs[i], data[i])

        # RVT
        data[i] = _rsp_eventrelated_rvt(epochs[i], data[i])

        # Fill with more info
        data[i] = _eventrelated_addinfo(epochs[i], data[i])

    df = _eventrelated_sanitizeoutput(data)

    return df


# =============================================================================
# Internals
# =============================================================================


def _rsp_eventrelated_amplitude(epoch, output={}):

    # Sanitize input
    if "RSP_Amplitude" not in epoch:
        warn(
            "Input does not have an `RSP_Amplitude` column."
            " Will skip all amplitude-related features.",
            category=NeuroKitWarning,
        )
        return output

    # Get baseline
    zero = find_closest(0, epoch.index.values, return_index=True)  # Find index closest to 0
    baseline = epoch["RSP_Amplitude"].iloc[zero]
    signal = epoch["RSP_Amplitude"].values[zero + 1 : :]

    # Max / Min / Mean
    output["RSP_Amplitude_Baseline"] = baseline
    output["RSP_Amplitude_Max"] = np.max(signal) - baseline
    output["RSP_Amplitude_Min"] = np.min(signal) - baseline
    output["RSP_Amplitude_MeanRaw"] = np.mean(signal)
    output["RSP_Amplitude_Mean"] = output["RSP_Amplitude_MeanRaw"] - baseline
    output["RSP_Amplitude_SD"] = np.std(signal)

    return output


def _rsp_eventrelated_inspiration(epoch, output={}):

    # Sanitize input
    if "RSP_Phase" not in epoch:
        warn(
            "Input does not have an `RSP_Phase` column."
            " Will not indicate whether event onset concurs with inspiration.",
            category=NeuroKitWarning,
        )
        return output

    # Indication of inspiration
    output["RSP_Phase"] = epoch["RSP_Phase"][epoch.index > 0].iloc[0]
    output["RSP_Phase_Completion"] = epoch["RSP_Phase_Completion"][epoch.index > 0].iloc[0]

    return output


def _rsp_eventrelated_rvt(epoch, output={}):

    # Sanitize input
    if "RSP_RVT" not in epoch:
        warn(
            "Input does not have an `RSP_RVT` column. Will skip all RVT-related features.",
            category=NeuroKitWarning,
        )
        return output

    # Get baseline
    zero = find_closest(0, epoch.index.values, return_index=True)  # Find index closest to 0
    baseline = epoch["RSP_RVT"].iloc[zero]
    signal = epoch["RSP_RVT"].values[zero + 1 : :]

    # Mean
    output["RSP_RVT_Baseline"] = baseline
    output["RSP_RVT_Mean"] = np.mean(signal) - baseline

    return output


def _rsp_eventrelated_symmetry(epoch, output={}):

    # Sanitize input
    if "RSP_Symmetry_PeakTrough" not in epoch:
        warn(
            "Input does not have an `RSP_Symmetry_PeakTrough` column."
            + " Will skip all symmetry-related features.",
            category=NeuroKitWarning,
        )
        return output

    # Get baseline
    zero = find_closest(0, epoch.index.values, return_index=True)  # Find index closest to 0
    baseline1 = epoch["RSP_Symmetry_PeakTrough"].iloc[zero]
    signal1 = epoch["RSP_Symmetry_PeakTrough"].values[zero + 1 : :]

    baseline2 = epoch["RSP_Symmetry_RiseDecay"].iloc[zero]
    signal2 = epoch["RSP_Symmetry_RiseDecay"].values[zero + 1 : :]

    # Mean
    output["RSP_Symmetry_PeakTrough_Baseline"] = baseline1
    output["RSP_Symmetry_RiseDecay_Baseline"] = baseline2
    output["RSP_Symmetry_PeakTrough_Mean"] = np.mean(signal1) - baseline1
    output["RSP_Symmetry_RiseDecay_Mean"] = np.mean(signal2) - baseline2

    return output
