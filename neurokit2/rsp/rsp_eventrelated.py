# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..epochs.eventrelated_utils import (
    _eventrelated_addinfo,
    _eventrelated_rate,
    _eventrelated_sanitizeinput,
    _eventrelated_sanitizeoutput,
)
from ..misc import NeuroKitWarning


def rsp_eventrelated(epochs, silent=False):
    """Performs event-related RSP analysis on epochs.

    Parameters
    ----------
    epochs : Union[dict, pd.DataFrame]
        A dict containing one DataFrame per event/trial, usually obtained via `epochs_create()`,
        or a DataFrame containing all epochs, usually obtained via `epochs_to_df()`.
    silent : bool
        If True, silence possible warnings.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed RSP features for each epoch,
        with each epoch indicated by the `Label` column (if not
        present, by the `Index` column). The analyzed features
        consist of the following:
        - *"RSP_Rate_Max"*: the maximum respiratory rate after stimulus onset.
        - *"RSP_Rate_Min"*: the minimum respiratory rate after stimulus onset.
        - *"RSP_Rate_Mean"*: the mean respiratory rate after stimulus onset.
        - *"RSP_Rate_Max_Time"*: the time at which maximum respiratory rate occurs.
        - *"RSP_Rate_Min_Time"*: the time at which minimum respiratory rate occurs.
        - *"RSP_Amplitude_Max"*: the maximum respiratory amplitude after stimulus onset.
        - *"RSP_Amplitude_Min"*: the minimum respiratory amplitude after stimulus onset.
        - *"RSP_Amplitude_Mean"*: the mean respiratory amplitude after stimulus onset.
        - *"RSP_Phase"*: indication of whether the onset of the event concurs with respiratory
        inspiration (1) or expiration (0).
        - *"RSP_PhaseCompletion"*: indication of the stage of the current respiration phase (0 to 1)
        at the onset of the event.

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Example with simulated data
    >>> rsp, info = nk.rsp_process(nk.rsp_simulate(duration=120))
    >>> epochs = nk.epochs_create(rsp, events=[5000, 10000, 15000], epochs_start=-0.1, epochs_end=1.9)
    >>>
    >>> # Analyze
    >>> rsp1 = nk.rsp_eventrelated(epochs)
    >>> rsp1 #doctest: +SKIP
    >>>
    >>> # Example with real data
    >>> data = nk.data("bio_eventrelated_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(rsp=data["RSP"], sampling_rate=100)
    >>> events = nk.events_find(data["Photosensor"], threshold_keep='below',
    ...                         event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
    >>> epochs = nk.epochs_create(df, events, sampling_rate=100, epochs_start=-0.1, epochs_end=2.9)
    >>>
    >>> # Analyze
    >>> rsp2 = nk.rsp_eventrelated(epochs)
    >>> rsp2 #doctest: +SKIP

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
            category=NeuroKitWarning
        )
        return output

    # Get baseline
    if np.min(epoch.index.values) <= 0:
        baseline = epoch["RSP_Amplitude"][epoch.index <= 0].values
        signal = epoch["RSP_Amplitude"][epoch.index > 0].values
    else:
        baseline = epoch["RSP_Amplitude"][np.min(epoch.index.values) : np.min(epoch.index.values)].values
        signal = epoch["RSP_Amplitude"][epoch.index > np.min(epoch.index)].values

    # Max / Min / Mean
    output["RSP_Amplitude_Max"] = np.max(signal) - np.mean(baseline)
    output["RSP_Amplitude_Min"] = np.min(signal) - np.mean(baseline)
    output["RSP_Amplitude_Mean"] = np.mean(signal) - np.mean(baseline)

    return output


def _rsp_eventrelated_inspiration(epoch, output={}):

    # Sanitize input
    if "RSP_Phase" not in epoch:
        warn(
            "Input does not have an `RSP_Phase` column."
            " Will not indicate whether event onset concurs with inspiration.",
            category=NeuroKitWarning
        )
        return output

    # Indication of inspiration
    output["RSP_Phase"] = epoch["RSP_Phase"][epoch.index > 0].iloc[0]
    output["RSP_Phase_Completion"] = epoch["RSP_Phase_Completion"][epoch.index > 0].iloc[0]

    return output
