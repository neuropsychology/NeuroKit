# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..epochs.eventrelated_utils import (_eventrelated_addinfo,
                                         _eventrelated_rate,
                                         _eventrelated_sanitizeinput,
                                         _eventrelated_sanitizeoutput)
from ..misc import NeuroKitWarning, find_closest


def rsp_eventrelated(epochs, silent=False, subepoch_rate=[None, None]):
    """Performs event-related RSP analysis on epochs.

    Parameters
    ----------
    epochs : Union[dict, pd.DataFrame]
        A dict containing one DataFrame per event/trial, usually obtained via `epochs_create()`,
        or a DataFrame containing all epochs, usually obtained via `epochs_to_df()`.
    silent : bool
        If True, silence possible warnings.
    subepoch_rate : list
        A smaller "sub-epoch" within the epoch of an event can be specified.
        The ECG rate-related features of this "sub-epoch" (e.g., RSP_Rate, RSP_Rate_Max),
        relative to the baseline (where applicable), will be computed. The first value of the list specifies
        the start of the sub-epoch and the second specifies the end of the sub-epoch (in seconds),
        e.g., subepoch_rate = [1, 3] or subepoch_rate = [1, None]. Defaults to [None, None].

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
        - *"RSP_Rate_SD"*: the standard deviation of the respiratory rate after stimulus onset.
        - *"RSP_Rate_Max_Time"*: the time at which maximum respiratory rate occurs.
        - *"RSP_Rate_Min_Time"*: the time at which minimum respiratory rate occurs.
        - *"RSP_Amplitude_Baseline"*: the respiratory amplitude at stimulus onset.
        - *"RSP_Amplitude_Max"*: the change in maximum respiratory amplitude from before stimulus onset.
        - *"RSP_Amplitude_Min"*: the change in minimum respiratory amplitude from before stimulus onset.
        - *"RSP_Amplitude_Mean"*: the change in mean respiratory amplitude from before stimulus onset.
        - *"RSP_Amplitude_SD"*: the standard deviation of the respiratory amplitude after stimulus onset.
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
        data[i] = _eventrelated_rate(epochs[i], data[i], var="RSP_Rate", subepoch_rate=subepoch_rate)

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
    zero = find_closest(0, epoch.index.values, return_index=True)  # Find index closest to 0
    baseline = epoch["RSP_Amplitude"].iloc[zero]
    signal = epoch["RSP_Amplitude"].values[zero + 1 : :]

    # Max / Min / Mean
    output["RSP_Amplitude_Baseline"] = baseline
    output["RSP_Amplitude_Max"] = np.max(signal) - baseline
    output["RSP_Amplitude_Min"] = np.min(signal) - baseline
    output["RSP_Amplitude_Mean"] = np.mean(signal) - baseline
    output["RSP_Amplitude_SD"] = np.std(signal)

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
