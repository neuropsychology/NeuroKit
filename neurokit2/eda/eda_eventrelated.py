# -** coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..epochs.eventrelated_utils import (
    _eventrelated_addinfo,
    _eventrelated_sanitizeinput,
    _eventrelated_sanitizeoutput,
)
from ..misc import NeuroKitWarning


def eda_eventrelated(epochs, silent=False):
    """**Performs event-related EDA analysis on epochs**

    Parameters
    ----------
    epochs : Union[dict, pd.DataFrame]
        A dict containing one DataFrame per event/trial,
        usually obtained via ``"epochs_create()"``, or a DataFrame
        containing all epochs, usually obtained via ``"epochs_to_df()"``.
    silent : bool
        If True, silence possible warnings.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed EDA features for each epoch, with each epoch indicated
        by the `Label` column (if not present, by the `Index` column). The analyzed features consist
        the following:

        .. codebookadd::
            EDA_SCR|indication of whether Skin Conductance Response (SCR) occurs following the \
                event (1 if an SCR onset is present and 0 if absent) and if so, its corresponding \
                peak amplitude, time of peak, rise and recovery time. If there is no occurrence \
                of SCR, nans are displayed for the below features.
            EDA_Peak_Amplitude|The maximum amplitude of the phasic component of the signal.
            SCR_Peak_Amplitude|The peak amplitude of the first SCR in each epoch.
            SCR_Peak_Amplitude_Time|The timepoint of each first SCR peak amplitude.
            SCR_RiseTime|The risetime of each first SCR i.e., the time it takes for SCR to \
                reach peak amplitude from onset.
            SCR_RecoveryTime|The half-recovery time of each first SCR i.e., the time it takes \
                for SCR to decrease to half amplitude.

    See Also
    --------
    .events_find, .epochs_create, .bio_process

    Examples
    ----------
    * **Example 1: Simulated Data**

    .. ipython:: python

      import neurokit2 as nk

      # Example with simulated data
      eda = nk.eda_simulate(duration=15, scr_number=3)

      # Process data
      eda_signals, info = nk.eda_process(eda, sampling_rate=1000)
      epochs = nk.epochs_create(eda_signals, events=[5000, 10000, 15000], sampling_rate=1000,
                                epochs_start=-0.1, epochs_end=1.9)

      # Analyze
      nk.eda_eventrelated(epochs)

    * **Example 2: Real Data**

    .. ipython:: python

      import neurokit2 as nk

      # Example with real data
       data = nk.data("bio_eventrelated_100hz")

      # Process the data
      df, info = nk.bio_process(eda=data["EDA"], sampling_rate=100)
      events = nk.events_find(data["Photosensor"], threshold_keep='below',
                              event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
      epochs = nk.epochs_create(df, events, sampling_rate=100, epochs_start=-0.1, epochs_end=6.9)

      # Analyze
      nk.eda_eventrelated(epochs)

    """
    # Sanity checks
    epochs = _eventrelated_sanitizeinput(epochs, what="eda", silent=silent)

    # Extract features and build dataframe
    data = {}  # Initialize an empty dict
    for i in epochs.keys():

        data[i] = {}  # Initialize an empty dict for the current epoch

        # Maximum phasic amplitude
        data[i] = _eda_eventrelated_eda(epochs[i], data[i])

        # Detect activity following the events
        if np.any(epochs[i]["SCR_Peaks"][epochs[i].index > 0] == 1) and np.any(
            epochs[i]["SCR_Onsets"][epochs[i].index > 0] == 1
        ):
            data[i]["EDA_SCR"] = 1
        else:
            data[i]["EDA_SCR"] = 0

        # Analyze based on if activations are present
        if data[i]["EDA_SCR"] != 0:
            data[i] = _eda_eventrelated_scr(epochs[i], data[i])
        else:
            data[i]["SCR_Peak_Amplitude"] = np.nan
            data[i]["SCR_Peak_Amplitude_Time"] = np.nan
            data[i]["SCR_RiseTime"] = np.nan
            data[i]["SCR_RecoveryTime"] = np.nan

        # Fill with more info
        data[i] = _eventrelated_addinfo(epochs[i], data[i])

    df = _eventrelated_sanitizeoutput(data)

    return df


# =============================================================================
# Internals
# =============================================================================
def _eda_eventrelated_eda(epoch, output={}):

    # Sanitize input
    if "EDA_Phasic" not in epoch:
        warn(
            "Input does not have an `EDA_Phasic` column."
            " Will skip computation of maximum amplitude of phasic EDA component.",
            category=NeuroKitWarning,
        )
        return output

    output["EDA_Peak_Amplitude"] = epoch["EDA_Phasic"].max()
    return output


def _eda_eventrelated_scr(epoch, output={}):

    # Sanitize input
    if "SCR_Amplitude" not in epoch:
        warn(
            "Input does not have an `SCR_Amplitude` column."
            " Will skip computation of SCR peak amplitude.",
            category=NeuroKitWarning,
        )
        return output

    if "SCR_RecoveryTime" not in epoch:
        warn(
            "Input does not have an `SCR_RecoveryTime` column."
            " Will skip computation of SCR half-recovery times.",
            category=NeuroKitWarning,
        )
        return output

    if "SCR_RiseTime" not in epoch:
        warn(
            "Input does not have an `SCR_RiseTime` column."
            " Will skip computation of SCR rise times.",
            category=NeuroKitWarning,
        )
        return output

    epoch_postevent = epoch[epoch.index > 0]
    # Peak amplitude
    first_peak = np.where(epoch_postevent["SCR_Amplitude"] != 0)[0][0]
    output["SCR_Peak_Amplitude"] = epoch_postevent["SCR_Amplitude"].iloc[first_peak]
    # Time of peak (Raw, from epoch onset)
    output["SCR_Peak_Amplitude_Time"] = epoch_postevent.index[first_peak]
    # Rise Time (From the onset of the peak)
    output["SCR_RiseTime"] = epoch_postevent["SCR_RiseTime"].iloc[first_peak]

    # Recovery Time (from peak to half recovery time)
    if any(epoch["SCR_RecoveryTime"][epoch.index > 0] != 0):
        recov_t = np.where(epoch_postevent["SCR_RecoveryTime"] != 0)[0][0]
        output["SCR_RecoveryTime"] = epoch_postevent["SCR_RecoveryTime"].iloc[recov_t]
    else:
        output["SCR_RecoveryTime"] = np.nan

    return output
