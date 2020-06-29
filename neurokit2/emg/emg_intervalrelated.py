# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def emg_intervalrelated(data):
    """Performs EMG analysis on longer periods of data (typically > 10 seconds), such as resting-state data.

    Parameters
    ----------
    data : Union[dict, pd.DataFrame]
        A DataFrame containing the different processed signal(s) as different columns, typically generated
        by `emg_process()` or `bio_process()`. Can also take a dict containing sets of separately
        processed DataFrames.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed EMG features. The analyzed features consist of the following:
        - *"EMG_Activation_N"*: the number of bursts of muscular activity.
        - *"EMG_Amplitude_Mean"*: the mean amplitude of the muscular activity.

    See Also
    --------
    bio_process, emg_eventrelated

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Example with simulated data
    >>> emg = nk.emg_simulate(duration=40, sampling_rate=1000, burst_number=3)
    >>> emg_signals, info = nk.emg_process(emg, sampling_rate=1000)
    >>>
    >>> # Single dataframe is passed
    >>> nk.emg_intervalrelated(emg_signals) #doctest: +SKIP
    >>>
    >>> epochs = nk.epochs_create(emg_signals, events=[0, 20000], sampling_rate=1000, epochs_end=20)
    >>> nk.emg_intervalrelated(epochs) #doctest: +SKIP

    """
    intervals = {}

    # Format input
    if isinstance(data, pd.DataFrame):
        activity_cols = [col for col in data.columns if "EMG_Onsets" in col]
        if len(activity_cols) == 1:
            intervals["Activation_N"] = data[activity_cols[0]].values.sum()
        else:
            raise ValueError(
                "NeuroKit error: emg_intervalrelated(): Wrong"
                "input, we couldn't extract activity bursts."
                "Please make sure your DataFrame"
                "contains an `EMG_Onsets` column."
            )
        amplitude_cols = ["EMG_Amplitude", "EMG_Activity"]
        len([col in data.columns for col in amplitude_cols])
        if len(amplitude_cols) == 2:
            data_bursts = data.loc[data["EMG_Activity"] == 1]
            intervals["Amplitude_Mean"] = data_bursts["EMG_Amplitude"].values.mean()
        else:
            raise ValueError(
                "NeuroKit error: emg_intervalrelated(): Wrong"
                "input, we couldn't extract EMG amplitudes."
                "Please make sure your DataFrame contains both"
                "`EMG_Amplitude` and `EMG_Activity` columns."
            )

        emg_intervals = pd.DataFrame.from_dict(intervals, orient="index").T.add_prefix("EMG_")

    elif isinstance(data, dict):
        for index in data:
            intervals[index] = {}  # Initialize empty container

            intervals[index] = _emg_intervalrelated_formatinput(data[index], intervals[index])
        emg_intervals = pd.DataFrame.from_dict(intervals, orient="index")

    return emg_intervals


# =============================================================================
# Internals
# =============================================================================


def _emg_intervalrelated_formatinput(interval, output={}):
    """Format input for dictionary."""
    # Sanitize input
    colnames = interval.columns.values
    if len([i for i in colnames if "EMG_Onsets" in i]) == 0:
        raise ValueError(
            "NeuroKit error: emg_intervalrelated(): Wrong"
            "input, we couldn't extract activity bursts."
            "Please make sure your DataFrame"
            "contains an `EMG_Onsets` column."
        )

    activity_cols = ["EMG_Amplitude", "EMG_Activity"]
    if len([i in colnames for i in activity_cols]) != 2:
        raise ValueError(
            "NeuroKit error: emg_intervalrelated(): Wrong"
            "input, we couldn't extract EMG amplitudes."
            "Please make sure your DataFrame contains both"
            "`EMG_Amplitude` and `EMG_Activity` columns."
        )

    bursts = interval["EMG_Onsets"].values
    data_bursts = interval.loc[interval["EMG_Activity"] == 1]

    output["EMG_Activation_N"] = np.sum(bursts)
    output["EMG_Amplitude_Mean"] = data_bursts["EMG_Amplitude"].values.mean()

    return output
