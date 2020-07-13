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


def eog_eventrelated(epochs, silent=False):
    """Performs event-related EOG analysis on epochs.

    Parameters
    ----------
    epochs : Union[dict, pd.DataFrame]
        A dict containing one DataFrame per event/trial,
        usually obtained via `epochs_create()`, or a DataFrame
        containing all epochs, usually obtained via `epochs_to_df()`.
    silent : bool
        If True, silence possible warnings.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed EOG features for each epoch, with each epoch indicated by
        the `Label` column (if not present, by the `Index` column). The analyzed features consist of
        the following:

        - *"EOG_Rate_Baseline"*: the baseline EOG rate before stimulus onset.

        - *"EOG_Rate_Max"*: the maximum EOG rate after stimulus onset.

        - *"EOG_Rate_Min"*: the minimum EOG rate after stimulus onset.

        - *"EOG_Rate_Mean"*: the mean EOG rate after stimulus onset.

        - *"EOG_Rate_Max_Time"*: the time at which maximum EOG rate occurs.

        - *"EOG_Rate_Min_Time"*: the time at which minimum EOG rate occurs.

        - *"EOG_Blinks_Presence"*: marked with '1' if a blink occurs in the epoch, and '0' if not.

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Example with real data
    >>> eog = nk.data('eog_100hz')
    >>>
    >>> # Process the data
    >>> eog_signals, info = nk.bio_process(eog=eog, sampling_rate=100)
    >>> epochs = nk.epochs_create(eog_signals, events=[500, 4000, 6000, 9000], sampling_rate=100,
    ...                           epochs_start=-0.1,epochs_end=1.9)
    >>>
    >>> # Analyze
    >>> nk.eog_eventrelated(epochs) #doctest: +ELLIPSIS
      Label  Event_Onset  ...  EOG_Rate_Min_Time  EOG_Blinks_Presence
    1     1          ...  ...                ...                  ...
    2     2          ...  ...                ...                  ...
    3     3          ...  ...                ...                  ...
    4     4          ...  ...                ...                  ...
    [4 rows x 9 columns]

    """
    # Sanity checks
    epochs = _eventrelated_sanitizeinput(epochs, what="eog", silent=silent)

    # Extract features and build dataframe
    data = {}  # Initialize an empty dict
    for i in epochs.keys():

        data[i] = {}  # Initialize an empty dict for the current epoch

        # Rate
        data[i] = _eventrelated_rate(epochs[i], data[i], var="EOG_Rate")

        # Number of blinks per epoch
        data[i] = _eog_eventrelated_features(epochs[i], data[i])
        for x in ["EOG_Rate_Trend_Quadratic", "EOG_Rate_Trend_Linear", "EOG_Rate_Trend_R2"]:
            data[i].pop(x, None)

        # Fill with more info
        data[i] = _eventrelated_addinfo(epochs[i], data[i])

    df = _eventrelated_sanitizeoutput(data)

    return df


# =============================================================================
# Internals
# =============================================================================
def _eog_eventrelated_features(epoch, output={}):

    # Sanitize input
    if "EOG_Blinks" not in epoch:
        warn(
            "Input does not have an `EOG_Blinks` column."
            " Unable to process blink features.",
            category=NeuroKitWarning
        )
        return output

    if "EOG_Rate" not in epoch:
        warn(
            "Input does not have an `EOG_Rate` column."
            " Will skip computation of EOG rate.",
            category=NeuroKitWarning
        )
        return output

    # Detect whether blink exists after onset of stimulus
    blinks_presence = len(np.where(epoch["EOG_Blinks"][epoch.index > 0] == 1)[0])

    if blinks_presence > 0:
        output["EOG_Blinks_Presence"] = 1
    else:
        output["EOG_Blinks_Presence"] = 0

    return output
