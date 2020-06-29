# -*- coding: utf-8 -*-

from ..epochs.eventrelated_utils import (
    _eventrelated_addinfo,
    _eventrelated_rate,
    _eventrelated_sanitizeinput,
    _eventrelated_sanitizeoutput,
)


def ecg_eventrelated(epochs, silent=False):
    """Performs event-related ECG analysis on epochs.

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
        A dataframe containing the analyzed ECG features for each epoch, with each epoch indicated by
        the `Label` column (if not present, by the `Index` column). The analyzed features consist of
        the following:

        - *"ECG_Rate_Max"*: the maximum heart rate after stimulus onset.

        - *"ECG_Rate_Min"*: the minimum heart rate after stimulus onset.

        - *"ECG_Rate_Mean"*: the mean heart rate after stimulus onset.

        - *"ECG_Rate_Max_Time"*: the time at which maximum heart rate occurs.

        - *"ECG_Rate_Min_Time"*: the time at which minimum heart rate occurs.

        - *"ECG_Phase_Atrial"*: indication of whether the onset of the event concurs with respiratory
          systole (1) or diastole (0).

        - *"ECG_Phase_Ventricular"*: indication of whether the onset of the event concurs with respiratory
          systole (1) or diastole (0).

        - *"ECG_Phase_Atrial_Completion"*: indication of the stage of the current cardiac (atrial) phase
          (0 to 1) at the onset of the event.

        - *"ECG_Phase_Ventricular_Completion"*: indication of the stage of the current cardiac (ventricular)
          phase (0 to 1) at the onset of the event.

        We also include the following *experimental* features related to the parameters of a
        quadratic model:

        - *"ECG_Rate_Trend_Linear"*: The parameter corresponding to the linear trend.

        - *"ECG_Rate_Trend_Quadratic"*: The parameter corresponding to the curvature.

        - *"ECG_Rate_Trend_R2"*: the quality of the quadratic model. If too low, the parameters might
          not be reliable or meaningful.

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Example with simulated data
    >>> ecg, info = nk.ecg_process(nk.ecg_simulate(duration=20))
    >>>
    >>> # Process the data
    >>> epochs = nk.epochs_create(ecg, events=[5000, 10000, 15000],
    ...                           epochs_start=-0.1, epochs_end=1.9)
    >>> nk.ecg_eventrelated(epochs) #doctest: +ELLIPSIS
      Label  Event_Onset  ...  ECG_Phase_Completion_Ventricular  ECG_Quality_Mean
    1     1          ...  ...                               ...               ...
    2     2          ...  ...                               ...               ...
    3     3          ...  ...                               ...               ...

    [3 rows x 16 columns]
    >>>
    >>> # Example with real data
    >>> data = nk.data("bio_eventrelated_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(ecg=data["ECG"], sampling_rate=100)
    >>> events = nk.events_find(data["Photosensor"],
    ...                         threshold_keep='below',
    ...                         event_conditions=["Negative", "Neutral",
    ...                                           "Neutral", "Negative"])
    >>> epochs = nk.epochs_create(df, events, sampling_rate=100,
    ...                           epochs_start=-0.1, epochs_end=1.9)
    >>> nk.ecg_eventrelated(epochs) #doctest: +ELLIPSIS
      Label Condition  ...  ECG_Phase_Completion_Ventricular  ECG_Quality_Mean
    1     1  Negative  ...                               ...               ...
    2     2   Neutral  ...                               ...               ...
    3     3   Neutral  ...                               ...               ...
    4     4  Negative  ...                               ...               ...

    [4 rows x 17 columns]

    """
    # Sanity checks
    epochs = _eventrelated_sanitizeinput(epochs, what="ecg", silent=silent)

    # Extract features and build dataframe
    data = {}  # Initialize an empty dict
    for i in epochs.keys():

        data[i] = {}  # Initialize empty container

        # Rate
        data[i] = _eventrelated_rate(epochs[i], data[i], var="ECG_Rate")

        # Cardiac Phase
        data[i] = _ecg_eventrelated_phase(epochs[i], data[i])

        # Quality
        data[i] = _ecg_eventrelated_quality(epochs[i], data[i])

        # Fill with more info
        data[i] = _eventrelated_addinfo(epochs[i], data[i])

    # Return dataframe
    return _eventrelated_sanitizeoutput(data)


# =============================================================================
# Internals
# =============================================================================


def _ecg_eventrelated_phase(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "ECG_Phase_Atrial" in i]) == 0:
        print(
            "NeuroKit warning: ecg_eventrelated(): input does not"
            "have an `ECG_Phase_Artrial` or `ECG_Phase_Ventricular` column."
            "Will not indicate whether event onset concurs with cardiac"
            "phase."
        )
        return output

    # Indication of atrial systole
    output["ECG_Phase_Atrial"] = epoch["ECG_Phase_Atrial"][epoch.index > 0].iloc[0]
    output["ECG_Phase_Completion_Atrial"] = epoch["ECG_Phase_Completion_Atrial"][epoch.index > 0].iloc[0]

    # Indication of ventricular systole
    output["ECG_Phase_Ventricular"] = epoch["ECG_Phase_Ventricular"][epoch.index > 0].iloc[0]
    output["ECG_Phase_Completion_Ventricular"] = epoch["ECG_Phase_Completion_Ventricular"][epoch.index > 0].iloc[0]

    return output


def _ecg_eventrelated_quality(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "ECG_Quality" in i]) == 0:
        print(
            "NeuroKit warning: ecg_eventrelated(): input does not"
            "have an `ECG_Quality` column. Quality of the signal"
            "is not computed."
        )
        return output

    # Average signal quality over epochs
    output["ECG_Quality_Mean"] = epoch["ECG_Quality"].mean()

    return output
