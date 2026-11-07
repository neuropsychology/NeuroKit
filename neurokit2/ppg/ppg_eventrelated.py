from ..epochs.eventrelated_utils import (
    _eventrelated_addinfo,
    _eventrelated_rate,
    _eventrelated_sanitizeinput,
    _eventrelated_sanitizeoutput,
)


def ppg_eventrelated(epochs, silent=False):
    """**Performs event-related PPG analysis on epochs**

    Parameters
    ----------
    epochs : Union[dict, pd.DataFrame]
        A dict containing one DataFrame per event/trial, usually obtained
        via :func:`.epochs_create`, or a DataFrame containing all epochs, usually obtained
        via :func:`.epochs_to_df`.
    silent : bool
        If ``True``, silence possible warnings.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed PPG features for each epoch, with each epoch indicated
        by the `Label` column (if not present, by the `Index` column). The analyzed features
        consist of the following:

        Rate features use the sample nearest stimulus onset as baseline and summarize the samples
        after that baseline sample. Maximum, minimum and mean rates are reported as changes from
        baseline, not absolute rates. The standard deviation is the population standard deviation
        of the post-baseline rates, not a change from baseline. Times are in seconds relative to
        stimulus onset.

        .. codebookadd::
            PPG_Rate_Baseline|The heart rate at the sample nearest stimulus onset.
            PPG_Rate_Max|The maximum change in heart rate from the onset baseline.
            PPG_Rate_Min|The minimum change in heart rate from the onset baseline.
            PPG_Rate_Mean|The mean change in heart rate from the onset baseline.
            PPG_Rate_SD|The population standard deviation of the post-baseline heart rate.
            PPG_Rate_Max_Time|The time of maximum post-baseline heart rate, relative to stimulus onset (seconds).
            PPG_Rate_Min_Time|The time of minimum post-baseline heart rate, relative to stimulus onset (seconds).

        We also include the following *experimental* features related to the parameters of a
        quadratic model:

        .. codebookadd::
            PPG_Rate_Trend_Linear|The parameter corresponding to the linear trend.
            PPG_Rate_Trend_Quadratic|The parameter corresponding to the curvature.
            PPG_Rate_Trend_R2|The quality of the quadratic model. If too low, the parameters \
                might not be reliable or meaningful.

    See Also
    --------
    events_find, epochs_create, ppg_process

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Example with simulated data
      ppg, info = nk.ppg_process(nk.ppg_simulate(duration=20))

      # Process the data
      epochs = nk.epochs_create(ppg, events=[5000, 10000, 15000],
                               epochs_start=-0.1, epochs_end=1.9)
      nk.ppg_eventrelated(epochs)

    """
    # Sanity checks
    epochs = _eventrelated_sanitizeinput(epochs, what="ppg", silent=silent)

    # Extract features and build dataframe
    data = {}  # Initialize an empty dict
    for i in epochs.keys():
        data[i] = {}  # Initialize empty container

        # Rate
        data[i] = _eventrelated_rate(epochs[i], data[i], var="PPG_Rate")

        # Fill with more info
        data[i] = _eventrelated_addinfo(epochs[i], data[i])

    # Return dataframe
    return _eventrelated_sanitizeoutput(data)
