import pandas as pd


def signal_fillmissing(signal, method="both"):
    """**Handle missing values**

    Fill missing values in a signal using specific methods.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        The method to use to fill missing values. Can be one of ``"forward"``, ``"backward"``,
        or ``"both"``. The default is ``"both"``.

    Returns
    -------
    signal

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, np.nan]
      nk.signal_fillmissing(signal, method="forward")
      nk.signal_fillmissing(signal, method="backward")
      nk.signal_fillmissing(signal, method="both")

    """
    if method in ["forward", "forwards", "ffill", "both"]:
        signal = pd.Series(signal).ffill().values

    if method in ["backward", "backwards", "back", "bfill", "both"]:
        signal = pd.Series(signal).bfill().values
    return signal
