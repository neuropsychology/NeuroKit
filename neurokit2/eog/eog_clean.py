# -*- coding: utf-8 -*-
import pandas as pd

from ..misc import as_vector
from ..signal import signal_filter


def eog_clean(eog_signal, sampling_rate=1000):
    """Clean an EOG signal.

    Prepare a raw EOG signal for eye blinks detection. Only Agarwal & Sivakumar (2019)'s method is implemented for now.

    Parameters
    ----------
    eog_signal : list or array or Series
        The raw EOG channel.
    sampling_rate : int
        The sampling frequency of `eog_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.

    Returns
    -------
    array
        Vector containing the cleaned EOG signal.

    See Also
    --------
    eog_extract, signal_filter

    Examples
    --------
    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> eog_signal = nk.data('eog_100hz')["vEOG"]
    >>> eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100)
    >>> fig = pd.DataFrame({"Raw": eog_signal,
    ...                     "Cleaned": eog_cleaned}).plot() #doctest: +ELLIPSIS
    <matplotlib.axes._subplots.AxesSubplot at ...>


    References
    ----------
    - Agarwal, M., & Sivakumar, R. (2019, September). Blink: A Fully Automated Unsupervised Algorithm for
    Eye-Blink Detection in EEG Signals. In 2019 57th Annual Allerton Conference on Communication, Control,
    and Computing (Allerton) (pp. 1113-1121). IEEE.

    """
    # Sanitize input
    if isinstance(eog_signal, pd.DataFrame):
        if len(eog_signal.columns) == 1:
            eog_signal = as_vector(eog_signal)
        elif len(eog_signal.columns) == 2:
            eog_signal = eog_signal.iloc[:, 0] - eog_signal.iloc[:, 1]
            eog_signal = as_vector(eog_signal)

    # Filter
    clean = signal_filter(
        eog_signal, sampling_rate=sampling_rate, method="butterworth", order=4, lowcut=None, highcut=10
    )

    return clean
