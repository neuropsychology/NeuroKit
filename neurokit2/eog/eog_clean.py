# -*- coding: utf-8 -*-

from ..misc import as_vector
from ..signal import signal_filter


def eog_clean(eog_signal, sampling_rate=1000):
    """Clean an EOG signal. Only Agarwal & Sivakumar (2019)'s method is implemented for now.

    Prepare a raw EOG signal for eye blinks detection.

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
    >>> eog = nk.eog_extract(raw, channels=["124", "125"], resampling_rate=None, raw_return=True)
    >>> sampling_rate = raw.info['sfreq']
    >>> eog_cleaned = nk.eog_clean(eog, sampling_rate=sampling_rate)
    >>> fig = pd.DataFrame({"Raw": eog,
    ...                     "Cleaned": filtered}).plot() #doctest: +ELLIPSIS
    <matplotlib.axes._subplots.AxesSubplot at ...>


    References
    ----------
    - Agarwal, M., & Sivakumar, R. (2019, September). Blink: A Fully Automated Unsupervised Algorithm for
    Eye-Blink Detection in EEG Signals. In 2019 57th Annual Allerton Conference on Communication, Control,
    and Computing (Allerton) (pp. 1113-1121). IEEE.

    """
    eog_signal = as_vector(eog_signal)

    # Filter
    clean = signal_filter(
        eog_signal, sampling_rate=sampling_rate, method="butterworth", order=4, lowcut=None, highcut=10
    )

    return clean
