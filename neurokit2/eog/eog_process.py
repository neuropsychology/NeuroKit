# -*- coding: utf-8 -*-
import pandas as pd

from ..misc import as_vector
from ..signal import signal_rate
from ..signal.signal_formatpeaks import _signal_from_indices
from .eog_clean import eog_clean
from .eog_findpeaks import eog_findpeaks


def eog_process(veog_signal, sampling_rate=1000, **kwargs):
    """Process an EOG signal.

    Convenience function that automatically processes an EOG signal.


    Parameters
    ----------
    veog_signal : Union[list, np.array, pd.Series]
        The raw vertical EOG channel. Note that it must be positively oriented, i.e., blinks must
        appear as upward peaks.
    sampling_rate : int
        The sampling frequency of `eog_signal` (in Hz, i.e., samples/second). Defaults to 1000.
    **kwargs
        Other arguments passed to other functions.

    Returns
    -------
    signals : DataFrame
        A DataFrame of the same length as the `eog_signal` containing the following columns:
        - *"EOG_Raw"*: the raw signal.
        - *"EOG_Clean"*: the cleaned signal.
        - *"EOG_Blinks"*: the blinks marked as "1" in a list of zeros.
        - *"EOG_Rate"*: eye blinks rate interpolated between blinks.

    info : dict
        A dictionary containing the samples at which the eye blinks occur, accessible with the key "EOG_Blinks".

    See Also
    --------
    eog_clean, eog_findpeaks

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Get data
    >>> eog_signal = nk.data('eog_100hz')
    >>>
    >>> signals, info = nk.eog_process(eog_signal, sampling_rate=100)

    References
    ----------
    - Agarwal, M., & Sivakumar, R. (2019, September). Blink: A Fully Automated Unsupervised Algorithm for
    Eye-Blink Detection in EEG Signals. In 2019 57th Annual Allerton Conference on Communication, Control,
    and Computing (Allerton) (pp. 1113-1121). IEEE.

    """
    # Sanitize input
    eog_signal = as_vector(veog_signal)

    # Clean signal
    eog_cleaned = eog_clean(eog_signal, sampling_rate=sampling_rate, **kwargs)

    # Find peaks
    peaks = eog_findpeaks(eog_cleaned, sampling_rate=sampling_rate, **kwargs)

    info = {"EOG_Blinks": peaks}

    # Mark (potential) blink events
    signal_blinks = _signal_from_indices(peaks, desired_length=len(eog_cleaned))

    # Rate computation
    rate = signal_rate(peaks, sampling_rate=sampling_rate, desired_length=len(eog_cleaned))

    # Prepare output
    signals = pd.DataFrame(
        {"EOG_Raw": eog_signal, "EOG_Clean": eog_cleaned, "EOG_Blinks": signal_blinks, "EOG_Rate": rate}
    )

    return signals, info
