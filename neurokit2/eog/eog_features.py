# -*- coding: utf-8 -*-
import numpy as np

from ..eog.eog_findpeaks import _eog_findpeaks_blinker_delineate


def eog_features(eog_cleaned, peaks, sampling_rate=1000):
    """
    Parameters
    ----------
    eog_cleaned : Union[list, np.array, pd.Series]
        The cleaned EOG channel, extracted from `eog_clean()`.
    peaks : np.array
        Vector containing the samples at which EOG-peaks occur.
    sampling_rate : int
        The sampling frequency of `eog_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.

    Returns
    -------
    pAVR : list
        - Statistics for the positive amplitude velocity ratio calculated from the leftZero to
        peak of each blink.
    nAVR : list
        - Statistics for the negative amplitude velocity ratio calculated from the peak to the
        rightZero of each blink.

    See Also
    --------
    eog_clean, eog_findpeaks

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Get data
    >>> eog_signal = nk.data('eog_100hz')
    >>> eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100)
    >>> peaks = nk.eog_findpeaks(eog_cleaned, sampling_rate=100, method="blinker")
    >>> pAVR, nAVR = nk.eog_features(eog_cleaned, peaks, sampling_rate=100)

    """

    candidate_blinks, _, leftzeros, rightzeros, downstrokes, upstrokes = _eog_findpeaks_blinker_delineate(eog_cleaned, peaks, sampling_rate=sampling_rate)

    pAVR_list = []
    nAVR_list = []

    for i in range(len(peaks)):
        # Closing blink (pAVR)
        blink_close = upstrokes[i].Signal
        change_close = np.diff(blink_close)
        duration_close = len(change_close)/sampling_rate
        pAVR = abs(change_close.max()/duration_close)*100
        pAVR_list.append(pAVR)

        # Opening blink (nAVR)
        blink_open = downstrokes[i].Signal
        change_open = np.diff(blink_open)
        duration_open = len(change_open)/sampling_rate
        nAVR = abs(change_open.max()/duration_open)*100
        nAVR_list.append(nAVR)

    return pAVR, nAVR
