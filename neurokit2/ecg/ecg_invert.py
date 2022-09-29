import numpy as np
from .ecg_clean import ecg_clean


def ecg_invert(ecg_signal, sampling_rate=1000, check_inverted=True):
    inverted_ecg_signal = ecg_signal * -1 + 2 * np.nanmean(ecg_signal)
    if check_inverted:
        if _ecg_inverted(ecg_signal, sampling_rate=sampling_rate):
            return inverted_ecg_signal
        else:
            return ecg_signal
    else:
        return inverted_ecg_signal


def _ecg_inverted(ecg_signal, sampling_rate=1000):
    ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    med_max = np.nanmedian(
        _roll_func(ecg_cleaned, window=1 * sampling_rate, func=_abs_max)
    )
    return med_max < np.nanmean(ecg_cleaned)


def _roll_func(x, window, func, func_args={}):
    roll_x = np.array(
        [func(x[i : i + window], **func_args) for i in range(len(x) - window)]
    )
    return roll_x


def _abs_max(x):
    return x[np.argmax(np.abs(x))]
