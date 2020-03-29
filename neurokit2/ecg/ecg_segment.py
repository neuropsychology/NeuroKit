# - * - coding: utf-8 - * -
import numpy as np

from .ecg_peaks import ecg_peaks
from ..epochs import epochs_create



def ecg_segment(ecg_cleaned, rpeaks=None, sampling_rate=1000, show=False):
    """
    """
    # Sanitize inputs
    if rpeaks is None:
        _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks["ECG_R_Peaks"]

    epochs_start, epochs_end = _ecg_segment_window(rpeaks=rpeaks,
                                                   sampling_rate=sampling_rate)
    heartbeats = epochs_create(ecg_cleaned,
                               rpeaks,
                               sampling_rate=sampling_rate,
                               epochs_start=epochs_start,
                               epochs_end=epochs_end)
    return heartbeats














def _ecg_segment_window(heart_rate=None, rpeaks=None, sampling_rate=1000):

    # Extract heart rate
    if heart_rate is not None:
        heart_rate = np.mean(heart_rate)
    if rpeaks is not None:
        heart_rate = np.mean(np.diff(rpeaks) / sampling_rate * 60)

    # Modulator
    m = heart_rate/80

    # Window
    epochs_start = -0.35/m
    epochs_end = 0.5/m

    return epochs_start, epochs_end
