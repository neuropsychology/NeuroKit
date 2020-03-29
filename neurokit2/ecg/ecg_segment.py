# - * - coding: utf-8 - * -
import numpy as np
from ..epochs import epochs_create





def ecg_segment(ecg_cleaned, rpeaks=None, sampling_rate=1000, show=False):
    """
    """
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
