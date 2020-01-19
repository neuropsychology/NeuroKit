# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def _signal_formatpeaks(peaks, desired_length=None):
    # Retrieve length.
    if desired_length is None:
        if isinstance(peaks, np.ndarray):
            desired_length = np.max(peaks)
        else:
            desired_length = len(peaks)

    if desired_length < len(peaks):
        raise ValueError("NeuroKit error: signal_rate(): 'desired_length' cannot",
                         " be lower than the length of the signal. Please input a greater 'desired_length'.")


    # Attempt to retrieve column.
    if isinstance(peaks, pd.DataFrame):
        col = [col for col in peaks.columns if 'Peaks' in col]
        if len(col) == 0:
            TypeError("NeuroKit error: signal_rate(): wrong type of input ",
                      "provided. Please provide indices of peaks.")
        peaks_signal = peaks[col[0]].values
        peaks = np.where(peaks_signal == 1)[0]

    if isinstance(peaks, dict):
        col = [col for col in list(peaks.keys()) if 'Peaks' in col]
        if len(col) == 0:
            TypeError("NeuroKit error: signal_rate(): wrong type of input ",
                      "provided. Please provide indices of peaks.")
        peaks = peaks[col[0]]


    # Sanity checks.
    if len(peaks) <= 3:
        print("NeuroKit warning: signal_rate(): too few peaks detected to "
              "compute the rate. Returning empty vector.")
        return np.full(desired_length, np.nan)

    return peaks, desired_length