# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def _signal_formatpeaks(peaks, desired_length=None, key="Peaks"):
    # Retrieve length.
    if desired_length is None:
        if isinstance(peaks, np.ndarray):
            desired_length = np.max(peaks)
        else:
            desired_length = len(peaks)

    if desired_length < len(peaks):
        raise ValueError("NeuroKit error: _signal_formatpeaks(): 'desired_length' cannot",
                         " be lower than the length of the signal. Please input a greater 'desired_length'.")


    # Attempt to retrieve column.
    if isinstance(peaks, pd.DataFrame):
        col = [col for col in peaks.columns if key in col]
        if len(col) == 0:
            TypeError("NeuroKit error: _signal_formatpeaks(): wrong type of input ",
                      "provided. Please provide indices of peaks.")
        peaks_signal = peaks[col[0]].values
        peaks = np.where(peaks_signal == 1)[0]

    if isinstance(peaks, dict):
        col = [col for col in list(peaks.keys()) if key in col]
        if len(col) == 0:
            TypeError("NeuroKit error: _signal_formatpeaks(): wrong type of input ",
                      "provided. Please provide indices of peaks.")
        peaks = peaks[col[0]]


    return peaks, desired_length
