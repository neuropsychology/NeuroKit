# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def _signal_from_indices(indices, length, value=1):
    """Generates array of 0 and given values at given indices. Used in *_findpeaks to
    transform vectors of peak indices to signal.
    """
    signal = np.zeros(length)

    # Force indices as int
    if isinstance(indices[0], np.float):
        indices = indices[~np.isnan(indices)].astype(np.int)


    if isinstance(value, int) or isinstance(value, float):
        signal[indices] = value
    else:
        if len(value) != len(indices):
            raise ValueError("NeuroKit error: _signal_from_indices(): The number of values "
                             "is different from the number of indices.")
        signal[indices] = value
    return signal







def _signals_from_peakinfo(peak_info, peak_indices, length):
    signals = {}
    for feature in peak_info.keys():
        if "Peak" in str(feature) or "Onset" in str(feature):
            signals[feature] = _signal_from_indices(peak_info[feature], length, 1)
        else:
            signals[feature] = _signal_from_indices(peak_indices, length, peak_info[feature])
    signals = pd.DataFrame(signals)
    return signals
