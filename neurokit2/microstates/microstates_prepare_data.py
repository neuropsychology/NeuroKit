# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .microstates_peaks import microstates_peaks
from ..eeg import eeg_gfp
from ..stats import standardize


def _microstates_prepare_data(eeg, sampling_rate=None, train="gfp", standardize_eeg=True, **kwargs):
    """
    """
    # If MNE object
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        sampling_rate = eeg.info["sfreq"]
        info = eeg.info
        eeg = eeg.get_data()
    else:
        info = None

    # Normalization
    if standardize_eeg is True:
        eeg = standardize(eeg, **kwargs)

    # Get GFP
    gfp = eeg_gfp(eeg, sampling_rate=sampling_rate, normalize=True, method="l1", **kwargs)

    # Find peaks in the global field power (GFP) or take a given amount of indices
    if train == "gfp":
        train = gfp
    peaks = microstates_peaks(eeg, gfp=train, sampling_rate=sampling_rate, **kwargs)

    return eeg, peaks, gfp, info