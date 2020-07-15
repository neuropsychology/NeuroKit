# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .microstates_peaks import microstates_peaks
from ..eeg import eeg_gfp
from ..stats import standardize


def _microstates_prepare_data(eeg, sampling_rate=None, select="gfp", standardize_eeg=True, **kwargs):
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

    # Find peaks in the global field power (GFP)
    if select == "gfp":
        gfp = eeg_gfp(eeg, sampling_rate=sampling_rate, normalize=True, method="l1", **kwargs)

    # Get Peaks
    peaks = microstates_peaks(eeg, gfp, sampling_rate=sampling_rate, **kwargs)

    # Get GFP regardless of the selection
    if select != "gfp":
        gfp = eeg_gfp(eeg, sampling_rate=sampling_rate, normalize=True, method="l1", **kwargs)

    return eeg, peaks, gfp, info