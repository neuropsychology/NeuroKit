# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .microstates_peaks import microstates_peaks
from ..eeg import eeg_gfp
from ..stats import standardize


def microstates_clean(eeg, sampling_rate=None, train="gfp", standardize_eeg=True, normalize=True, gfp_method="l1", **kwargs):
    """Prepare eeg data for microstates extraction.

    Parameters
    ----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Defaults to None.
    train : Union[str, int, float]
        Method for selecting the timepoints how which to train the clustering algorithm. Can be
        'gfp' to use the peaks found in the Peaks in the global field power. Can be 'all', in which
        case it will select all the datapoints. It can also be a number or a ratio, in which case
        it will select the corresponding number of evenly spread data points. For instance,
        ``train=10`` will select 10 equally spaced datapoints, whereas ``train=0.5`` will select
        half the data. See ``microstates_peaks()``.
    standardize_eeg : bool
        Standardize (z-score) the data across time using ``nk.standardize()``, prior to GFP extraction and
        running k-means algorithm. Defaults to True.
    normalize : bool
        Normalize (divide each data point by the maximum value of the data) across time prior to GFP extraction
        and running k-means algorithm. Defaults to True.
    gfp_method : str
        The GFP extraction method to be passed into ``nk.eeg_gfp()``. Can be either 'l1' (default) or 'l2'
        to use the L1 or L2 norm.
    **kwargs : optional
        Other arguments.

    Returns
    -------
    eeg : array
        The eeg data which has a shape of channels x samples.
    peaks : array
        The index of the sample where GFP peaks occur.
    gfp : array
        The global field power of each sample point in the data.
    info : dict
        Other information pertaining to the eeg raw object.

    See Also
    --------
    eeg_gfp, microstates_peaks

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
    gfp = eeg_gfp(eeg, sampling_rate=sampling_rate, normalize=normalize, method=gfp_method, **kwargs)

    # Find peaks in the global field power (GFP) or take a given amount of indices
    if train == "gfp":
        train = gfp
    peaks = microstates_peaks(eeg, gfp=train, sampling_rate=sampling_rate, **kwargs)

    return eeg, peaks, gfp, info
