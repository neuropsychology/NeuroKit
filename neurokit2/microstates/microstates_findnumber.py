# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .microstates_segment import microstates_segment
from ..stats.cluster_quality import _cluster_quality_sklearn

def microstates_findnumber(eeg, n_max=12, show=False, **kwargs):
    """Estimate optimal number of microstates.

    Estimate the optimal number of microstates using a variety of indices.

    Parameters
    ----------
    seeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    n_max : int
        Maximum number of microstates to try. A higher number leads to a longer process.
    show : bool
        Plot indices normalized on the same scale.
    **kwargs
        Arguments to be passed to ``microstates_segment()``

    Returns
    -------
    DataFrame
        The different quality scores for each number of microstates.

    See Also
    ---------
    microstates_segment

    Examples
    ------------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw").filter(1, 35)
    >>> eeg = nk.eeg_rereference(eeg, 'average')
    >>>
    >>> # Estimate optimal number (takes some time)
    >>> results = nk.microstates_findnumber(eeg, n_max=6, show=True, method="kmod")

    """
    # Retrieve data
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        data = eeg.get_data()
    elif isinstance(eeg, pd.DataFrame):
        data = eeg.values
    else:
        data = eeg.copy()


    # Loop accross number and get indices of fit
    results = []
    for n_microstates in range(2, n_max):
        out = microstates_segment(eeg, n_microstates=n_microstates, **kwargs)

        segmentation = out["Sequence"]
        microstates = out["Microstates"]
#        info = out["Info_algorithm"]
#        sd = out["GFP"]

#        nk.cluster_quality(data.T, segmentation, clusters=microstates, info=info, n_random=10, sd=gfp)
        _, rez = _cluster_quality_sklearn(data.T, segmentation, microstates)

        rez["Score_GEV"] = out["GEV"]
        results.append(pd.DataFrame.from_dict(rez, orient="index").T)

    results = pd.concat(results, axis=0).reset_index(drop=True)

    if show is True:
        normalized = (results - results.min()) / (results.max() - results.min())
        normalized["n_Clusters"] = np.rint(np.arange(2, n_max))
        normalized.columns = normalized.columns.str.replace('Score', 'Normalized')
        normalized.plot(x="n_Clusters")

    return results
