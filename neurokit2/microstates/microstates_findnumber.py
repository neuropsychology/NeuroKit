# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..stats.cluster_quality import _cluster_quality_dispersion
from .microstates_segment import microstates_segment


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
    >>> eeg = nk.mne_data("filt-0-40_raw").filter(1, 35)  #doctest: +ELLIPSIS
    Filtering raw data ...
    >>> eeg = nk.eeg_rereference(eeg, 'average')
    >>>
    >>> # Estimate optimal number (currently comment out due to memory error)
    >>> # results = nk.microstates_findnumber(eeg, n_max=4, show=True, method="kmod")

    """
    # Retrieve data
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        data = eeg.get_data()
    elif isinstance(eeg, pd.DataFrame):
        data = eeg.values
    else:
        data = eeg.copy()

    # Loop accross number and get indices of fit
    n_channel, _ = data.shape
    dispersion_previous = np.nan
    dispersion_diff_previous = np.nan
    results = []
    for idx, n_microstates in enumerate(range(2, n_max + 1)):
        print(idx, n_microstates)
        out = microstates_segment(eeg, n_microstates=n_microstates)

        segmentation = out["Sequence"]
        #        info = out["Info_algorithm"]
        #        sd = out["GFP"]

        #        nk.cluster_quality(data.T, segmentation, clusters=microstates, info=info, n_random=10, sd=gfp)

        #        _, rez = _cluster_quality_sklearn(data.T, segmentation, microstates)
        rez = {}

        rez["Score_GEV"] = out["GEV"]

        # Dispersion
        dispersion = _cluster_quality_dispersion(
            data.T, clustering=segmentation, n_clusters=n_microstates
        )
        # Dispersion(k)

        dispersion_current = dispersion * n_microstates ** (2 / n_channel)
        # dispersion_dff(k) = dispersion(k-1) - dispersion(k)
        dispersion_diff = dispersion_previous - dispersion_current

        # Calculate KL criterion
        # KL(k) = abs(dispersion_diff(k) / dispersion_diff(k+1))
        rez["KL_Criterion"] = np.nan
        if idx not in [0]:
            results[idx - 1]["KL_Criterion"] = np.abs(
                dispersion_diff_previous / dispersion_diff
            )
        # Update for next round
        dispersion_previous = dispersion_current.copy()
        dispersion_diff_previous = dispersion_diff.copy()

        results.append(rez)
    results = pd.DataFrame(results)
    #        results.append(pd.DataFrame.from_dict(rez, orient="index").T)
    #    results = pd.concat(results, axis=0).reset_index(drop=True)

    if show is True:
        normalized = (results - results.min()) / (results.max() - results.min())
        normalized["n_Clusters"] = np.rint(np.arange(2, n_max))
        normalized.columns = normalized.columns.str.replace("Score", "Normalized")
        normalized.plot(x="n_Clusters")

    return results
