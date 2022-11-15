# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import find_knee, progress_bar
from ..stats.cluster_quality import _cluster_quality_dispersion
from .microstates_segment import microstates_segment


def microstates_findnumber(
    eeg, n_max=12, method="GEV", clustering_method="kmod", show=False, verbose=True, **kwargs
):
    """**Estimate optimal number of microstates**

    Computes statistical indices useful for estimating the optimal number of microstates using a

    * **Global Explained Variance (GEV)**: measures how similar each EEG sample is to its assigned
      microstate class. The **higher** (closer to 1), the better the segmentation.
    * **Krzanowski-Lai Criterion (KL)**: measures quality of microstate segmentation based on the
      dispersion measure (average distance between samples in the same microstate class); the
      **larger the KL value**, the better the segmentation. Note that KL is not a polarity
      invariant measure and thus might not be a suitable measure of fit for polarity-invariant
      methods such as modified K-means and (T)AAHC.

    Parameters
    ----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    n_max : int
        Maximum number of microstates to try. A higher number leads to a longer process.
    method : str
        The method to use to estimate the optimal number of microstates. Can be "GEV" (the elbow,
        detected using :func:`find_knee`), or "KL" (the location of the maximum value).
    show : bool
        Plot indices normalized on the same scale.
    verbose : bool
        Print progress bar.
    **kwargs
        Arguments to be passed to :func:`.microstates_segment`

    Returns
    -------
    int
        Optimal number of microstates.
    DataFrame
        The different quality scores for each number of microstates.

    See Also
    ---------
    microstates_segment

    Examples
    ------------
    .. ipython:: python

      import neurokit2 as nk

      eeg = nk.mne_data("filt-0-40_raw").crop(0, 5)

      # Estimate optimal number
      @savefig p_microstates_findnumber1.png scale=100%
      n_clusters, results = nk.microstates_findnumber(eeg, n_max=8, show=True)
      @suppress
      plt.close()

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
    for idx, n_microstates in progress_bar(range(2, n_max + 1), verbose=verbose):

        out = microstates_segment(
            eeg, n_microstates=n_microstates, method=clustering_method, **kwargs
        )

        segmentation = out["Sequence"]
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
            results[idx - 1]["KL_Criterion"] = np.abs(dispersion_diff_previous / dispersion_diff)
        # Update for next round
        dispersion_previous = dispersion_current.copy()
        dispersion_diff_previous = dispersion_diff.copy()

        results.append(rez)

    results = pd.DataFrame(results)

    # Estimate optimal number
    if method == "KL":
        n_clusters = int(np.argmax(results["KL_Criterion"]) + 2)
    else:
        n_clusters = find_knee(results["Score_GEV"], np.rint(np.arange(2, n_max + 1)))

    if show is True:
        normalized = (results - results.min()) / (results.max() - results.min())
        normalized["n_Clusters"] = np.rint(np.arange(2, n_max + 1))
        normalized.columns = normalized.columns.str.replace("Score", "Normalized")
        normalized.plot(x="n_Clusters")
        plt.axvline(n_clusters, color="red", linestyle="--", label=f"Method: {method}")
        plt.legend()
        plt.xticks(np.rint(np.arange(2, n_max + 1)))
        plt.xlabel("Number of microstates")
        plt.ylabel("Normalized score")
        plt.title("Optimal number of microstates")

    return n_clusters, results
