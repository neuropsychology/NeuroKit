import numpy as np
import pandas as pd
import scipy.signal

from ..eeg import eeg_gfp


def microstates_peaks(eeg, gfp=None, sampling_rate=None, distance_between=0.01, **kwargs):
    """**Find peaks of stability using the GFP**

    Peaks in the global field power (GFP) are often used to find microstates.

    Parameters
    ----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    gfp : list
        The Global Field Power (GFP). If ``None``, will be obtained via :func:`.eeg_gfp`.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    distance_between : float
        The minimum distance (this value is to be multiplied by the sampling rate) between peaks.
        The default is 0.01, which corresponds to 10 ms (as suggested in the Microstate EEGlab
        toolbox).
    **kwargs
        Additional arguments to be passed to :func:`.eeg_gfp`.

    Returns
    -------
    peaks : array
        The index of the sample where GFP peaks occur.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      eeg = nk.mne_data("filt-0-40_raw")

      gfp = nk.eeg_gfp(eeg)
      peaks1 = nk.microstates_peaks(eeg, distance_between=0.01)
      peaks2 = nk.microstates_peaks(eeg, distance_between=0.05)
      peaks3 = nk.microstates_peaks(eeg, distance_between=0.10)

      @savefig p_microstates_peaks1.png scale=100%
      nk.events_plot([peaks1[peaks1 < 500], peaks2[peaks2 < 500], peaks3[peaks3 < 500]], gfp[0:500])
      @suppress
      plt.close()

    See Also
    --------
    .eeg_gfp

    """
    eeg, sampling_rate, _ = _microstates_sanitize_eeg(eeg, sampling_rate=sampling_rate)

    # Deal with string inputs
    if isinstance(gfp, str):
        if gfp.lower() == "all":
            return np.arange(eeg.shape[1])
        if gfp.lower() == "gfp":
            gfp = None
        else:
            raise ValueError("The `gfp` argument was not understood.")

    # If we don't want to rely on peaks but take uniformly spaced samples
    # (used in microstates_clustering)
    if isinstance(gfp, (int, float, np.integer, np.floating)) and not isinstance(gfp, (bool, np.bool_)):
        if gfp <= 1:  # If fraction
            gfp = int(gfp * eeg.shape[1])
        if not float(gfp).is_integer() or gfp < 1 or gfp > eeg.shape[1]:
            raise ValueError("The number of training samples must be between 1 and the number of timepoints.")
        return np.linspace(0, eeg.shape[1], int(gfp), endpoint=False, dtype=int)

    if gfp is None or gfp is True:
        gfp = eeg_gfp(eeg, sampling_rate=sampling_rate, **kwargs)
    else:
        gfp = np.asarray(gfp)
        if gfp.ndim != 1 or len(gfp) != eeg.shape[1]:
            raise ValueError("The precomputed `gfp` must contain one value per timepoint.")

    if sampling_rate is None:
        raise ValueError(
            "NeuroKit error: microstates_peaks(): `sampling_rate` is required when detecting GFP peaks."
        )

    peaks = _microstates_peaks_gfp(gfp=gfp, sampling_rate=sampling_rate, distance_between=distance_between)

    return peaks


def _microstates_sanitize_eeg(eeg, sampling_rate=None):
    """Return EEG data as a channels-by-timepoints array."""
    info = None
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        sampling_rate = eeg.info["sfreq"]
        info = eeg.info
        eeg = eeg.get_data()
    elif isinstance(eeg, pd.DataFrame):
        eeg = eeg.values

    eeg = np.asarray(eeg)
    if eeg.ndim == 3:
        # MNE Epochs are ordered as epochs, channels, timepoints.
        eeg = eeg.transpose(1, 0, 2).reshape(eeg.shape[1], -1)
    if eeg.ndim != 2:
        raise ValueError("EEG data must have shape (channels, timepoints) or (epochs, channels, timepoints).")

    return eeg, sampling_rate, info


# =============================================================================
# Methods
# =============================================================================
def _microstates_peaks_gfp(gfp=None, sampling_rate=None, distance_between=0.01):
    minimum_separation = int(distance_between * sampling_rate)  # 10 ms (Microstate EEGlab toolbox)
    if minimum_separation == 0:
        minimum_separation = 1

    peaks_gfp, _ = scipy.signal.find_peaks(gfp, distance=minimum_separation)

    # Alternative methods: (doesn't work best IMO)
    #    peaks_gfp = scipy.signal.find_peaks_cwt(gfp, np.arange(minimum_separation, int(0.2 * sampling_rate)))
    #    peaks_gfp = scipy.signal.argrelmax(gfp)[0]

    # Use DISS
    #    diss = nk.eeg_diss(eeg, gfp)
    #    peaks_diss, _ = scipy.signal.find_peaks(diss, distance=minimum_separation)

    return peaks_gfp
