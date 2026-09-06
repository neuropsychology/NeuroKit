import numpy as np

from ..eeg import eeg_gfp
from ..stats import standardize
from .microstates_peaks import _microstates_sanitize_eeg, microstates_peaks


def microstates_clean(eeg, sampling_rate=None, train="gfp", standardize_eeg=True, normalize=True, gfp_method="l1", **kwargs):
    """**Prepare eeg data for microstates extraction**

    This is mostly a utility function to get the data ready for :func:`.microstates_segment`.

    Parameters
    ----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Defaults to ``None``.
    train : Union[str, int, float]
        Method for selecting the timepoints how which to train the clustering algorithm. Can be
        ``"gfp"`` to use the peaks found in the global field power (GFP). Can be
        ``"all"``, in which case it will select all the datapoints. It can also be a number or a
        ratio, in which case it will select the corresponding number of evenly spread data points.
        For instance, ``train=10`` will select 10 equally spaced datapoints, whereas ``train=0.5``
        will select half the data. See :func:`.microstates_peaks`.
    standardize_eeg : bool
        Standardize (z-score) the data across time using :func:`.nk.standardize`, prior to GFP
        extraction and running k-means algorithm. Defaults to ``True``.
    normalize : bool
        Normalize (divide each data point by the maximum value of the data) across time prior to
        GFP extraction and running k-means algorithm. Defaults to ``True``.
    gfp_method : str
        The GFP extraction method to be passed into :func:`.nk.eeg_gfp`. Can be either ``"l1"``
        (default) or ``"l2"`` to use the L1 or L2 norm.
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

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      eeg = nk.mne_data("filt-0-40_raw")
      eeg, peaks, gfp, info = nk.microstates_clean(eeg, train="gfp")


    See Also
    --------
    .eeg_gfp, microstates_peaks, .microstates_segment

    """
    eeg, sampling_rate, info = _microstates_sanitize_eeg(eeg, sampling_rate=sampling_rate)

    # Normalization
    if standardize_eeg is True:
        standardize_kwargs = {key: kwargs[key] for key in ["robust", "window"] if key in kwargs}
        eeg = standardize(eeg.T, **standardize_kwargs).T

    # Get GFP
    gfp_kwargs = {key: kwargs[key] for key in ["robust", "smooth"] if key in kwargs}
    gfp = eeg_gfp(eeg, sampling_rate=sampling_rate, normalize=normalize, method=gfp_method, **gfp_kwargs)

    # If train is a custom of vector (assume it's the pre-computed peaks)
    if isinstance(train, (list, np.ndarray)):
        peaks = np.asarray(train, dtype=int)
    # Find peaks in the global field power (GFP) or take a given amount of indices
    else:
        peaks = microstates_peaks(eeg, gfp=gfp if train == "gfp" else train, sampling_rate=sampling_rate)

    return eeg, peaks, gfp, info
