import numpy as np
import pandas as pd

from .eeg_gfp import eeg_gfp


def eeg_diss(eeg, gfp=None, **kwargs):
    """**Global dissimilarity (DISS)**

    Global dissimilarity (DISS) is an index of configuration differences between two electric
    fields, independent of their strength. Like GFP, DISS was first introduced by Lehmann and
    Skrandies (1980). This parameter equals the square root of the mean of the squared differences
    between the potentials measured at each electrode (versus the average reference), each of which
    is first scaled to unitary strength by dividing by the instantaneous GFP.

    Parameters
    ----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    gfp : list
        The Global Field Power (GFP). If ``None``, will be obtained via ``eeg_gfp()``.
    **kwargs
        Optional arguments to be passed into ``nk.eeg_gfp()``.

    Returns
    -------
    np.ndarray
        DISS of each sample point in the data.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      eeg = nk.mne_data("filt-0-40_raw")
      eeg = eeg.set_eeg_reference('average')

      gfp = nk.eeg_gfp(eeg)
      diss = nk.eeg_diss(eeg, gfp=gfp)
      @savefig p_eeg_diss1.png scale=100%
      nk.signal_plot([gfp[0:300], diss[0:300]], standardize=True)
      @suppress
      plt.close()


    References
    ----------
    * Lehmann, D., & Skrandies, W. (1980). Reference-free identification of components of
      checkerboard-evoked multichannel potential fields. Electroencephalography and clinical
      neurophysiology, 48(6), 609-621.

    """
    if isinstance(eeg, pd.DataFrame):
        eeg = eeg.to_numpy(dtype=float).T
    elif not isinstance(eeg, np.ndarray):
        eeg = eeg.get_data()
    else:
        eeg = np.asarray(eeg, dtype=float)

    if eeg.ndim == 3:
        eeg = np.transpose(eeg, (1, 0, 2)).reshape(eeg.shape[1], -1)

    if gfp is None:
        gfp = eeg_gfp(eeg, **kwargs)
    gfp = np.asarray(gfp, dtype=float)

    # Reference-independent potentials relative to instantaneous average across electrodes (Lehmann & Skrandies, 1980)
    centered = eeg - np.mean(eeg, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(gfp != 0, centered / gfp, 0.0)

    diff = np.diff(normalized, axis=1)
    diss = np.sqrt(np.mean(np.square(diff), axis=0))

    # Preserve length
    diss = np.insert(diss, 0, 0.0, axis=0)

    return diss
