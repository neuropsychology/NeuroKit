# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def eeg_rereference(eeg, reference="average", robust=False, **kwargs):
    """**EEG Rereferencing**

    This function can be used for arrays as well as MNE objects.

    EEG recordings measure differences in electrical potentials between two points, which means the
    signal displayed at any channel is in fact the difference in electrical potential to some other
    recording site. Primarily, this other recording site is the ground electrode, which picks up
    electrical noise that does not reach the other scalp electrodes. Consequently, the voltage
    difference between ground and EEG scalp electrodes is also affected by this noise.

    The idea behind re-referencing is to express the voltage at the EEG scalp channels with respect
    to another, new reference. It can be composed of any recorded channel or an average of several
    channels.

    Parameters
    -----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    reference : str
        See :func:`.mne.set_eeg_reference()`. Can be a string (e.g., 'average', 'lap' for Laplacian
        "reference-free" transformation, i.e., CSD), or a list (e.g., ['TP9', 'TP10'] for mastoid
        reference).
    robust : bool
        Only applied if reference is ``average``. If ``True``, will substract the median instead
        of the mean.
    **kwargs
        Optional arguments to be passed into ``mne.set_eeg_rereference()``.

    Returns
    -------
    object
        The rereferenced raw mne object.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      raw = nk.mne_data("filt-0-40_raw")
      eeg = raw.get_data()

    * **Example 1:** Difference between robust average

    .. ipython:: python

      avg = nk.eeg_rereference(eeg, 'average', robust=False)
      avg_r = nk.eeg_rereference(eeg, 'average', robust=True)

      @savefig p_eeg_rereference1.png scale=100%
      nk.signal_plot([avg[0, 0:1000], avg_r[0, 0:1000]], labels=["Normal", "Robust"])
      @suppress
      plt.close()

    * **Example 2:** Compare the rereferencing of an array vs. the MNE object

    .. ipython:: python

      avg_mne = raw.copy().set_eeg_reference('average', verbose=False)
      @savefig p_eeg_rereference2.png scale=100%
      nk.signal_plot([avg[0, 0:1000], avg_mne.get_data()[0, 0:1000]])
      @suppress
      plt.close()

    * **Example 3:** Difference between average and LAP

    .. ipython:: python

      lap = nk.eeg_rereference(raw, 'lap')
      @savefig p_eeg_rereference3.png scale=100%
      nk.signal_plot(
          [avg_mne.get_data()[0, 0:1000], lap.get_data()[0, 0:1000]],
          standardize=True
      )
      @suppress
      plt.close()

    References
    -----------
    * Trujillo, L. T., Stanfield, C. T., & Vela, R. D. (2017). The effect of electroencephalogram
      (EEG) reference choice on information-theoretic measures of the complexity and integration of
      EEG signals. Frontiers in Neuroscience, 11, 425.

    """
    # If MNE object
    if isinstance(eeg, (pd.DataFrame, np.ndarray)):
        eeg = eeg_rereference_array(eeg, reference=reference, robust=robust)
    else:
        eeg = eeg_rereference_mne(eeg, reference=reference, robust=robust, **kwargs)
    return eeg


# =============================================================================
# Methods
# =============================================================================
def eeg_rereference_array(eeg, reference="average", robust=False):

    # Average reference
    if reference == "average":
        if robust is False:
            eeg = eeg - np.mean(eeg, axis=0, keepdims=True)
        else:
            eeg = eeg - np.median(eeg, axis=0, keepdims=True)
    else:
        raise ValueError(
            "NeuroKit error: eeg_rereference(): Only 'average' rereferencing",
            " is supported for data arrays for now.",
        )

    return eeg


def eeg_rereference_mne(eeg, reference="average", robust=False, **kwargs):

    eeg = eeg.copy()
    if reference == "average" and robust is True:
        # Assigning "custom_ref_applied" to True throws an error with the
        # latest MNE. If this error goes away in the future, we might able to
        # restore this feature.
        # > eeg._data = eeg_rereference_array(eeg._data, reference=reference, robust=robust)
        # > eeg.info["custom_ref_applied"] = True
        raise ValueError(
            "NeuroKit error: eeg_rereference(): 'robust=True' currently not supported for MNE",
            " objects.",
        )
    elif reference in ["lap", "csd"]:
        try:
            import mne

            if mne.__version__ < "0.20":
                raise ImportError
        except ImportError as e:
            raise ImportError(
                "NeuroKit error: eeg_rereference(): the 'mne' module (version > 0.20) is required "
                "for this function to run. Please install it first (`pip install mne`).",
            ) from e
        old_verbosity_level = mne.set_log_level(verbose="WARNING", return_old_level=True)
        eeg = mne.preprocessing.compute_current_source_density(eeg, **kwargs)

        # Reconvert CSD type to EEG (https://github.com/mne-tools/mne-python/issues/11426)
        # channels = np.array(eeg.ch_names)[mne.pick_types(eeg.info, csd=True)]
        # eeg.set_channel_types(dict(zip(channels, ["eeg"] * len(channels))))
        mne.set_log_level(old_verbosity_level)
    else:
        eeg = eeg.set_eeg_reference(reference, verbose=False, **kwargs)

    return eeg
