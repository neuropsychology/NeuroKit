from ..signal import signal_formatpeaks
from .eog_findpeaks import eog_findpeaks


def eog_peaks(veog_cleaned, sampling_rate=None, method="mne", **kwargs):
    """**Locate EOG eye blinks**

    Parameters
    ----------
    veog_cleaned : Union[list, np.array, pd.Series]
        The cleaned vertical EOG channel. Note that it must be positively oriented, i.e., blinks
        must appear as upward peaks.
    sampling_rate : int
        The signal sampling rate (in Hz, i.e., samples/second). Needed for method ``"blinker"`` or
        ``"jammes2008"``.
    method : str
        The peak detection algorithm. Can be one of ``"neurokit"``, ``"mne"`` (requires the MNE
        package to be installed), or ``"brainstorm"`` or ``"blinker"``.
    sampling_rate : int
        The sampling frequency of the EOG signal (in Hz, i.e., samples/second). Needs to be
        supplied if the method to be used is ``"blinker"``, otherwise defaults to ``None``.
    **kwargs
        Other arguments passed to functions.

    Returns
    -------
    array
        Vector containing the samples at which EOG-peaks occur,

    See Also
    --------
    eog_clean

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Get data
      eog_signal = nk.data('eog_100hz')
      eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100)

    * **Example 1:** NeuroKit method

    .. ipython:: python

      signals, info_nk = nk.eog_peaks(eog_cleaned,
                                      sampling_rate=100,
                                      method="neurokit",
                                      threshold=0.33,
                                      show=True)
      @savefig p_eog_peaks1.png scale=100%
      nk.events_plot(info_nk["EOG_Blinks"], eog_cleaned)
      @suppress
      plt.close()

    * **Example 2:** MNE-method

    .. ipython:: python

      signals, info_mne = nk.eog_peaks(eog_cleaned, method="mne")
      @savefig p_eog_peaks2.png scale=100%
      nk.events_plot(info_mne["EOG_Blinks"], eog_cleaned)
      @suppress
      plt.close()

    * **Example 3:** brainstorm method

    .. ipython:: python

      signals, info_brainstorm = nk.eog_peaks(eog_cleaned, method="brainstorm")
      @savefig p_eog_peaks3.png scale=100%
      nk.events_plot(info_brainstorm["EOG_Blinks"], eog_cleaned)
      @suppress
      plt.close()

    * **Example 4:** blinker method

    .. ipython:: python

      signals, info_blinker = nk.eog_peaks(eog_cleaned, sampling_rate=100, method="blinker")
      @savefig p_eog_peaks4.png scale=100%
      nk.events_plot(info_blinker["EOG_Blinks"], eog_cleaned)
      @suppress
      plt.close()


    References
    ----------
    * Agarwal, M., & Sivakumar, R. (2019). Blink: A Fully Automated Unsupervised Algorithm for
      Eye-Blink Detection in EEG Signals. In 2019 57th Annual Allerton Conference on Communication,
      Control, and Computing (Allerton) (pp. 1113-1121). IEEE.
    * Kleifges, K., Bigdely-Shamlo, N., Kerick, S. E., & Robbins, K. A. (2017). BLINKER: automated
      extraction of ocular indices from EEG enabling large-scale analysis. Frontiers in neuroscience, 11, 12.

    """
    peaks = eog_findpeaks(veog_cleaned, sampling_rate=sampling_rate, method=method, **kwargs)
    info = {"EOG_Blinks": peaks}

    instant_peaks = signal_formatpeaks(info, desired_length=len(veog_cleaned), peak_indices=peaks)
    signals = instant_peaks
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    return signals, info
