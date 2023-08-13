from ..signal import signal_formatpeaks
from .ppg_findpeaks import ppg_findpeaks


def ppg_peaks(ppg_cleaned, sampling_rate=1000, method="elgendi", **kwargs):
    """**Find systolic peaks in a photoplethysmogram (PPG) signal**

    Find the peaks in an PPG signal using the specified method. You can pass an unfiltered PPG
    signals as input, but typically a filtered PPG (cleaned using ``ppg_clean()``) will result in
    better results.

    .. note::

      Please help us improve the methods' documentation by adding a small description.


    Parameters
    ----------
    ppg_cleaned : Union[list, np.array, pd.Series]
        The cleaned PPG channel as returned by ``ppg_clean()``.
    sampling_rate : int
        The sampling frequency of ``ppg_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"elgendi"``, ``"bishop"``. The default is
        ``"elgendi"``.
    **kwargs
        Additional keyword arguments, usually specific for each method.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as the input signal in which occurrences of R-peaks marked as
        ``1`` in a list of zeros with the same length as ``ppg_cleaned``. Accessible with the keys
        ``"PPG_Peaks"``.
    info : dict
        A dictionary containing additional information, in this case the samples at which R-peaks
        occur, accessible with the key ``"PPG_Peaks"``, as well as the signals' sampling rate,
        accessible with the key ``"sampling_rate"``.

    See Also
    --------
    ppg_clean, ppg_segment, .signal_fixpeaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      ppg = nk.ppg_simulate(heart_rate=75, duration=20, sampling_rate=50)
      ppg_clean = nk.ppg_clean(ppg, sampling_rate=50)

      # Default method (Elgendi et al., 2013)
      @savefig p_ppg_peaks1.png scale=100%
      peaks, info = nk.ppg_peaks(ppg_clean, sampling_rate=100, method="elgendi", show=True)
      @suppress
      plt.close()
      peaks_idx = info["PPG_Peaks"]

      # Method by Bishop et al., (2018)
      @savefig p_ppg_peaks2.png scale=100%
      peaks = nk.ppg_peaks(ppg_clean, sampling_rate=100, method="bishop", show=True)
      @suppress
      plt.close()

    References
    ----------
    * Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D. (2013). Systolic peak
      detection in acceleration photoplethysmograms measured from emergency responders in tropical
      conditions. PloS one, 8(10), e76585.
    * Bishop, S. M., & Ercole, A. (2018). Multi-scale peak and trough detection optimised for
      periodic and quasi-periodic neuroscience data. In Intracranial Pressure & Neuromonitoring XVI
      (pp. 189-195). Springer International Publishing.

    """
    peaks = ppg_findpeaks(
        ppg_cleaned, sampling_rate=sampling_rate, method=method, **kwargs
    )

    instant_peaks = signal_formatpeaks(
        peaks, desired_length=len(ppg_cleaned), peak_indices=peaks
    )
    signals = instant_peaks
    info = peaks
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    return signals, info
