# -*- coding: utf-8 -*-

from ..signal import signal_formatpeaks
from .rsp_findpeaks import rsp_findpeaks
from .rsp_fixpeaks import rsp_fixpeaks


def rsp_peaks(rsp_cleaned, sampling_rate=1000, method="khodadad2018", **kwargs):
    """**Identify extrema in a respiration (RSP) signal**

    This function runs :func:`.rsp_findpeaks` and :func:`.rsp_fixpeaks` to identify and process
    peaks (exhalation onsets) and troughs (inhalation onsets) in a preprocessed respiration signal
    using different sets of parameters, such as:

    * **khodad2018**: Uses the parameters in Khodadad et al. (2018).
    * **biosppy**: Uses the parameters in `BioSPPy's <https://github.com/PIA-Group/BioSPPy>`_
      ``resp()`` function.
    * **scipy** Uses the `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
      peak-detection function.

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by :func:`.rsp_clean`.
    sampling_rate : int
        The sampling frequency of :func:`.rsp_cleaned` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of ``"khodadad2018"`` (default), ``"biosppy"``
        or ``"scipy"``.
    **kwargs
        Other arguments to be passed to the different peak finding methods. See
        :func:`.rsp_findpeaks`.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the samples at which peaks
        (exhalation onsets) and troughs (inhalation onsets) occur, accessible with the keys
        ``"RSP_Peaks"``, and ``"RSP_Troughs"``, respectively, as well as the signals' sampling rate.
    peak_signal : DataFrame
        A DataFrame of same length as the input signal in which occurrences of peaks (exhalation
        onsets) and troughs (inhalation onsets) are marked as "1" in lists of zeros with the same
        length as :func:`.rsp_cleaned`. Accessible with the keys ``"RSP_Peaks"`` and
        ``"RSP_Troughs"`` respectively.


    See Also
    --------
    rsp_clean, signal_rate, rsp_findpeaks, rsp_fixpeaks, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import pandas as pd

      rsp = nk.rsp_simulate(duration=30, respiratory_rate=15)
      cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
      peak_signal, info = nk.rsp_peaks(cleaned, sampling_rate=1000)

      data = pd.concat([pd.DataFrame({"RSP": rsp}), peak_signal], axis=1)
      @savefig p_rsp_peaks1.png scale=100%
      fig = nk.signal_plot(data)
      @suppress
      plt.close()

    References
    ----------
    * Khodadad, D., Nordebo, S., Müller, B., Waldmann, A., Yerworth, R., Becher, T., ... & Bayford,
      R. (2018). Optimized breath detection algorithm in electrical impedance tomography.
      Physiological measurement, 39(9), 094001.

    """
    info = rsp_findpeaks(rsp_cleaned, sampling_rate=sampling_rate, method=method, **kwargs)
    info = rsp_fixpeaks(info)
    peak_signal = signal_formatpeaks(
        info, desired_length=len(rsp_cleaned), peak_indices=info["RSP_Peaks"]
    )

    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    return peak_signal, info
