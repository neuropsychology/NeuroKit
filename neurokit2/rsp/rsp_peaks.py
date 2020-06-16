# -*- coding: utf-8 -*-

from ..signal import signal_formatpeaks
from .rsp_findpeaks import rsp_findpeaks
from .rsp_fixpeaks import rsp_fixpeaks


def rsp_peaks(rsp_cleaned, sampling_rate=1000, method="khodadad2018", amplitude_min=0.3):
    """Identify extrema in a respiration (RSP) signal.

    This function `rsp_findpeaks()` and `rsp_fixpeaks` to identify and process inhalation peaks and
    exhalation troughs in a preprocessed respiration signal using different sets of parameters, such as:

    - `Khodadad et al. (2018)
    <https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta>`_

    - `BioSPPy
    <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py>`_

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by `rsp_clean()`.
    sampling_rate : int
        The sampling frequency of 'rsp_cleaned' (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of "khodadad2018" (default) or "biosppy".
    amplitude_min : float
        Only applies if method is "khodadad2018". Extrema that have a vertical distance smaller than
        (outlier_threshold * average vertical distance) to any direct neighbour are removed as false
        positive outliers. i.e., outlier_threshold should be a float with positive sign (the default is
        0.3). Larger values of outlier_threshold correspond to more conservative thresholds
        (i.e., more extrema removed as outliers).

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the samples at which inhalation
        peaks and exhalation troughs occur, accessible with the keys "RSP_Peaks", and "RSP_Troughs", respectively.
    peak_signal : DataFrame
        A DataFrame of same length as the input signal in which occurences of inhalation peaks and
        exhalation troughs are marked as "1" in lists of zeros with the same length as `rsp_cleaned`.
        Accessible with the keys "RSP_Peaks" and "RSP_Troughs" respectively.


    See Also
    --------
    rsp_clean, signal_rate, rsp_findpeaks, rsp_fixpeaks, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> rsp = nk.rsp_simulate(duration=30, respiratory_rate=15)
    >>> cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> peak_signal, info = nk.rsp_peaks(cleaned, sampling_rate=1000)
    >>>
    >>> data = pd.concat([pd.DataFrame({"RSP": rsp}), peak_signal], axis=1)
    >>> fig = nk.signal_plot(data)
    >>> fig #doctest: +SKIP

    """
    info = rsp_findpeaks(rsp_cleaned, sampling_rate=sampling_rate, method=method, amplitude_min=amplitude_min)
    info = rsp_fixpeaks(info)
    peak_signal = signal_formatpeaks(info, desired_length=len(rsp_cleaned), peak_indices=info["RSP_Peaks"])

    return peak_signal, info
