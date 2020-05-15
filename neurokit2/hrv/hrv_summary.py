# -*- coding: utf-8 -*-
import pandas as pd
from . import hrv_time, hrv_frequency, hrv_nonlinear


def hrv_summary(peaks, sampling_rate=1000, show=False):
    """ Computes indices of Heart Rate Variability (HRV).

    Note that a minimum recording is recommended for somenindices to be
    meaninful. For instance, 1, 2 and 5 minutes of good signal are the
    recomended minimums for HF, LF and LF/HF, respectively.

    Parameters
    ----------
    heart_period : array
        Array containing the heart period as returned by `signal_period()`.
    peaks : dict
        The samples at which the peaks occur. Returned by `ecg_peaks()` or
        `ppg_peaks`. Defaults to None.
    sampling_rate : int, optional
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    show : bool, optional
        If True, will return a Poincaré plot, a scattergram, which plots each
        RR interval against the next successive one. The ellipse centers around
        the average RR interval. Defaults to False.

    Returns
    -------
    DataFrame
        Contains HRV metrics from three domains:
        - frequency (for details see hrv_frequency)
        - time (for details see hrv_time)
        - non-linear (for details see hrv_nonlinear)
    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_frequency, hrv_time, hrv_nonlinear

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=240, sampling_rate=1000)
    >>> ecg, info = nk.ecg_process(ecg, sampling_rate=1000)
    >>> hrv = nk.ecg_hrv(ecg, sampling_rate=1000, show=True)
    >>> hrv
    >>> hrv[["HRV_HF"]]
    >>>
    >>> ecg = nk.ecg_simulate(duration=240, sampling_rate=200)
    >>> ecg, info = nk.ecg_process(ecg, sampling_rate=200)
    >>> hrv = nk.ecg_hrv(ecg, sampling_rate=200)

    References
    ----------
    - Stein, P. K. (2002). Assessing heart rate variability from real-world
      Holter reports. Cardiac electrophysiology review, 6(3), 239-244.
    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. Frontiers in public health, 5, 258.
    """
    # Get indices
    hrv = {}    # initialize empty dict
    hrv.update(hrv_time(peaks, sampling_rate=sampling_rate, show=show))
    hrv.update(hrv_frequency(peaks, sampling_rate=sampling_rate, show=show))
    hrv.update(hrv_nonlinear(peaks, sampling_rate=sampling_rate, show=show))

    hrv = pd.DataFrame.from_dict(hrv, orient='index').T.add_prefix("HRV_")

    return hrv
