# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..signal import signal_merge


def eda_simulate(duration=10, length=None, sampling_rate=1000, noise=0.01,
                 n_scr=1, drift=-0.01, random_state=42):
    """Simulate Electrodermal Activity (EDA) signal.

    Generate an artificial (synthetic) EDA signal of a given duration and sampling rate.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate, length : int
        The desired sampling rate (in Hz, i.e., samples/second) or the desired
        length of the signal (in samples).
    noise : float
        Noise level (gaussian noise).
    n_scr : int
        Desired number of skin conductance responses (SCRs), i.e., peaks.
    drift : float or list
        The slope of a linear drift of the signal.
    random_state : int
        Seed for the random number generator.

    Returns
    ----------
    array
        Vector containing the EDA signal.

    Examples
    ----------
    >>> import neurokit as nk
    >>> import pandas as pd
    >>>
    >>> eda = nk.eda_simulate(duration=10, n_scr=3)
    >>> nk.signal_plot(eda)

    See Also
    --------
    ecg_simulate, rsp_simulate, emg_simulate, ppg_simulate


    References
    -----------
    - Bach, D. R., Flandin, G., Friston, K. J., & Dolan, R. J. (2010). Modelling event-related skin conductance responses. International Journal of Psychophysiology, 75(3), 349-356.
    """
    # Seed the random generator for reproducible results
    np.random.seed(random_state)

    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate

    eda = np.full(length, 1.0)
    eda += (drift * np.linspace(0, duration, length))
    time = [0, duration]

    start_peaks = np.linspace(0, duration, n_scr, endpoint=False)

    for start_peak in start_peaks:
        relative_time_peak = np.abs(np.random.normal(0, 5, size=1)) + 3.0745
        scr = _eda_simulate_scr(sampling_rate=sampling_rate,
                                      time_peak=relative_time_peak)
        time_scr = [start_peak, start_peak+9]
        if time_scr[0] < 0:
            scr = scr[int(np.round(np.abs(time_scr[0])*sampling_rate))::]
            time_scr[0] = 0
        if time_scr[1] > duration:
            scr = scr[0:int(np.round((duration - time_scr[0])*sampling_rate))]
            time_scr[1] = duration

        eda = signal_merge(signal1=eda, signal2=scr, time1=time, time2=time_scr)

    # Add random (gaussian distributed) noise
    eda += np.random.normal(0, noise, len(eda))
    return eda






def _eda_simulate_scr(sampling_rate=1000,
                      length=None,
                      time_peak=3.0745,
                      rise=0.7013,
                      decay=[3.1487, 14.1257]):
    """Simulate a canonical skin conductance response (SCR)

    Based on `Bach (2010) <https://sourceforge.net/p/scralyze/code/HEAD/tree/branches/version_b2.1.8/scr_bf_crf.m#l24>`_

    Parameters
    -------------
    time_peak : float
        Time to peak.
    rise : float
        Variance of rise defining gaussian.
    decay : list
        Decay constants.

    Examples
    --------
    >>> scr1 = _eda_simulate_canonical(time_peak=3.0745)
    >>> scr2 = _eda_simulate_canonical(time_peak=10)
    >>> pd.DataFrame({"SCR1": scr1, "SCR2": scr2}).plot()
    """
    if length is None:
        length = 9*sampling_rate
    t = np.linspace(sampling_rate/10000, 90, length)

    gt = np.exp(-((t - time_peak)**2)/(2*rise**2))
    ht = np.exp(-t/decay[0]) + np.exp(-t/decay[1])

    ft = np.convolve(gt, ht)
    ft = ft[0:len(t)]
    ft = ft/np.max(ft)
    return ft