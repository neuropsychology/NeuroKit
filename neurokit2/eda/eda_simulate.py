# -*- coding: utf-8 -*-
import numpy as np

from ..signal import signal_distort, signal_merge


def eda_simulate(
    duration=10, length=None, sampling_rate=1000, noise=0.01, scr_number=1, drift=-0.01, random_state=None
):
    """Simulate Electrodermal Activity (EDA) signal.

    Generate an artificial (synthetic) EDA signal of a given duration and sampling rate.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second). Defaults to 1000Hz.
    length : int
        The desired length of the signal (in samples). Defaults to None.
    noise : float
        Noise level (amplitude of the laplace noise). Defaults to 0.01.
    scr_number : int
        Desired number of skin conductance responses (SCRs), i.e., peaks. Defaults to 1.
    drift : float or list
        The slope of a linear drift of the signal. Defaults to -0.01.
    random_state : int
        Seed for the random number generator. Defaults to None.

    Returns
    ----------
    array
        Vector containing the EDA signal.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> eda = nk.eda_simulate(duration=10, scr_number=3)
    >>> fig = nk.signal_plot(eda)
    >>> fig #doctest: +SKIP

    See Also
    --------
    ecg_simulate, rsp_simulate, emg_simulate, ppg_simulate


    References
    -----------
    - Bach, D. R., Flandin, G., Friston, K. J., & Dolan, R. J. (2010). Modelling event-related skin
      conductance responses. International Journal of Psychophysiology, 75(3), 349-356.

    """
    # Seed the random generator for reproducible results
    np.random.seed(random_state)

    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate

    eda = np.full(length, 1.0)
    eda += drift * np.linspace(0, duration, length)
    time = [0, duration]

    start_peaks = np.linspace(0, duration, scr_number, endpoint=False)

    for start_peak in start_peaks:
        relative_time_peak = np.abs(np.random.normal(0, 5, size=1)) + 3.0745
        scr = _eda_simulate_scr(sampling_rate=sampling_rate, time_peak=relative_time_peak)
        time_scr = [start_peak, start_peak + 9]
        if time_scr[0] < 0:
            scr = scr[int(np.round(np.abs(time_scr[0]) * sampling_rate)) : :]
            time_scr[0] = 0
        if time_scr[1] > duration:
            scr = scr[0 : int(np.round((duration - time_scr[0]) * sampling_rate))]
            time_scr[1] = duration

        eda = signal_merge(signal1=eda, signal2=scr, time1=time, time2=time_scr)

    # Add random noise
    if noise > 0:
        eda = signal_distort(
            eda,
            sampling_rate=sampling_rate,
            noise_amplitude=noise,
            noise_frequency=[5, 10, 100],
            noise_shape="laplace",
            silent=True,
        )
    # Reset random seed (so it doesn't affect global)
    np.random.seed(None)
    return eda


def _eda_simulate_scr(sampling_rate=1000, length=None, time_peak=3.0745, rise=0.7013, decay=[3.1487, 14.1257]):
    """Simulate a canonical skin conductance response (SCR)

    Based on `Bach (2010)
    <https://sourceforge.net/p/scralyze/code/HEAD/tree/branches/version_b2.1.8/scr_bf_crf.m#l24>`_

    Parameters
    -----------
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second). Defaults to 1000Hz.
    length : int
        The desired length of the signal (in samples). Defaults to None.
    time_peak : float
        Time to peak.
    rise : float
        Variance of rise defining gaussian.
    decay : list
        Decay constants.

    Returns
    ----------
    array
        Vector containing the SCR signal.

    Examples
    --------
    >>> # scr1 = _eda_simulate_scr(time_peak=3.0745)
    >>> # scr2 = _eda_simulate_scr(time_peak=10)
    >>> # pd.DataFrame({"SCR1": scr1, "SCR2": scr2}).plot()

    """
    if length is None:
        length = 9 * sampling_rate
    t = np.linspace(sampling_rate / 10000, 90, length)

    gt = np.exp(-((t - time_peak) ** 2) / (2 * rise ** 2))
    ht = np.exp(-t / decay[0]) + np.exp(-t / decay[1])  # pylint: disable=E1130

    ft = np.convolve(gt, ht)
    ft = ft[0 : len(t)]
    ft = ft / np.max(ft)
    return ft


def _eda_simulate_bateman(sampling_rate=1000, t1=0.75, t2=2):
    """Generates the bateman function:

    :math:`b = e^{-t/T1} - e^{-t/T2}`

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency
    t1 : float
        Defaults to 0.75.
    t2 : float
        Defaults to 2.

        Parameters of the bateman function
    Returns
    -------
    bateman : array
        The bateman function

    Examples
    ----------
    >>> # bateman = _eda_simulate_bateman()
    >>> # nk.signal_plot(bateman)

    """

    idx_T1 = t1 * sampling_rate
    idx_T2 = t2 * sampling_rate
    len_bat = idx_T2 * 10
    idx_bat = np.arange(len_bat)
    bateman = np.exp(-idx_bat / idx_T2) - np.exp(-idx_bat / idx_T1)

    # normalize
    bateman = sampling_rate * bateman / np.sum(bateman)
    return bateman
