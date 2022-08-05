# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from ..misc import NeuroKitWarning
from ..stats import rescale


def rsp_rvt(
    rsp_signal,
    sampling_rate=1000,
    boundaries=[2.0, 1 / 30],
    iterations=10,
    show=False,
    silent=False,
):
    """**Respiratory Volume per Time (RVT)**

    Computes Respiratory Volume per Time (RVT). It can be used to improve physiological noise
    correction in functional magnetic resonance imaging (fMRI).

    Parameters
    ----------
    rsp_signal : array
        Array containing the respiratory rate, produced by :func:`.signal_rate`.
    sampling_rate : int, optional
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    boundaries : list
        Lower and upper limit of (humanly possible) breath frequency in Hertz.
    iterations : int
        Amount of phase refinement estimates to remove high frequencies. Synthetic samples often
        take less than 3.
    show : bool
        If ``True``, will return a simple plot of the RVT (with the re-scaled original RSP signal).
    silent : bool
        If ``True``, warnings will not be printed.

    Returns
    -------
    array
        Array containing the current RVT at every timestep.

    See Also
    --------
    signal_rate, rsp_peaks, rsp_process, rsp_clean

    Examples
    --------
    .. ipython:: python

        import neurokit2 as nk

        rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
        rsp = nk.rsp_clean(rsp)
        nk.rsp_rvt(rsp, show=True)


    References
    ----------
    * Harrison, S. J., Bianchi, S., Heinzle, J., Stephan, K. E., Iglesias, S., & Kasper, L. (2021).
      A Hilbert-based method for processing respiratory timeseries. Neuroimage, 230, 117787.
    """
    # low-pass filter at not too far above breathing-rate to remove high-frequency noise
    n_pad = int(np.ceil(10 * sampling_rate))

    d = scipy.signal.iirfilter(
        N=10, Wn=0.75, btype="lowpass", analog=False, output="sos", fs=sampling_rate
    )
    fr_lp = scipy.signal.sosfiltfilt(d, np.pad(rsp_signal, n_pad, "symmetric"))
    fr_lp = fr_lp[n_pad : (len(fr_lp) - n_pad)]

    # derive Hilbert-transform
    fr_filt = fr_lp
    fr_mag = abs(scipy.signal.hilbert(fr_filt))

    for _ in range(iterations):
        # analytic signal to phase
        fr_phase = np.unwrap(np.angle(scipy.signal.hilbert(fr_filt)))
        # Remove any phase decreases that may occur
        # Find places where the gradient changes sign
        # maybe can be changed with signal.signal_zerocrossings
        fr_phase_diff = np.diff(np.sign(np.gradient(fr_phase)))
        decrease_inds = np.argwhere(fr_phase_diff < 0)
        increase_inds = np.append(np.argwhere(fr_phase_diff > 0), [len(fr_phase) - 1])
        for n_max in decrease_inds:
            # Find value of `fr_phase` at max and min:
            fr_max = fr_phase[n_max].squeeze()
            n_min, fr_min = _rsp_rvt_find_min(increase_inds, fr_phase, n_max, silent)
            if n_min is None:
                # There is no finishing point to the interpolation at the very end
                continue
            # Find where `fr_phase` passes `fr_min` for the first time
            n_start = np.argwhere(fr_phase > fr_min)
            if len(n_start) == 0:
                n_start = n_max
            else:
                n_start = n_start[0].squeeze()
            # Find where `fr_phase` exceeds `fr_max` for the first time
            n_end = np.argwhere(fr_phase < fr_max)
            if len(n_end) == 0:
                n_end = n_min
            else:
                n_end = n_end[-1].squeeze()

            # Linearly interpolate from n_start to n_end
            fr_phase[n_start:n_end] = np.linspace(fr_min, fr_max, num=n_end - n_start).squeeze()
        # Filter out any high frequencies from phase-only signal
        fr_filt = scipy.signal.sosfiltfilt(d, np.pad(np.cos(fr_phase), n_pad, "symmetric"))
        fr_filt = fr_filt[n_pad : (len(fr_filt) - n_pad)]
    # Keep phase only signal as reference
    fr_filt = np.cos(fr_phase)

    # Make RVT

    # Low-pass filter to remove within_cycle changes
    # Note factor of two is for compatability with the common definition of RV
    # as the difference between max and min inhalation (i.e. twice the amplitude)
    d = scipy.signal.iirfilter(
        N=10, Wn=0.2, btype="lowpass", analog=False, output="sos", fs=sampling_rate
    )
    fr_rv = 2 * scipy.signal.sosfiltfilt(d, np.pad(fr_mag, n_pad, "symmetric"))
    fr_rv = fr_rv[n_pad : (len(fr_rv) - n_pad)]
    fr_rv[fr_rv < 0] = 0

    # Breathing rate is instantaneous frequency
    fr_if = sampling_rate * np.gradient(fr_phase) / (2 * np.pi)
    fr_if = scipy.signal.sosfiltfilt(d, np.pad(fr_if, n_pad, "symmetric"))
    fr_if = fr_if[n_pad : (len(fr_if) - n_pad)]
    # remove in-human patterns, since both limits are in Hertz, the upper_limit is lower
    fr_if = np.clip(fr_if, boundaries[1], boundaries[0])

    # RVT = magnitude * breathing rate
    rvt = np.multiply(fr_rv, fr_if)

    # Downsampling is not needed as we assume always the same sampling rate and operate always in the same sampling rate
    if show:
        _rsp_rvt_plot(rvt, rsp_signal, sampling_rate)
    return rvt


def _rsp_rvt_find_min(increase_inds, fr_phase, smaller_index, silent):
    bigger_n_max = np.argwhere(increase_inds > smaller_index)
    if len(bigger_n_max) == 0:
        if not silent:
            warn(
                "rsp_rvt(): There is no next increasing point as end point for the interpolation. "
                "Interpolation is skipped for this case.",
                category=NeuroKitWarning,
            )
        return None, None
    bigger_n_max = bigger_n_max[0].squeeze()
    n_min = increase_inds[bigger_n_max]
    fr_min = fr_phase[n_min].squeeze()
    # Sometime fr_min is the same as n_max and it caused problems
    if fr_phase[smaller_index].squeeze() < fr_min:
        if not silent:
            warn(
                "rsp_rvt(): The next bigger increasing index has a bigger value than the chosen decreasing index, "
                "this might be due to very small/noisy breaths or saddle points. "
                "Interpolation is skipped for this case.",
                category=NeuroKitWarning,
            )
        return None, None
    return n_min, fr_min


def _rsp_rvt_plot(rvt, rsp_signal, sampling_rate):
    plt.figure(figsize=(12, 12))
    plt.title("Respiratory Volume per Time (RVT)")
    plt.xlabel("Time [s]")
    plt.plot(rescale(rsp_signal, to=[np.nanmin(rvt), np.nanmax(rvt)]), label="RSP", color="#CFD8DC")
    plt.plot(rvt, label="RVT", color="#00BCD4")
    plt.legend()
    tickpositions = plt.gca().get_xticks()[1:-1]
    plt.xticks(tickpositions, [tickposition / sampling_rate for tickposition in tickpositions])
