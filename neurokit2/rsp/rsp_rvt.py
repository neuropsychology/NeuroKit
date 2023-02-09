# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from ..misc import NeuroKitWarning
from ..signal import signal_interpolate
from ..stats import rescale
from .rsp_clean import rsp_clean
from .rsp_peaks import rsp_findpeaks


def rsp_rvt(
    rsp_signal,
    sampling_rate=1000,
    method="power2020",
    boundaries=[2.0, 1 / 30],
    iterations=10,
    show=False,
    silent=False,
    **kwargs
):
    """**Respiratory Volume per Time (RVT)**

    Computes Respiratory Volume per Time (RVT). RVT is the product of respiratory volume and
    breathing rate. RVT can be used to identify the global fMRI confounds of breathing, which is
    often considered noise.

    Parameters
    ----------
    rsp_signal : array
        Array containing the respiratory rate, produced by :func:`.signal_rate`.
    sampling_rate : int, optional
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    method: str, optional
        The rvt method to apply. Can be one of  ``"power2020"`` (default), ``"harrison2021"`` or
        ``"birn2006"``.
    boundaries : list, optional
        Only applies if method is ``"harrison"``. Lower and upper limit of (humanly possible)
        breath frequency in Hertz.
    iterations : int, optional
        Only applies if method is ``"harrison"``. Amount of phase refinement estimates
        to remove high frequencies. Synthetic samples often take less than 3.
    show : bool, optional
        If ``True``, will return a simple plot of the RVT (with the re-scaled original RSP signal).
    silent : bool, optional
        If ``True``, warnings will not be printed.
    **kwargs
        Arguments to be passed to the underlying peak detection algorithm.

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

      rsp = nk.rsp_simulate(duration=60, random_state=1)

      @savefig p_rsp_rvt1.png scale=100%
      nk.rsp_rvt(rsp, method="power2020", show=True)
      @suppress
      plt.close()

      @savefig p_rsp_rvt2.png scale=100%
      nk.rsp_rvt(rsp, method="harrison2021", show=True)
      @suppress
      plt.close()

      @savefig p_rsp_rvt3.png scale=100%
      nk.rsp_rvt(rsp, method="birn2006", show=True)
      @suppress
      plt.close()

    References
    ----------
    * Birn, R. M., Diamond, J. B., Smith, M. A., & Bandettini, P. A. (2006). Separating
      respiratory-variation-related fluctuations from neuronal-activity-related fluctuations in
      fMRI. Neuroimage, 31(4), 1536-1548.
    * Power, J. D., Lynch, C. J., Dubin, M. J., Silver, B. M., Martin, A., & Jones, R. M. (2020).
      Characteristics of respiratory measures in young adults scanned at rest, including systematic
      changes and "missed" deep breaths. Neuroimage, 204, 116234.
    * Harrison, S. J., Bianchi, S., Heinzle, J., Stephan, K. E., Iglesias, S., & Kasper, L. (2021).
      A Hilbert-based method for processing respiratory timeseries. Neuroimage, 230, 117787.
    """
    method = method.lower()  # remove capitalised letters
    if method in ["harrison", "harrison2021"]:
        rvt = _rsp_rvt_harrison(
            rsp_signal,
            sampling_rate=sampling_rate,
            silent=silent,
            boundaries=boundaries,
            iterations=iterations,
        )
    elif method in ["birn", "birn2006"]:
        rvt = _rsp_rvt_birn(rsp_signal, sampling_rate=sampling_rate, silent=silent, **kwargs)
    elif method in ["power", "power2020"]:
        rvt = _rsp_rvt_power(rsp_signal, sampling_rate=sampling_rate, silent=silent, **kwargs)
    else:
        raise ValueError("NeuroKit error: rsp_rvt(): 'method' should be one of 'birn', 'power' or 'harrison'.")
    if show:
        _rsp_rvt_plot(rvt, rsp_signal, sampling_rate)
    return rvt


def _rsp_rvt_birn(
    rsp_signal,
    sampling_rate=1000,
    silent=False,
    window_length=0.4,
    peak_distance=0.8,
    peak_prominence=0.5,
    interpolation_method="linear",
):
    zsmooth_signal = _smooth_rsp_data(
        rsp_signal,
        sampling_rate=sampling_rate,
        window_length=window_length,
        silent=silent,
    )
    info = rsp_findpeaks(
        zsmooth_signal,
        method="scipy",
        peak_distance=peak_distance,
        peak_prominence=peak_prominence,
    )
    peak_coords = info["RSP_Peaks"]
    trough_coords = info["RSP_Troughs"]
    # prepare for loop
    seconds_delta = [np.nan]
    mid_peak = [np.nan]
    # loop over peaks
    for ending_peak_index in range(1, len(peak_coords)):
        starting_peak = peak_coords[ending_peak_index - 1]
        ending_peak = peak_coords[ending_peak_index]
        mid_peak.append(round((starting_peak + ending_peak) / 2))
        seconds_delta.append((ending_peak - starting_peak) / sampling_rate)

    # Interpolate
    output_range = range(len(zsmooth_signal))
    rvt_time = signal_interpolate(mid_peak, seconds_delta, output_range, method=interpolation_method)
    rvt_peaks = signal_interpolate(
        peak_coords,
        zsmooth_signal[peak_coords],
        output_range,
        method=interpolation_method,
    )
    rvt_troughs = signal_interpolate(
        trough_coords,
        zsmooth_signal[trough_coords],
        output_range,
        method=interpolation_method,
    )

    # what is trigvec?
    # trigvec = (TR * signal_rate):len(zsmoothresp)
    rvt = (rvt_peaks - rvt_troughs) / rvt_time
    rvt[np.isinf(rvt)] = np.nan
    return rvt


def _rsp_rvt_power(
    rsp_signal,
    sampling_rate=1000,
    silent=False,
    window_length=0.4,
    peak_distance=0.8,
    peak_prominence=0.5,
    interpolation_method="linear",
):
    # preprocess signal
    zsmooth_signal = _smooth_rsp_data(
        rsp_signal,
        sampling_rate=sampling_rate,
        silent=silent,
        window_length=window_length,
    )
    # find peaks and troughs
    info = rsp_findpeaks(
        zsmooth_signal,
        method="scipy",
        peak_distance=peak_distance,
        peak_prominence=peak_prominence,
    )
    peak_coords = info["RSP_Peaks"]
    trough_coords = info["RSP_Troughs"]
    # initialize for loop
    peak_heights = [np.nan] * len(peak_coords)

    # go over loop
    for peak_index in range(1, len(peak_coords)):
        # find peak and trough
        peak_loc = peak_coords[peak_index]
        prev_peak_loc = peak_coords[peak_index - 1]
        # find troughs between prev_peak_loc and peak_loc
        trough_locs = trough_coords[(trough_coords > prev_peak_loc) & (trough_coords < peak_loc)]

        # safety catch if there is no trough found
        if len(trough_locs) == 0:
            continue

        trough_loc = max(trough_locs)
        # calculate peak_height for peak at peak_index
        peak_heights[peak_index] = (zsmooth_signal[peak_loc] - zsmooth_signal[trough_loc]) / (peak_loc - prev_peak_loc)

    return signal_interpolate(peak_coords, peak_heights, range(len(rsp_signal)), method=interpolation_method)


def _smooth_rsp_data(signal, sampling_rate=1000, window_length=0.4, silent=False):
    signal = rsp_clean(
        signal,
        sampling_rate=sampling_rate,
        window_length=window_length,
        method="hampel",
    )
    smooth_signal = scipy.signal.savgol_filter(
        signal,
        window_length=_make_uneven_filter_size(window_length * sampling_rate, silent),
        polyorder=2,
    )
    zsmooth_signal = scipy.stats.zscore(smooth_signal)
    return zsmooth_signal


def _rsp_rvt_harrison(
    rsp_signal,
    sampling_rate=1000,
    boundaries=[2.0, 1 / 30],
    iterations=10,
    silent=False,
):
    # low-pass filter at not too far above breathing-rate to remove high-frequency noise
    n_pad = int(np.ceil(10 * sampling_rate))

    d = scipy.signal.iirfilter(N=10, Wn=0.75, btype="lowpass", analog=False, output="sos", fs=sampling_rate)
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
    d = scipy.signal.iirfilter(N=10, Wn=0.2, btype="lowpass", analog=False, output="sos", fs=sampling_rate)
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
    plt.figure()
    plt.title("Respiratory Volume per Time (RVT)")
    plt.xlabel("Time [s]")
    plt.plot(
        rescale(rsp_signal, to=[np.nanmin(rvt), np.nanmax(rvt)]),
        label="RSP",
        color="#CFD8DC",
    )
    plt.plot(rvt, label="RVT", color="#00BCD4")
    plt.legend()
    tickpositions = plt.gca().get_xticks()[1:-1]
    plt.xticks(tickpositions, [tickposition / sampling_rate for tickposition in tickpositions])


def _make_uneven_filter_size(number, silent=False):
    if number < 0:
        if not silent:
            warn(
                "Received a negative filter size, progressed with filter size 1.",
                category=NeuroKitWarning,
            )
        return 1
    if number % 2 == 1:
        return int(number)
    if number > 0:
        return int(number - 1)
    return 1
