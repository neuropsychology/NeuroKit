# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from ..signal import signal_smooth


def ppg_findpeaks(
    ppg_cleaned, sampling_rate=1000, method="elgendi", show=False, **kwargs
):
    """**Find systolic peaks in a photoplethysmogram (PPG) signal**

    Low-level function used by :func:`ppg_peaks` to identify peaks in a PPG signal using a
    different set of algorithms. Use the main function and see its documentation for details.

    Parameters
    ----------
    ppg_cleaned : Union[list, np.array, pd.Series]
        The cleaned PPG channel as returned by :func:`.ppg_clean`.
    sampling_rate : int
        The sampling frequency of the PPG (in Hz, i.e., samples/second). The default is 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"elgendi"``, ``"bishop"``. The default is
        ``"elgendi"``.
    show : bool
        If ``True``, returns a plot of the thresholds used during peak detection. Useful for
        debugging. The default is ``False``.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the samples at which systolic
        peaks occur, accessible with the key ``"PPG_Peaks"``.

    See Also
    --------
    ppg_simulate, ppg_clean

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      ppg = nk.ppg_simulate(heart_rate=75, duration=20, sampling_rate=50)
      ppg_clean = nk.ppg_clean(ppg, sampling_rate=50)

      @savefig p_ppg_findpeaks1.png scale=100%
      peaks = nk.ppg_findpeaks(ppg_clean, sampling_rate=100, show=True)
      @suppress
      plt.close()

      # Method by Bishop et al., (2018)
      @savefig p_ppg_findpeaks2.png scale=100%
      peaks = nk.ppg_findpeaks(ppg, method="bishop", show=True)
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
    * Charlton, P. H. et al. (2025). The MSPTDfast photoplethysmography beat detection algorithm:
      design, benchmarking, and open-source distribution. Physiological Measurement, 46, 035002.
    * Charlton, P. H. et al. (2024). MSPTDfast: An Efficient Photoplethysmography Beat Detection
      Algorithm. Proc CinC.

    """
    method = method.lower()
    if method in ["elgendi"]:
        peaks = _ppg_findpeaks_elgendi(ppg_cleaned, sampling_rate, show=show, **kwargs)
    elif method in ["msptd", "bishop2018", "bishop"]:
        peaks, _ = _ppg_findpeaks_bishop(ppg_cleaned, show=show, **kwargs)
    elif method in ["msptdfast", "msptdfastv2", "charlton2025", "charlton"]:
        peaks, onsets = _ppg_findpeaks_charlton(ppg_cleaned, sampling_rate, win_durn=6, show=show, **kwargs)
    elif method in ["msptdfastv1", "charlton2024"]:
        peaks, onsets = _ppg_findpeaks_charlton(ppg_cleaned, sampling_rate, win_durn=8, show=show, **kwargs)
    else:
        raise ValueError(
            "`method` not found. Must be one of the following: 'elgendi', 'bishop', 'charlton', 'charlton2024'."
        )

    # Prepare output.
    info = {"PPG_Peaks": peaks}
    if 'onsets' in locals():
        info["PPG_Onsets"] = onsets
        info["method_fixpeaks"] = 'Charlton2022'  # This was the default methodology used in MSPTDfastv1 and MSPTDfastv2

    return info


def _ppg_findpeaks_elgendi(
    signal,
    sampling_rate=1000,
    peakwindow=0.111,
    beatwindow=0.667,
    beatoffset=0.02,
    mindelay=0.3,
    show=False,
):
    """Implementation of Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions. PLoS ONE 8(10): e76585.
    doi:10.1371/journal.pone.0076585.

    All tune-able parameters are specified as keyword arguments. `signal` must be the bandpass-filtered raw PPG
    with a lowcut of .5 Hz, a highcut of 8 Hz.

    """
    if show:
        _, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax0.plot(signal, label="filtered")

    # Ignore the samples with negative amplitudes and square the samples with
    # values larger than zero.
    signal_abs = signal.copy()
    signal_abs[signal_abs < 0] = 0
    sqrd = signal_abs**2

    # Compute the thresholds for peak detection. Call with show=True in order
    # to visualize thresholds.
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak = signal_smooth(sqrd, kernel="boxcar", size=ma_peak_kernel)

    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat = signal_smooth(sqrd, kernel="boxcar", size=ma_beat_kernel)

    thr1 = ma_beat + beatoffset * np.mean(sqrd)  # threshold 1

    if show:
        ax1.plot(sqrd, label="squared")
        ax1.plot(thr1, label="threshold")
        ax1.legend(loc="upper right")

    # Identify start and end of PPG waves.
    waves = ma_peak > thr1
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    min_len = int(
        np.rint(peakwindow * sampling_rate)
    )  # this is threshold 2 in the paper
    min_delay = int(np.rint(mindelay * sampling_rate))
    peaks = [0]

    for i in range(num_waves):
        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        if len_wave < min_len:
            continue

        # Visualize wave span.
        if show:
            ax1.axvspan(beg, end, facecolor="m", alpha=0.5)

        # Find local maxima and their prominence within wave span.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks.
            if peak - peaks[-1] > min_delay:
                peaks.append(peak)

    peaks.pop(0)

    if show:
        ax0.scatter(peaks, signal_abs[peaks], c="r")
        ax0.legend(loc="upper right")
        ax0.set_title("PPG Peaks (Method by Elgendi et al., 2013)")

    peaks = np.asarray(peaks).astype(int)
    return peaks


def _ppg_findpeaks_bishop(
    signal,
    show=False,
):
    """Implementation of Bishop SM, Ercole A (2018) Multi-scale peak and trough detection optimised
    for periodic and quasi-periodic neuroscience data. doi:10.1007/978-3-319-65798-1_39.

    Currently designed for short signals of relatively low sampling frequencies (e.g. 6 seconds at
    100 Hz). Also, the function currently only returns peaks, but it does identify pulse onsets too.
    """

    # TODO: create ppg_peaks() that also returns onsets and stuff

    # Setup
    N = len(signal)
    L = int(np.ceil(N / 2) - 1)

    # Step 1: calculate local maxima and local minima scalograms

    # - detrend: this removes the best-fit straight line
    x = scipy.signal.detrend(signal, type="linear")

    # - initialise LMS matrices
    m_max = np.full((L, N), False)
    m_min = np.full((L, N), False)

    # - populate LMS matrices
    for k in range(1, L):  # scalogram scales
        for i in range(k + 2, N - k + 1):
            if x[i - 1] > x[i - k - 1] and x[i - 1] > x[i + k - 1]:
                m_max[k - 1, i - 1] = True
            if x[i - 1] < x[i - k - 1] and x[i - 1] < x[i + k - 1]:
                m_min[k - 1, i - 1] = True

    # Step 2: find the scale with the most local maxima (or local minima)
    # - row-wise summation (i.e. sum each row)
    gamma_max = np.sum(m_max, axis=1)
    # the "axis=1" option makes it row-wise
    gamma_min = np.sum(m_min, axis=1)
    # - find scale with the most local maxima (or local minima)
    lambda_max = np.argmax(gamma_max)
    lambda_min = np.argmax(gamma_min)

    # Step 3: Use lambda to remove all elements of m for which k>lambda
    m_max = m_max[: (lambda_max + 1), :]
    m_min = m_min[: (lambda_min + 1), :]

    # Step 4: Find peaks (and onsets)
    # - column-wise summation
    m_max_sum = np.sum(m_max == False, axis=0)
    m_min_sum = np.sum(m_min == False, axis=0)
    peaks = np.where(m_max_sum == 0)[0].astype(int)
    onsets = np.where(m_min_sum == 0)[0].astype(int)

    if show:
        _, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
        ax0.plot(signal, label="signal")
        ax0.scatter(peaks, signal[peaks], c="r")
        ax0.scatter(onsets, signal[onsets], c="b")
        ax0.set_title("PPG Peaks (Method by Bishop et al., 2018)")

    return peaks, onsets


def _ppg_findpeaks_charlton(
    signal,
    sampling_rate=1000,
    win_durn=6,
    show=False,
):
    """Implementation of MSPTDfastv2, as described in:
    Charlton, P. H. et al. (2025). The MSPTDfast photoplethysmography beat detection algorithm: design, benchmarking,
    and open-source distribution. Physiological Measurement, 46, 035002, doi:10.1088/1361-6579/adb89e

    Also, when win_durn=8, this is an implementation of MSPTDfastv1, as described in:
    Charlton, P. H. et al. (2024). MSPTDfast: An Efficient Photoplethysmography Beat Detection Algorithm. Proc CinC.
    """

    # Inner functions

    def find_m_max(x, N, max_scale, m_max):
        """Find local maxima scalogram for peaks
        """

        for k in range(1, max_scale + 1):  # scalogram scales
            for i in range(k + 2, N - k + 2):
                if x[i - 2] > x[i - k - 2] and x[i - 2] > x[i + k - 2]:
                    m_max[k - 1, i - 2] = True

        return m_max

    def find_m_min(x, N, max_scale, m_min):
        """Find local minima scalogram for onsets
        """

        for k in range(1, max_scale + 1):  # scalogram scales
            for i in range(k + 2, N - k + 2):
                if x[i - 2] < x[i - k - 2] and x[i - 2] < x[i + k - 2]:
                    m_min[k - 1, i - 2] = True

        return m_min

    def find_lms_using_msptd_approach(max_scale, x, options):
        """Find local maxima (or minima) scalogram(s) using the
        MSPTD approach
        """

        # Setup
        N = len(x)

        # Find local maxima scalogram (if required)
        if options["find_pks"]:
            m_max = np.full((max_scale, N), False)  # matrix for maxima
            m_max = find_m_max(x, N, max_scale, m_max)
        else:
            m_max = None

        # Find local minima scalogram (if required)
        if options["find_trs"]:
            m_min = np.full((max_scale, N), False)  # matrix for minima
            m_min = find_m_min(x, N, max_scale, m_min)
        else:
            m_min = None

        return m_max, m_min

    def downsample(win_sig, ds_factor):
        """Downsamples signal by picking out every nth sample, where n is
        specified by ds_factor
        """

        return win_sig[::ds_factor]

    def detect_peaks_and_onsets_using_msptd(signal, fs, options):
        """Detect peaks and onsets in a PPG signal using a modified MSPTD approach
        (where the modifications are those specified in Charlton et al. 2024)
        """

        # Setup
        N = len(signal)
        L = int(np.ceil(N / 2) - 1)

        # Step 0: Don't calculate scales outside the range of plausible HRs

        plaus_hr_hz = np.array(options['plaus_hr_bpm']) / 60  # in Hz
        init_scales = np.arange(1, L + 1)
        durn_signal = len(signal) / fs
        init_scales_fs = (L / init_scales) / durn_signal
        if options['use_reduced_lms_scales']:
            init_scales_inc_log = init_scales_fs >= plaus_hr_hz[0]
        else:
            init_scales_inc_log = np.ones_like(init_scales_fs, dtype=bool)  # DIDN"T FULLY UNDERSTAND

        max_scale_index = np.where(init_scales_inc_log)[0]  # DIDN"T FULLY UNDERSTAND THIS AND NEXT FEW LINES
        if max_scale_index.size > 0:
            max_scale = max_scale_index[-1] + 1  # Add 1 to convert from 0-based to 1-based index
        else:
            max_scale = None  # Or handle the case where no scales are valid

        # Step 1: calculate local maxima and local minima scalograms

        # - detrend
        x = scipy.signal.detrend(signal, type="linear")

        # - populate LMS matrices
        [m_max, m_min] = find_lms_using_msptd_approach(max_scale, x, options)

        # Step 2: find the scale with the most local maxima (or local minima)

        # - row-wise summation (i.e. sum each row)
        if options["find_pks"]:
            gamma_max = np.sum(m_max, axis=1)  # the "axis=1" option makes it row-wise
        if options["find_trs"]:
            gamma_min = np.sum(m_min, axis=1)
        # - find scale with the most local maxima (or local minima)
        if options["find_pks"]:
            lambda_max = np.argmax(gamma_max)
        if options["find_trs"]:
            lambda_min = np.argmax(gamma_min)

        # Step 3: Use lambda to remove all elements of m for which k>lambda
        first_scale_to_include = np.argmax(init_scales_inc_log)
        if options["find_pks"]:
            m_max = m_max[first_scale_to_include:lambda_max + 1, :]
        if options["find_trs"]:
            m_min = m_min[first_scale_to_include:lambda_min + 1, :]

        # Step 4: Find peaks (and onsets)
        # - column-wise summation
        if options["find_pks"]:
            m_max_sum = np.sum(m_max == False, axis=0)
            peaks = np.where(m_max_sum == 0)[0].astype(int)
        else:
            peaks = []

        if options["find_trs"]:
            m_min_sum = np.sum(m_min == False, axis=0)
            onsets = np.where(m_min_sum == 0)[0].astype(int)
        else:
            onsets = []

        return peaks, onsets

    # ~~~ Main function ~~~
    
    # Specify settings
    # - version: optimal selection (CinC 2024)
    options = {
        'find_trs': True,  # whether or not to find onsets
        'find_pks': True,  # whether or not to find peaks
        'do_ds': True,  # whether or not to do downsampling
        'ds_freq': 20,  # the target downsampling frequency
        'use_reduced_lms_scales': True,  # whether or not to reduce the number of scales (default 30 bpm)
        'win_len': win_durn,  # duration of individual windows for analysis (8 secs for MSPTDfastv1; 6 secs for MSPTDfastv2)
        'win_overlap': 0.2,  # proportion of window overlap
        'plaus_hr_bpm': [30, 200],  # range of plausible HRs (only the lower bound is used)
        'tol_durn': 0.05,  # tolerance window (+/- tol_durn) within which to search for true peak either side of candidate peak
    }

    options['tol_durn'] = 0.15  # added for neurokit implementation

    # Split into overlapping windows
    no_samps_in_win = options["win_len"] * sampling_rate
    if len(signal) <= no_samps_in_win:
        win_starts = 0
        win_ends = len(signal) - 1
    else:
        win_offset = round(no_samps_in_win * (1 - options["win_overlap"]))
        win_starts = list(range(0, len(signal) - no_samps_in_win + 1, win_offset))
        win_ends = [start + 1 + no_samps_in_win for start in win_starts]
        if win_ends[-1] < len(signal):
            win_starts.append(len(signal) - 1 - no_samps_in_win)
            win_ends.append(len(signal))
        # this ensures that the windows include the entire signal duration

    # Set up downsampling if the sampling frequency is particularly high
    if options["do_ds"]:
        min_fs = options["ds_freq"]
        if sampling_rate > min_fs:
            ds_factor = int(np.floor(sampling_rate / min_fs))
            ds_fs = sampling_rate / np.floor(sampling_rate / min_fs)
        else:
            options["do_ds"] = False

    # detect peaks and onsets in each window
    peaks = []
    onsets = []

    # cycle through each window
    for win_no in range(len(win_starts)):
        # Extract this window's data
        win_sig = signal[win_starts[win_no]:win_ends[win_no]]

        # Downsample signal
        if options['do_ds']:
            rel_sig = downsample(win_sig, ds_factor)
            rel_fs = ds_fs
        else:
            rel_sig = win_sig
            rel_fs = sampling_rate

        # Detect peaks and onsets
        p, t = detect_peaks_and_onsets_using_msptd(rel_sig, rel_fs, options)

        # Resample peaks
        if options['do_ds']:
            p = [peak * ds_factor for peak in p]
            t = [onset * ds_factor for onset in t]

        # Correct peak indices by finding highest point within tolerance either side of detected peaks
        tol_durn = options['tol_durn']  # note that this is only used if sampling frequency is 20 Hz or higher.
        if rel_fs < 10:
            tol_durn = 0.2
        elif rel_fs < 20:
            tol_durn = 0.1
        tol = int(np.ceil(rel_fs * tol_durn))

        for pk_no in range(len(p)):
            segment = win_sig[(p[pk_no] - tol):(p[pk_no] + tol + 1)]
            temp = np.argmax(segment)
            p[pk_no] = p[pk_no] - tol + temp

        # Correct onset indices by finding highest point within tolerance either side of detected onsets
        for onset_no in range(len(t)):
            segment = win_sig[(t[onset_no] - tol):(t[onset_no] + tol + 1)]
            temp = np.argmin(segment)
            t[onset_no] = t[onset_no] - tol + temp

        # Store peaks and onsets
        win_peaks = [peak + win_starts[win_no] for peak in p]
        peaks.extend(win_peaks)
        win_onsets = [onset + win_starts[win_no] for onset in t]
        onsets.extend(win_onsets)

    # Tidy up detected peaks and onsets (by ordering them and only retaining unique ones)
    peaks = sorted(set(peaks))
    onsets = sorted(set(onsets))

    # Plot results (optional)
    if show:
        _, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
        ax0.plot(signal, label="signal")
        ax0.scatter(peaks, signal[peaks], c="r")
        ax0.scatter(onsets, signal[onsets], c="b")
        ax0.set_title("PPG Onsets (Method by Charlton et al., 2024)")

    return peaks, onsets
