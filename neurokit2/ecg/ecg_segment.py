# - * - coding: utf-8 - * -
import matplotlib.pyplot as plt
import numpy as np

from ..epochs import epochs_create, epochs_to_df
from ..signal import signal_rate
from .ecg_peaks import ecg_peaks


def ecg_segment(ecg_cleaned, rpeaks=None, sampling_rate=1000, show=False):
    """Segment an ECG signal into single heartbeats.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by `ecg_clean()`.
    rpeaks : dict
        The samples at which the R-peaks occur. Dict returned by
        `ecg_peaks()`. Defaults to None.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    show : bool
        If True, will return a plot of heartbeats. Defaults to False.

    Returns
    -------
    dict
        A dict containing DataFrames for all segmented heartbeats.

    See Also
    --------
    ecg_clean, ecg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
    >>> ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
    >>> nk.ecg_segment(ecg_cleaned, rpeaks=None, sampling_rate=1000, show=True) #doctest: +ELLIPSIS
    {'1':              Signal  Index Label
     ...
     '2':              Signal  Index Label
     ...
     '19':              Signal  Index Label
     ...}

    """
    # Sanitize inputs
    if rpeaks is None:
        _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, correct_artifacts=True)
        rpeaks = rpeaks["ECG_R_Peaks"]

    epochs_start, epochs_end = _ecg_segment_window(
        rpeaks=rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned)
    )
    heartbeats = epochs_create(
        ecg_cleaned, rpeaks, sampling_rate=sampling_rate, epochs_start=epochs_start, epochs_end=epochs_end
    )

    if show:
        heartbeats_plot = epochs_to_df(heartbeats)
        heartbeats_pivoted = heartbeats_plot.pivot(index="Time", columns="Label", values="Signal")
        plt.plot(heartbeats_pivoted)
        plt.xlabel("Time (s)")
        plt.title("Individual Heart Beats")
        cmap = iter(
            plt.cm.YlOrRd(np.linspace(0, 1, num=int(heartbeats_plot["Label"].nunique())))
        )  # pylint: disable=no-member
        lines = []
        for x, color in zip(heartbeats_pivoted, cmap):
            (line,) = plt.plot(heartbeats_pivoted[x], color=color)
            lines.append(line)

    return heartbeats


def _ecg_segment_window(heart_rate=None, rpeaks=None, sampling_rate=1000, desired_length=None):

    # Extract heart rate
    if heart_rate is not None:
        heart_rate = np.mean(heart_rate)
    if rpeaks is not None:
        heart_rate = np.mean(signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=desired_length))

    # Modulator
    m = heart_rate / 60

    # Window
    epochs_start = -0.35 / m
    epochs_end = 0.5 / m

    # Adjust for high heart rates
    if heart_rate >= 80:
        c = 0.1
        epochs_start = epochs_start - c
        epochs_end = epochs_end + c

    return epochs_start, epochs_end
