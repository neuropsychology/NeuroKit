# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from .signal_period import signal_period


def signal_rate(
    peaks,
    sampling_rate=1000,
    desired_length=None,
    interpolation_method="monotone_cubic",
    show=False,
):
    """**Compute Signal Rate**

    Calculate signal rate (per minute) from a series of peaks. It is a general function that works
    for any series of peaks (i.e., not specific to a particular type of signal). It is computed as
    ``60 / period``, where the period is the time between the peaks (see func:`.signal_period`).

    .. note:: This function is implemented under :func:`.signal_rate`, but it also re-exported under
       different names, such as :func:`.ecg_rate`, or :func:`.rsp_rate`. The
       aliases are provided for consistency.

    Parameters
    ----------
    peaks : Union[list, np.array, pd.DataFrame, pd.Series, dict]
        The samples at which the peaks occur. If an array is passed in, it is assumed that it was
        obtained with :func:`.signal_findpeaks`. If a DataFrame is passed in, it is assumed it is
        of the same length as the input signal in which occurrences of R-peaks are marked as "1",
        with such containers obtained with e.g., :func:.`ecg_findpeaks` or :func:`.rsp_findpeaks`.
    sampling_rate : int
        The sampling frequency of the signal that contains peaks (in Hz, i.e., samples/second).
        Defaults to 1000.
    desired_length : int
        If left at the default None, the returned rated will have the same number of elements as
        ``peaks``. If set to a value larger than the sample at which the last peak occurs in the
        signal (i.e., ``peaks[-1]``), the returned rate will be interpolated between peaks over
        ``desired_length`` samples. To interpolate the rate over the entire duration of the signal,
        set ``desired_length`` to the number of samples in the signal. Cannot be smaller than or
        equal to the sample at which the last peak occurs in the signal. Defaults to ``None``.
    interpolation_method : str
        Method used to interpolate the rate between peaks. See :func:`.signal_interpolate`.
        ``"monotone_cubic"`` is chosen as the default interpolation method since it ensures monotone
        interpolation between data points (i.e., it prevents physiologically implausible
        "overshoots" or "undershoots" in the y-direction). In contrast, the widely used cubic
        spline interpolation does not ensure monotonicity.
     show : bool
        If ``True``, shows a plot. Defaults to ``False``.

    Returns
    -------
    array
        A vector containing the rate (peaks per minute).

    See Also
    --------
    signal_period, signal_findpeaks, signal_fixpeaks, signal_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Create signal of varying frequency
      freq = nk.signal_simulate(1, frequency = 1)
      signal = np.sin((freq).cumsum() * 0.5)

      # Find peaks
      info = nk.signal_findpeaks(signal)

      # Compute rate using 2 methods
      rate1 = nk.signal_rate(peaks=info["Peaks"],
                             desired_length=len(signal),
                             interpolation_method="nearest")

      rate2 = nk.signal_rate(peaks=info["Peaks"],
                             desired_length=len(signal),
                             interpolation_method="monotone_cubic")

      # Visualize signal and rate on the same scale
      @savefig p_signal_rate1.png scale=100%
      nk.signal_plot([signal, rate1, rate2],
                     labels = ["Original signal", "Rate (nearest)", "Rate (monotone cubic)"],
                     standardize = True)
      @suppress
      plt.close()

    """
    period = signal_period(peaks, sampling_rate, desired_length, interpolation_method)
    rate = 60 / period

    if show is True:
        _signal_rate_plot(rate, peaks, sampling_rate, interpolation_method)

    return rate


# =============================================================================
# Internals
# =============================================================================
def _signal_rate_plot(
    rate,
    peaks,
    sampling_rate=None,
    interpolation_method=None,
    title="Rate",
    ytitle="Cycle per minute",
    color="black",
    color_mean="orange",
    color_points="red",
    ax=None,
):
    # Prepare plot
    if ax is None:
        fig, ax = plt.subplots()

    if sampling_rate is None:
        x_axis = np.arange(0, len(rate))
        ax.set_xlabel("Time (samples)")
    else:
        x_axis = np.linspace(0, len(rate) / sampling_rate, len(rate))
        ax.set_xlabel("Time (seconds)")

    if interpolation_method is not None:
        title += " (interpolation method: " + str(interpolation_method) + ")"
    ax.set_title(title)
    ax.set_ylabel(ytitle)

    # Plot continuous rate
    ax.plot(
        x_axis,
        rate,
        color=color,
        label="Rate",
        linewidth=1.5,
    )

    # Plot points
    if peaks is not None:
        ax.scatter(
            x_axis[peaks],
            rate[peaks],
            color=color_points,
        )

    # Show average rate
    rate_mean = rate.mean()
    ax.axhline(y=rate_mean, label="Mean", linestyle="--", color=color_mean)

    ax.legend(loc="upper right")

    return ax
