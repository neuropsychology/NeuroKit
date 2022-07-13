# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal


def signal_synchrony(signal1, signal2, method="hilbert", window_size=50):
    """**Synchrony (coupling) between two signals**

    Signal coherence refers to the strength of the mutual relationship (i.e., the amount of shared
    information) between two signals. Synchrony is coherence "in phase" (two waveforms are "in
    phase" when the peaks and troughs occur at the same time). Synchrony will always be coherent,
    but coherence need not always be synchronous.

    This function computes a continuous index of coupling between two signals either using the
    ``"hilbert"`` method to get the instantaneous phase synchrony, or using a rolling window
    correlation.

    The instantaneous phase synchrony measures the phase similarities between signals at each
    timepoint. The phase refers to the angle of the signal, calculated through the hilbert
    transform, when it is resonating between -pi to pi degrees. When two signals line up in phase
    their angular difference becomes zero.

    For less clean signals, windowed correlations are widely used because of their simplicity, and
    can be a good a robust approximation of synchrony between two signals. The limitation is the
    need to select a window size.

    Parameters
    ----------
    signal1 : Union[list, np.array, pd.Series]
        Time series in the form of a vector of values.
    signal2 : Union[list, np.array, pd.Series]
        Time series in the form of a vector of values.
    method : str
        The method to use. Can be one of ``"hilbert"`` or ``"correlation"``.
    window_size : int
        Only used if ``method='correlation'``. The number of samples to use for rolling correlation.

    See Also
    --------
    scipy.signal.hilbert, mutual_information

    Returns
    -------
    array
        A vector containing the phase of the signal, between 0 and 2*pi.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      s1 = nk.signal_simulate(duration=10, frequency=1)
      s2 = nk.signal_simulate(duration=10, frequency=1.5)

      coupling1 = nk.signal_synchrony(s1, s2, method="hilbert")
      coupling2 = nk.signal_synchrony(s1, s2, method="correlation", window_size=1000/2)

      @savefig p_signal_synchrony1.png scale=100%
      nk.signal_plot([s1, s2, coupling1, coupling2], labels=["s1", "s2", "hilbert", "correlation"])
      @suppress
      plt.close()

    References
    ----------
    *  http://jinhyuncheong.com/jekyll/update/2017/12/10/Timeseries_synchrony_tutorial_and_simulations.html

    """
    if method.lower() in ["hilbert", "phase"]:
        coupling = _signal_synchrony_hilbert(signal1, signal2)
    elif method.lower() in ["correlation"]:
        coupling = _signal_synchrony_correlation(signal1, signal2, window_size=int(window_size))

    else:
        raise ValueError(
            "NeuroKit error: signal_synchrony(): 'method' should be one of 'hilbert' or 'correlation'."
        )

    return coupling


# =============================================================================
# Methods
# =============================================================================


def _signal_synchrony_hilbert(signal1, signal2):

    hill1 = scipy.signal.hilbert(signal1)
    hill2 = scipy.signal.hilbert(signal2)

    phase1 = np.angle(hill1, deg=False)
    phase2 = np.angle(hill2, deg=False)
    synchrony = 1 - np.sin(np.abs(phase1 - phase2) / 2)

    return synchrony


def _signal_synchrony_correlation(signal1, signal2, window_size, center=False):
    """**Calculates pairwise rolling correlation at each time**
    Grabs the upper triangle, at each timepoint.

    * window: window size of rolling corr in samples
    * center: whether to center result (Default: False, so correlation values are listed on the
      right.)

    """
    data = pd.DataFrame({"y1": signal1, "y2": signal2})

    rolled = data.rolling(window=window_size, center=center).corr()
    synchrony = rolled["y1"].loc[rolled.index.get_level_values(1) == "y2"].values

    # Realign
    synchrony = np.append(synchrony[int(window_size / 2) :], np.full(int(window_size / 2), np.nan))
    synchrony[np.isnan(synchrony)] = np.nanmean(synchrony)

    return synchrony
