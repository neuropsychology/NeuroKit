from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def fractal_nld(signal, corrected=False):
    """**Fractal dimension via Normalized Length Density (NLDFD)**

    NLDFD is a very simple index corresponding to the length of the amplitude-normalized signal
    divided by its number of samples *N* (Kalauzi et al., 2009):

    .. math::

      NLD = \\frac{1}{N} \\sum_{i=2}^{N} |y_n(i) - y_n(i - 1)|

    where :math:`y_n` is the signal centered on its mean and divided by its mean absolute deviation
    (:math:`y_n = y' / mean(|y'|)`, with :math:`y' = y - mean(y)`).
    This method was developed for measuring signal complexity of very short durations (< 30
    samples), and can be used for instance when continuous signal FD changes (or "running" FD) are
    of interest (by computing it on sliding windows, see example).

    For methods such as Higuchi's FD, the standard deviation of the window FD increases sharply
    when the epoch becomes shorter. The NLD method results in lower standard deviation especially
    for shorter epochs, though at the expense of lower accuracy in average window FD.

    See Also
    --------
    fractal_higuchi

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    corrected : bool
        If ``True``, will convert the NLD to an FD estimate using the calibration curve (power
        model) of Kalauzi et al. (2009), fitted on Weierstrass functions of known FD (1.01-1.99):
        ``FD = 1.9079*((NLD-0.097178)^0.18383)``. Note that this can result in ``np.nan`` if the
        result of the difference is negative. Kalauzi et al. (2009) also proposed parameters
        adjusted on 10-sample epochs of EEG signals (:math:`a = 1.8399` and :math:`k = 0.3523`,
        with the same :math:`NLD_0`), so that the window FDs stay within [1, 2].

    Returns
    --------
    fd : float
        The normalized length density (or the FD estimate if ``corrected=True``).
    info : dict
        A dictionary containing additional information (currently empty, but returned nonetheless
        for consistency with other functions).

    Examples
    ----------
    **Example 1**: Usage on a short signal

    .. ipython:: python

      import neurokit2 as nk

      # Simulate a short signal with duration of 0.5s
      signal = nk.signal_simulate(duration=0.5, frequency=[3, 5])

      # Compute Fractal Dimension
      fd, _ = nk.fractal_nld(signal, corrected=False)
      fd

    **Example 2**: Compute FD-NLD on non-overlapping windows

    .. ipython:: python

      import numpy as np

      # Simulate a long signal with duration of 5s
      signal = nk.signal_simulate(duration=5, frequency=[3, 5, 10], noise=0.1)

      # We want windows of size=100 (0.1s)
      n_windows = len(signal) // 100  # How many windows

      # Split signal into windows
      windows = np.array_split(signal, n_windows)

      # Compute FD-NLD on all windows
      nld = [nk.fractal_nld(i, corrected=False)[0] for i in windows]
      np.mean(nld)  # Get average


    **Example 3**: Calculate FD-NLD on sliding windows

    .. ipython:: python

      # Simulate a long signal with duration of 5s
      signal = nk.signal_simulate(duration=5, frequency=[3, 5, 10], noise=0.1)
      # Add period of noise
      signal[1000:3000] = signal[1000:3000] + np.random.normal(0, 1, size=2000)

      # Create function-wrapper that only return the NLD value
      nld = lambda x: nk.fractal_nld(x, corrected=False)[0]

      # Use them in a rolling window of 100 samples (0.1s)
      rolling_nld = pd.Series(signal).rolling(100, min_periods = 100, center=True).apply(nld)

      @savefig p_nld1.png scale=100%
      nk.signal_plot([signal, rolling_nld], subplots=True, labels=["Signal", "FD-NLD"])
      @suppress
      plt.close()


    References
    ----------
    * Kalauzi, A., Bojić, T., & Rakić, L. (2009). Extracting complexity waveforms from
      one-dimensional signals. Nonlinear biomedical physics, 3(1), 1-11.

    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    # Amplitude normalization (Kalauzi et al., 2009, eqs. 9-10): center on the mean and divide by
    # the mean absolute deviation (not the standard deviation)
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal)
    signal = signal / np.nanmean(np.abs(signal))

    # Calculate normalized length density (eq. 8): the N - 1 absolute differences are divided by N
    diffs = np.abs(np.diff(signal))
    nld = np.nansum(diffs) / np.sum(~np.isnan(signal))

    # TODO: Kalauzi et al. (2009) also describe (1) parameters of the power model adjusted on EEG
    # (a = 1.8399, k = 0.3523, for 10-sample epochs), and (2) an "integral normalization" (IN), in
    # which the whole signal is normalized once before computing NLD on its windows (instead of
    # normalizing each window, as done here when calling this function on windows), which was more
    # accurate for extracting FD waveforms (their Fig. 7). Consider offering them as options.
    if corrected:
        # Power model fitted on Weierstrass functions of known FD (from Kalauzi et al. 2009)
        a = 1.9079
        k = 0.18383
        nld_diff = nld - 0.097178  # NLD - NLD0

        if nld_diff < 0:
            warn(
                "Normalized Length Density of the signal may be too small, retuning `np.nan`.",
                category=NeuroKitWarning,
            )
            nld = np.nan
        else:
            nld = a * (nld_diff**k)

    # Compute fd
    return nld, {}
