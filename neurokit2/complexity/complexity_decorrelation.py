import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal import signal_autocor


def complexity_decorrelation(signal, show=False):
    """**Decorrelation Time (DT)**

    The decorrelation time (DT) is defined as the time (in samples) of the first zero crossing of
    the autocorrelation sequence. A shorter decorrelation time corresponds to a less correlated
    signal. For instance, a drop in the decorrelation time of EEG has been observed prior to
    seizures, related to a decrease in the low frequency power (Mormann et al., 2005).


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    show : bool
        If True, will return a plot of the autocorrelation.

    Returns
    -------
    float
        Decorrelation Time (DT)
    dict
        A dictionary containing additional information (currently empty, but returned nonetheless
        for consistency with other functions).

    See Also
    --------
    .signal_autocor

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a signal
      signal = nk.signal_simulate(duration=5, sampling_rate=100, frequency=[5, 6], noise=0.5)

      # Compute DT
      @savefig p_complexity_decorrelation1.png scale=100%
      dt, _ = nk.complexity_decorrelation(signal, show=True)
      @suppress
      plt.close()

      dt

    References
    ----------
    * Mormann, F., Kreuz, T., Rieke, C., Andrzejak, R. G., Kraskov, A., David, P., ... & Lehnertz,
      K. (2005). On the predictability of epileptic seizures. Clinical neurophysiology, 116(3),
      569-587.
    * Teixeira, C. A., Direito, B., Feldwisch-Drentrup, H., Valderrama, M., Costa, R. P.,
      Alvarado-Rojas, C., ... & Dourado, A. (2011). EPILAB: A software package for studies on the
      prediction of epileptic seizures. Journal of Neuroscience Methods, 200(2), 257-271.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Unbiased autocor (see https://github.com/mne-tools/mne-features/)
    autocor, _ = signal_autocor(signal, unbiased=True)

    # Get zero-crossings
    zc = np.diff(np.sign(autocor)) != 0
    if np.any(zc):
        dt = np.argmax(zc) + 1
    else:
        dt = -1

    if show is True:
        # Max length of autocorrelation to plot
        max_len = int(dt * 4)
        if max_len > len(autocor):
            max_len = len(autocor)

        plt.plot(autocor[0:max_len])
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.xticks(np.arange(0, max_len, step=dt).astype(int))
        plt.axvline(dt, color="red", linestyle="--", label=f"DT = {dt}")
        plt.axhline(0, color="black", linestyle="--")
        plt.title("Decorrelation Time (DT)")
        plt.legend()

    return dt, {}
