from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..signal.signal_psd import signal_psd


def entropy_wiener(signal, sampling_rate=1000, show=False, **kwargs):
    """**Wiener Entropy (WE, also known as Spectral Flatness)**

    The Wiener entropy (also known as Spectral Flatness, or tonality coefficient in sound
    processing) is a measure to quantify how noise-like a signal is, and is typically applied to
    characterize an audio spectrum (Johnston, 1988) or bird songs (Tchernichovski et al., 2000). It
    is the ratio of the geometric mean of the power spectrum to its arithmetic mean:

    .. math::

      SF = \\frac{\\exp\\left(\\frac{1}{N_f} \\sum_{f} \\ln P(f)\\right)}{\\frac{1}{N_f} \\sum_{f} P(f)}

    where :math:`P(f)` is the power at the frequency *f* and :math:`N_f` is the number of frequency
    bins. A high spectral flatness (closer to 1) indicates that the spectrum has a similar amount
    of power in all spectral bands, and is similar to white noise. A low spectral flatness
    (approaching 0 for a pure tone) indicates that the spectral power is concentrated in a
    relatively small number of spectral bands. The Wiener entropy is often reported on a
    logarithmic scale, :math:`\\ln(SF)`, from 0 (white noise) to minus infinity (pure tone), which
    is returned in ``info``.

    The geometric mean of the spectrum is the variance of the error of the best linear prediction of
    a sample from all the previous ones (the Kolmogorov-Szegő formula), so that SF is the proportion
    of the variance of the signal that cannot be linearly predicted from its past (Dubnov, 2004).
    Unlike the spectral entropy (:func:`entropy_spectral`), it is dominated by the frequencies with
    the least power: a single frequency bin with no power makes it 0.

    The estimator of the spectrum matters, as its fluctuations lower SF: for white noise, the
    expected flatness of a spectrum averaged over *K* independent windows is about
    :math:`\\exp(\\psi(K) - \\ln K)` (where :math:`\\psi` is the digamma function), whatever the
    length of the signal. By default, the PSD is therefore estimated by :func:`.signal_psd` with the
    Welch method on windows of a sixteenth of the signal (at least 32 samples), i.e., averaging about
    30 half-overlapping windows, which gives about 0.98 for white noise (instead of 0.84 with the
    default windows of :func:`.signal_psd`, half the signal, and 0.56 with a single periodogram,
    ``method="fft"``). Longer windows (``window``, in seconds) give a finer frequency resolution, at
    the cost of this bias. The lowest frequencies, including the DC component, are left out (those
    below :math:`4 \\times sampling\\_rate / N`, where *N* is the length of the signal).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). It only matters when the
        frequency range is restricted (with ``min_frequency`` or ``max_frequency``).
    show : bool
        Display the power spectrum, with its geometric and arithmetic means.
    **kwargs : optional
        Other arguments to be passed to :func:`.signal_psd` (such as ``method``, ``min_frequency``,
        ``max_frequency`` or ``window``). With the Welch method, ``window`` defaults to a sixteenth
        of the signal (at least 32 samples), see above.

    Returns
    -------
    SF : float
        The Spectral Flatness, between 0 and 1. It is NaN (with a warning) for constant signals,
        signals with missing values, and when the spectrum has fewer than 2 frequency bins.
    info : dict
        A dictionary containing additional information, such as the Wiener entropy on the
        logarithmic scale (``"Log"``, i.e., :math:`\\ln(SF)`) and the power spectrum (``"PSD"``).

    See Also
    --------
    entropy_spectral, .signal_psd

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Two sinusoids with a little noise: the power is concentrated in a few frequencies
      signal = nk.signal_simulate(duration=10, sampling_rate=100, frequency=[3, 10], noise=0.05)

      @savefig p_entropy_wiener1.png scale=100%
      sf, info = nk.entropy_wiener(signal, sampling_rate=100, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      sf
      info["Log"]

    White noise has a flat spectrum (SF close to 1), and pink noise is in between.

    .. ipython:: python

      white = nk.signal_noise(duration=10, sampling_rate=100, beta=0)
      pink = nk.signal_noise(duration=10, sampling_rate=100, beta=1)

      nk.entropy_wiener(white, sampling_rate=100)[0]
      nk.entropy_wiener(pink, sampling_rate=100)[0]

    References
    ----------
    * Johnston, J. D. (1988). Transform coding of audio signals using perceptual noise criteria.
      IEEE Journal on Selected Areas in Communications, 6(2), 314-323.
    * Tchernichovski, O., Nottebohm, F., Ho, C. E., Pesaran, B., & Mitra, P. P. (2000). A procedure
      for an automated measurement of song similarity. Animal Behaviour, 59(6), 1167-1176.
    * Dubnov, S. (2004). Generalization of spectral flatness measure for non-gaussian linear
      processes. IEEE Signal Processing Letters, 11(8), 698-701.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    signal = np.asarray(signal, dtype=float)
    empty = pd.DataFrame({"Frequency": [], "Power": []})
    if np.any(~np.isfinite(signal)):
        warn("The signal contains missing or infinite values. Returning NaN.", category=NeuroKitWarning)
        return np.nan, {"Log": np.nan, "PSD": empty}
    if len(signal) == 0 or np.ptp(signal) == 0:
        warn("The signal is constant (it has no power at any frequency). Returning NaN.", category=NeuroKitWarning)
        return np.nan, {"Log": np.nan, "PSD": empty}

    # Power spectrum density (PSD). The geometric mean is very sensitive to the fluctuations of the
    # estimated spectrum, so the Welch method averages many windows by default (about 30)
    if str(kwargs.get("method", "welch")).lower() == "welch" and kwargs.get("window") is None:
        n = len(signal)
        kwargs["window"] = max(n // 16, min(n // 2, 32)) / sampling_rate
    psd = signal_psd(signal, sampling_rate=sampling_rate, **kwargs)
    if len(psd) < 2:
        warn(
            f"The power spectrum has {len(psd)} frequency bin(s), the signal is too short or the frequency"
            " range too narrow. Returning NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, {"Log": np.nan, "PSD": psd}

    power = psd["Power"].values
    if np.any(power < 0):
        raise ValueError("NeuroKit error: entropy_wiener(): the power spectrum has negative values.")

    # Ratio of the geometric to the arithmetic mean, on the log scale (frequency bins with no power
    # give -inf, i.e., a flatness of 0)
    with np.errstate(divide="ignore"):
        log_sf = np.mean(np.log(power)) - np.log(np.mean(power))
    log_sf = min(log_sf, 0.0)  # Rounding errors for perfectly flat spectra
    sf = np.exp(log_sf)

    if show is True:
        plt.semilogy(psd["Frequency"], power, color="#2196F3")
        plt.axhline(np.mean(power), color="#E91E63", linestyle="--", label="Arithmetic mean")
        plt.axhline(np.mean(power) * sf, color="#FF9800", linestyle="--", label="Geometric mean")
        plt.title(f"Spectral Flatness (SF = {sf:.3f})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.legend()

    return sf, {"Log": log_sf, "PSD": psd}
