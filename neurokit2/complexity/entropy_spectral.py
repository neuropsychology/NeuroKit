from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..signal.signal_psd import signal_psd
from .entropy_shannon import entropy_shannon


def entropy_spectral(signal, bins=None, show=False, **kwargs):
    """**Spectral Entropy (SpEn)**

    Spectral entropy (SE or SpEn) treats the signal's normalized power spectrum density (PSD) in the
    frequency domain as a probability distribution, and calculates the Shannon entropy of it. It is
    divided by its maximum, the logarithm of the number of frequency bins (reached for a flat
    spectrum), so that it lies between 0 and 1 (Inouye et al., 1991).

    .. math:: SpEn = -\\frac{1}{\\log_2(N_f)} \\sum_{f} P(f) \\log_2[P(f)]

    where :math:`P(f)` is the power at the frequency *f* divided by the total power, and
    :math:`N_f` is the number of frequency bins (``bins`` when the spectrum is binned).

    A signal with a single frequency component (i.e., pure sinusoid) produces the smallest entropy.
    On the other hand, a signal with all frequency components of equal power value (white
    noise) produces the greatest entropy. Note that these bounds are not reached in practice, as
    the estimated spectrum of a sinusoid is spread over a few bins (e.g., by the window of the
    Welch method), and that of white noise fluctuates around its flat expectation.

    By default, the PSD is estimated by :func:`.signal_psd` with the Welch method, which leaves
    out the lowest frequencies, including the DC component (those below
    :math:`4 \\times sampling\\_rate / N`, where *N* is the length of the signal).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    bins : int
        If an integer (at least 2) is passed, the frequency range of the PSD is cut into this number
        of bins of equal width, whose powers are summed, and SpEn is normalized by
        :math:`\\log_2(bins)`. If ``None`` (default), each frequency of the PSD is a bin.
    show : bool
        Display the power spectrum.
    **kwargs : optional
        Keyword arguments to be passed to ``signal_psd()``, such as ``sampling_rate`` (default
        1000), ``method``, ``min_frequency`` or ``max_frequency``. The sampling rate only matters
        when the frequency range is restricted.

    Returns
    -------
    SpEn : float
        Spectral Entropy, between 0 and 1. It is NaN (with a warning) for constant signals, signals
        with missing values, and when the spectrum has fewer than 2 frequency bins (e.g., for very
        short signals or a very narrow frequency range).
    info : dict
        A dictionary containing additional information regarding the parameters used, such as the
        normalized power spectrum ``PSD``.

    See Also
    --------
    entropy_shannon, entropy_wiener, .signal_psd

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal with Laplace Noise
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6, 10], noise=0.1)

      # Compute Spectral Entropy
      @savefig p_entropy_spectral1.png scale=100%
      SpEn, info = nk.entropy_spectral(signal, show=True)
      @suppress
      plt.close()


    .. ipython:: python

      SpEn


    Bin the frequency spectrum.

    .. ipython:: python

        @savefig p_entropy_spectral2.png scale=100%
        SpEn, info = nk.entropy_spectral(signal, bins=10, show=True)
        @suppress
        plt.close()

    References
    ----------
    * Inouye, T., Shinosaki, K., Sakamoto, H., Toi, S., Ukai, S., Iyama, A., Katsuda, Y., &
      Hirano, M. (1991). Quantification of EEG irregularity by use of the entropy of the power
      spectrum. Electroencephalography and Clinical Neurophysiology, 79(3), 204-210.
    * Crepeau, J. C., & Isaacson, L. K. (1991). Spectral Entropy Measurements of Coherent
      Structures in an Evolving Shear Layer. Journal of Non-Equilibrium Thermodynamics, 16(2).
      doi:10.1515/jnet.1991.16.2.137

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if bins is not None and (not isinstance(bins, (int, np.integer)) or bins < 2):
        raise ValueError(
            f"NeuroKit error: entropy_spectral(): `bins` should be None or an integer of at least 2, not {bins!r}."
        )

    signal = np.asarray(signal, dtype=float)
    empty = pd.DataFrame({"Frequency": [], "Power": []})
    if np.any(~np.isfinite(signal)):
        warn(
            "The signal contains missing or infinite values. SpEn cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, {"PSD": empty}
    if np.ptp(signal) == 0:
        warn(
            "The signal is constant (it has no power at any frequency). SpEn cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, {"PSD": empty}

    # Power-spectrum density (PSD) (the actual sampling rate does not matter, unless the
    # frequency range is restricted with min_frequency or max_frequency)
    kwargs.setdefault("sampling_rate", 1000)
    psd = signal_psd(signal, **kwargs)

    if len(psd) < 2:
        warn(
            f"The power spectrum has {len(psd)} frequency bin(s), the signal is too short or the frequency"
            " range too narrow. SpEn cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, {"PSD": psd}

    # Cut into bins
    if bins is not None:
        psd = psd.groupby(pd.cut(psd["Frequency"], bins=bins), observed=False).agg("sum")
        idx = np.array(psd.index, dtype=str)
    else:
        idx = psd["Frequency"].values

    # Area under normalized spectrum should sum to 1 (np.sum(psd["Power"]))
    psd = psd.copy()
    psd["Power"] = psd["Power"] / psd["Power"].sum()

    if show is True:
        plt.bar(idx, psd["Power"])
        if not pd.api.types.is_float_dtype(idx):
            plt.xticks(rotation=90)
        plt.title("Normalized Power Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized Power")

    # Compute Shannon entropy
    se, _ = entropy_shannon(freq=psd["Power"].values, base=2)

    # Normalize by the maximum entropy (a flat spectrum)
    se /= np.log2(len(psd))  # between 0 and 1

    return se, {"PSD": psd}
