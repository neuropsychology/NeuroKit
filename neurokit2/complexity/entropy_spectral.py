import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal.signal_psd import signal_psd
from .entropy_shannon import entropy_shannon


def entropy_spectral(signal, bins=None, show=False, **kwargs):
    """**Spectral Entropy (SpEn)**

    Spectral entropy (SE or SpEn) treats the signal's normalized power spectrum density (PSD) in the
    frequency domain as a probability distribution, and calculates the Shannon entropy of it.

    .. math:: H(x, sf) =  -\\sum P(f) \\log_2[P(f)]

    A signal with a single frequency component (i.e., pure sinusoid) produces the smallest entropy.
    On the other hand, a signal with all frequency components of equal power value (white
    noise) produces the greatest entropy.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    bins : int
        If an integer is passed, will cut the PSD into a number of bins of frequency.
    show : bool
        Display the power spectrum.
    **kwargs : optional
        Keyword arguments to be passed to ``signal_psd()``, such as ``sampling_rate`` (default
        1000), ``min_frequency`` or ``max_frequency``. The sampling rate only matters when the
        frequency range is restricted.

    Returns
    -------
    SpEn : float
        Spectral Entropy
    info : dict
        A dictionary containing additional information regarding the parameters used.

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
    * Crepeau, J. C., & Isaacson, L. K. (1991). Spectral Entropy Measurements of Coherent
      Structures in an Evolving Shear Layer. Journal of Non-Equilibrium Thermodynamics, 16(2).
      doi:10.1515/jnet.1991.16.2.137

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    # Power-spectrum density (PSD) (the actual sampling rate does not matter, unless the
    # frequency range is restricted with min_frequency or max_frequency)
    kwargs.setdefault("sampling_rate", 1000)
    psd = signal_psd(signal, **kwargs)

    # Cut into bins
    if isinstance(bins, int):
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
    se, _ = entropy_shannon(freq=psd["Power"].values)

    # Normalize
    se /= np.log2(len(psd))  # between 0 and 1

    return se, {"PSD": psd}
