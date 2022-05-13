import numpy as np
import pandas as pd

from ..signal.signal_psd import signal_psd
from .entropy_shannon import entropy_shannon


def entropy_spectral(signal, c=None, **kwargs):
    """**Spectral Entropy (SpEn)**

    Spectral entropy (SE or SpEn) treats the signal's normalized power distribution in the
    frequency domain as a probability distribution, and calculates the Shannon entropy of it.

    A signal with a single frequency component (i.e., pure sinusoid) produces the smallest entropy.
    On the other hand, a signal with all frequency components of equal power value (white
    noise) produces the greatest entropy.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    c : int
        Number of bins of frequency.
    **kwargs : optional
        Keyword arguments to be passed to ``signal_psd()``.

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
      SpEn, info = nk.entropy_spectral(signal, c=20)
      SpEn

    References
    ----------
    * Crepeau, J. C., & Isaacson, L. K. (1991). Spectral Entropy Measurements of Coherent
      Structures in an Evolving Shear Layer. Journal of Non-Equilibrium Thermodynamics, 16(2).
      doi:10.1515/jnet.1991.16.2.137

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Power-spectrum density (PSD) (actual sampling rate does not matter)
    psd = signal_psd(signal, sampling_rate=1000, method="fft", n=c, **kwargs)["Power"]
    psd /= np.sum(psd)  # area under normalized spectrum should sum to 1 (np.sum(psd["Power"]))

    # Compute Shannon entropy
    se, _ = entropy_shannon(freq=psd)

    # Normalize
    se /= np.log2(len(psd))  # between 0 and 1

    return se, {"PSD": psd}
