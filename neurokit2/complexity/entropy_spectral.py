import numpy as np
import pandas as pd

from ..signal import signal_psd


def entropy_spectral(signal, sampling_rate, **kwargs):
    """Spectral Entropy (SpEn)

    Spectral entropy (SE or SpEn) treats the signal's normalized power distribution in the frequency domain as a probability distribution, and calculates the Shannon entropy of it.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).

    Returns
    -------
    SpEn : float
        Spectral Entropy
    info : dict
        A dictionary containing additional information regarding the parameters used.
    **kwargs
        Other arguments to be passed to ``signal_psd()`` (such as 'method').

    See Also
    --------
    entropy_shannon, signal_psd

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)
    >>>
    >>> # Spectral Entropy
    >>> SpEn, info = nk.entropy_spectral(signal, sampling_rate=200)
    >>> SpEn
    >>>
    >>> SpEn, info = nk.entropy_spectral(signal, sampling_rate=200, method='fft')
    >>> SpEn

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Power-spectrum density (PSD)
    psd = signal_psd(signal, sampling_rate=sampling_rate, **kwargs)
    psd = psd[psd["Power"] > 0]

    # Compute Shannon entropy
    se = -np.sum(psd["Power"] * np.log2(psd["Power"]))

    # Normalize by the number of samples
    # se /= np.log2(len(psd))  # TODO: Not sure what's the rationale of that

    return se, {"Sampling_Rate": sampling_rate, "PSD": psd}
