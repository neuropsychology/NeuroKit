import numpy as np
import pandas as pd

from .entropy_shannon import entropy_shannon
from .entropy_symbolicdynamic import _complexity_symbolization


def entropy_dispersion(
    signal, delay=1, dimension=3, c=6, method="NCDF", fluctuation=False, **kwargs
):
    """**Dispersion Entropy (DispEn)**

    The Dispersion Entropy (DispEn). Also returns the Reverse Dispersion Entropy (RDEn).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension()` to estimate the optimal value for this parameter.
    c : int
        Number of symbols *c*. Rostaghi (2016) recommend in practice a *c* between 4 and 8.
    method : str
        Method for transforming the signal into a symbolic sequence. Can be one of ``"NCDF"``
        (default), ``"MEP"``, ``"linear"``, ``"uniform"``, ``"kmeans"``, ``"equal"`` or
        ``"finesort"``.
    **kwargs : optional
        Other keyword arguments (currently not used).

    Returns
    -------
    DispEn : float
        Dispersion Entropy (DispEn) of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon, entropy_multiscale, entropy_symbolicdynamic

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute Dispersion Entropy (DispEn)
      dispen, info = nk.entropy_dispersion(signal, c=3)
      dispen

      # Get Reverse Dispersion Entropy (RDEn)
      info["RDEn"]

      # Fluctuation-based with another symbolization algorithm
      dispen, info = nk.entropy_dispersion(signal, c=3, method="finesort", fluctuation=True)
      dispen

    References
    ----------
    * Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for time-series analysis.
      IEEE Signal Processing Letters, 23(5), 610-614.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay, "c": c, "Method": method}

    # Symbolization and embedding
    embedded = _complexity_symbolization(
        signal,
        dimension=dimension,
        delay=delay,
        c=c,
        method=method,
        **kwargs,
    )

    if fluctuation is True:
        embedded = np.diff(embedded, axis=1)

    _, freq = np.unique(embedded, return_counts=True, axis=0)
    freq = freq / freq.sum()

    DispEn, _ = entropy_shannon(freq=freq, **kwargs)

    # Reverse Dispersion Entropy (RDEn)
    if fluctuation is True:
        rden = np.sum((freq - (1 / ((2 * c - 1) ** (dimension - 1)))) ** 2)
    else:
        rden = np.sum((freq - (1 / (c ** dimension))) ** 2)

    # Normalize
    DispEn = DispEn / np.log(c ** dimension)
    info["RDEn"] = rden / (1 - (1 / (c ** dimension)))

    return DispEn, info
