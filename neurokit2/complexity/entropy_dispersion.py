from warnings import warn

import numpy as np
import pandas as pd
import scipy.special

from ..misc import NeuroKitWarning
from ..stats import standardize
from .entropy_shannon import entropy_shannon
from .utils_complexity_embedding import complexity_embedding
from .utils_complexity_symbolize import complexity_symbolize


def entropy_dispersion(signal, delay=1, dimension=3, c=6, symbolize="NCDF", fluctuation=False, rho=1, **kwargs):
    """**Dispersion Entropy (DispEn)**

    The Dispersion Entropy (DispEn). Also returns the Reverse Dispersion Entropy (RDEn).

    Dispersion Entropy (Rostaghi & Azami, 2016) maps each sample of the signal to one of *c*
    classes, by default through the normal cumulative distribution function (NCDF) of the
    standardized signal (the class of a value *y* of the NCDF, between 0 and 1, is
    :math:`round(c \\times y + 0.5)`). The vectors of *m* (``dimension``) successive classes,
    spaced by ``delay`` samples, are the *dispersion patterns*, of which there are :math:`c^m`
    possible ones. DispEn is the :func:`Shannon entropy <entropy_shannon>` of the relative
    frequencies :math:`p(\\pi)` of the patterns, divided by its maximum, :math:`\\log(c^m)`, so
    that it lies between 0 and 1:

    .. math::

      DispEn = -\\frac{\\sum_{\\pi} p(\\pi) \\log p(\\pi)}{\\log(c^m)}

    * **Fluctuation-based DispEn** (FDispEn; Azami & Escudero, 2018) uses the differences between
      the neighbouring classes of each pattern instead of the classes themselves (so that patterns
      that differ only by a constant offset are the same). There are :math:`(2c - 1)^{m - 1}`
      possible patterns, by the logarithm of which FDispEn is normalized.
    * **Fine-sorted DispEn** (``symbolize="finesort"``; Fu et al., 2019) adds to each pattern a
      factor that distinguishes the vectors of the same pattern by their largest difference:
      :math:`f = \\lfloor \\max |\\Delta y| / (\\rho \\times SD(|\\Delta x|)) \\rfloor`, where
      :math:`\\Delta y` are the differences between the neighbouring NCDF values of the vector and
      :math:`\\Delta x` the differences between the successive samples of the signal (as defined,
      *f* therefore depends on the amplitude of the signal). With a large ``rho``, *f* is always 0,
      and it is the same as DispEn. The normalization is the same as for DispEn, which the finer
      patterns can exceed (the value can then be above 1).
    * **Reverse Dispersion Entropy** (RDEn; Li et al., 2019) is the squared distance between the
      distribution of the patterns and the uniform distribution, over the *n* possible patterns
      (those that do not occur included), :math:`\\sum_{\\pi} (p(\\pi) - 1/n)^2 = \\sum_{\\pi}
      p(\\pi)^2 - 1/n`, divided by its maximum, :math:`1 - 1/n`. Contrary to DispEn, it is lower
      for more irregular signals (0 when all the patterns are equally frequent).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter. It should
        be at least 2 with ``fluctuation=True`` or ``symbolize="finesort"``.
    c : int
        Number of symbols *c*, at least 2. Rostaghi (2016) recommend in practice a *c* between 4
        and 8.
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. Can be one
        of ``"NCDF"`` (default), ``"finesort"``, or others. See :func:`complexity_symbolize` for
        details.
    fluctuation : bool
        Fluctuation-based Dispersion entropy.
    rho : float
        Tuning parameter of "finesort". Only when ``symbolize="finesort"``.
    **kwargs : optional
        Other keyword arguments, such as the logarithmic ``base`` to use for
        :func:`entropy_shannon` (which does not change the normalized DispEn), or the
        ``random_state`` of :func:`complexity_symbolize` (e.g., for ``symbolize="kmeans"``).

    Returns
    -------
    DispEn : float
        Dispersion Entropy (DispEn) of the signal. It is NaN (with a warning) if the signal contains
        missing values.
    info : dict
        A dictionary containing additional information regarding the parameters used, and the
        Reverse Dispersion Entropy (``"RDEn"``).

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

      # Fluctuation-based DispEn with "finesort"
      dispen, info = nk.entropy_dispersion(signal, c=3, symbolize="finesort", fluctuation=True)
      dispen

    References
    ----------
    * Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for time-series analysis.
      IEEE Signal Processing Letters, 23(5), 610-614.
    * Azami, H., & Escudero, J. (2018). Amplitude- and fluctuation-based dispersion entropy.
      Entropy, 20(3), 210.
    * Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new complexity measure for
      sensor signal. Sensors, 19(23), 5203.
    * Fu, W., Tan, J., Xu, Y., Wang, K., & Chen, T. (2019). Fault diagnosis for rolling bearings
      based on fine-sorted dispersion entropy and SVM optimized with mutation SCA-PSO. Entropy,
      21(4), 404.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if c < 2:
        raise ValueError("The number of symbols c of Dispersion Entropy should be at least 2.")
    if dimension < 2 and (fluctuation is True or symbolize == "finesort"):
        raise ValueError("The dimension of fluctuation-based and fine-sorted Dispersion Entropy should be at least 2.")

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay, "c": c, "Symbolization": symbolize, "Fluctuation": fluctuation}

    signal = np.asarray(signal, dtype=float)
    if np.any(np.isnan(signal)):
        warn(
            "The signal contains missing values, which cannot be mapped to a class. Returning NaN.",
            category=NeuroKitWarning,
        )
        info["RDEn"] = np.nan
        return np.nan, info

    # Symbolization and embedding
    if symbolize == "finesort":
        symbolic = complexity_symbolize(signal, method="NCDF", c=c)
        # Fine-sorting factor (Fu et al., 2019): the largest absolute difference between the
        # neighbouring NCDF values of each vector, relative to rho times the SD of the absolute
        # differences between the successive samples of the signal (it is 0 if these differences
        # are all equal in size, e.g., for a constant signal, as it would not distinguish anything)
        sd = rho * np.std(np.abs(np.diff(signal)))
        if sd > 0:
            ncdf = complexity_embedding(scipy.special.ndtr(standardize(signal)), dimension=dimension, delay=delay)
            finesort = np.floor(np.max(np.abs(np.diff(ncdf, axis=1)), axis=1) / sd)
        else:
            finesort = np.zeros(len(signal) - (dimension - 1) * delay)
    else:
        symbolic = complexity_symbolize(signal, method=symbolize, c=c, random_state=kwargs.get("random_state"))
    embedded = complexity_embedding(symbolic, dimension=dimension, delay=delay)

    # Fluctuation: differences between the neighbouring classes, from -(c - 1) to c - 1
    if fluctuation is True:
        embedded = np.diff(embedded, axis=1)
        n_patterns = (2 * c - 1) ** (dimension - 1)
    else:
        n_patterns = c**dimension

    # The fine-sorting factor is an additional element of the pattern
    if symbolize == "finesort":
        embedded = np.hstack((embedded, finesort[:, np.newaxis]))

    _, freq = np.unique(embedded, return_counts=True, axis=0)
    freq = freq / freq.sum()

    DispEn, _ = entropy_shannon(freq=freq, **kwargs)

    # Normalize by the maximum entropy (all the patterns equally frequent), in the same logarithmic base
    DispEn = DispEn / (np.log(n_patterns) / np.log(kwargs.get("base", 2)))

    # Reverse Dispersion Entropy (RDEn), over all the possible patterns (the patterns that do not
    # occur contribute (1 / n_patterns)^2 each), normalized by its maximum
    rden = np.sum(freq**2) - 1 / n_patterns
    info["RDEn"] = rden / (1 - 1 / n_patterns)

    return DispEn, info
