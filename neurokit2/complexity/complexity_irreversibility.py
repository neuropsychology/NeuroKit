from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from .utils_complexity_embedding import complexity_embedding


def complexity_irreversibility(signal, delay=1):
    """**Time Irreversibility (Trev and Ordinal Contrasts)**

    A process is time-reversible when its statistical properties are the same whether time runs
    forward or backward. Linear Gaussian processes (and static, monotonic transformations of them)
    are reversible, so that time irreversibility is a marker of nonlinear dynamics, or of systems
    out of equilibrium. For instance, a signal that rises slowly and falls quickly (or the
    opposite) is irreversible. Irreversibility is often reduced in physiological signals with
    disease and aging (e.g., heart rate; Costa et al., 2005).

    This function computes indices that are 0 (in the population) for reversible processes, and
    whose sign changes when the signal is reversed in time:

    * **Trev**: the skewness of the increments :math:`d_t = x_{t + \\tau} - x_t` of the signal
      (Schreiber & Schmitz, 2000), :math:`\\langle d^3 \\rangle / \\langle d^2 \\rangle^{3/2}`
      (the ``trev`` feature of *hctsa*; its numerator is one of the *catch22* features).
      It is positive when large rises and small falls dominate (e.g., a signal that rises quickly
      and falls slowly), and negative in the opposite case. In a stationary signal, its numerator
      is also -3 times the asymmetry statistic :math:`\\langle x_t x_{t + \\tau}^2 - x_t^2
      x_{t + \\tau} \\rangle` of Kantz and Schreiber (2004).
    * **Ordinal contrasts** (Bandt, 2023): differences between the probabilities of ordinal
      patterns of 3 samples (``delay`` samples apart) and of their time-reversed counterparts,
      which are based on ranks only, and therefore insensitive to the distribution of the values
      and robust to outliers. With :math:`p_{\\pi}` the probability of the pattern :math:`\\pi`,
      where :math:`\\pi` gives the ranks of the 3 samples (e.g., *132* for
      :math:`x_1 < x_3 < x_2`):

      * The **up-down balance** :math:`\\beta = p_{123} - p_{321}`, the difference between the
        probabilities of monotonic increases and decreases. It also changes sign when the signal
        is flipped upside down.
      * The **rotational asymmetry** :math:`\\gamma = p_{213} + p_{231} - p_{132} - p_{312}`, which
        does not change when the signal is flipped upside down, and can therefore detect the
        irreversibility of signals symmetric in amplitude.

      The vectors with equal values (ties), frequent in quantized signals, are left out, as their
      pattern is not defined.

    The significance of these indices can be assessed by comparing them to those of surrogates
    with the same linear properties, which are time-reversible (see :func:`.signal_surrogate`).
    No single statistic detects all forms of irreversibility (Zanin & Papo, 2021).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples,
        between the samples that are compared.

    Returns
    -------
    trev : float
        The Trev index, the skewness of the increments of the signal.
    info : dict
        A dictionary containing additional information, such as the ordinal contrasts
        (``"UpDownBalance"`` and ``"RotationalAsymmetry"``).

    See Also
    --------
    .signal_surrogate, complexity_ordinalpatterns, .hrv_nonlinear

    Examples
    ----------
    .. ipython:: python

      import numpy as np
      import neurokit2 as nk

      # A sawtooth signal (slow rises, fast falls) with noise
      t = np.linspace(0, 10, 2000)
      signal = (t % 1) + np.random.normal(0, 0.1, len(t))

      trev, info = nk.complexity_irreversibility(signal, delay=10)
      trev
      info

      # Time-reversed signal
      trev, info = nk.complexity_irreversibility(signal[::-1], delay=10)
      trev

    References
    ----------
    * Schreiber, T., & Schmitz, A. (2000). Surrogate time series. Physica D: Nonlinear Phenomena,
      142(3-4), 346-382.
    * Kantz, H., & Schreiber, T. (2004). Nonlinear time series analysis (2nd ed.). Cambridge
      University Press.
    * Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Broken asymmetry of the human heartbeat:
      loss of time irreversibility in aging and disease. Physical Review Letters, 95(19), 198102.
    * Bandt, C. (2023). Statistics and contrasts of order patterns in univariate time series.
      Chaos, 33(3), 033124.
    * Zanin, M., & Papo, D. (2021). Algorithmic approaches for assessing irreversibility in time
      series: Review and comparison. Entropy, 23(11), 1474.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if not isinstance(delay, (int, np.integer)) or delay < 1:
        raise ValueError("NeuroKit error: complexity_irreversibility(): `delay` must be an integer of at least 1.")

    signal = np.asarray(signal, dtype=float)
    info = {"Delay": delay}
    if np.any(np.isnan(signal)):
        warn("The signal contains missing values. Returning NaN.", category=NeuroKitWarning)
        return np.nan, {**info, "UpDownBalance": np.nan, "RotationalAsymmetry": np.nan}
    if len(signal) < 2 * delay + 1:
        raise ValueError("NeuroKit error: complexity_irreversibility(): the signal must be longer than 2 * delay samples.")

    # Trev: skewness of the increments
    d = signal[delay:] - signal[:-delay]
    d2 = np.mean(d**2)
    if d2 == 0:
        warn("The signal is constant (at this delay). Returning NaN.", category=NeuroKitWarning)
        trev = np.nan
    else:
        trev = np.mean(d**3) / d2**1.5

    # Ordinal contrasts (Bandt, 2023), from the vectors without ties
    x1, x2, x3 = complexity_embedding(signal, delay=delay, dimension=3).T
    valid = (x1 != x2) & (x1 != x3) & (x2 != x3)
    if np.any(valid):
        x1, x2, x3 = x1[valid], x2[valid], x3[valid]
        p = {
            "123": np.mean((x1 < x2) & (x2 < x3)),
            "321": np.mean((x1 > x2) & (x2 > x3)),
            "132": np.mean((x1 < x3) & (x3 < x2)),
            "213": np.mean((x2 < x1) & (x1 < x3)),
            "231": np.mean((x3 < x1) & (x1 < x2)),
            "312": np.mean((x2 < x3) & (x3 < x1)),
        }
        info["UpDownBalance"] = p["123"] - p["321"]
        info["RotationalAsymmetry"] = p["213"] + p["231"] - p["132"] - p["312"]
    else:
        warn("All the ordinal patterns have ties. The ordinal contrasts are NaN.", category=NeuroKitWarning)
        info["UpDownBalance"] = info["RotationalAsymmetry"] = np.nan

    return trev, info
