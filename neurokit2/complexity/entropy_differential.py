import numpy as np
import pandas as pd
import scipy.stats


def entropy_differential(signal, base=2, **kwargs):
    """**Differential entropy (DiffEn)**

    Differential entropy (DiffEn; also referred to as continuous entropy) started as an
    attempt by Shannon to extend Shannon entropy. However, differential entropy presents some
    issues too, such as that it can be negative even for simple distributions (such as the uniform
    distribution).

    This function can be called either via ``entropy_differential()`` or ``complexity_diffen()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    base: float
        The logarithmic base to use, defaults to ``2``, giving a unit in *bits*. Note that ``scipy.
        stats.entropy()`` uses Euler's number (``np.e``) as default (the natural logarithm), giving
        a measure of information expressed in *nats*.
    **kwargs : optional
        Other arguments passed to ``scipy.stats.differential_entropy()``.

    Returns
    --------
    diffen : float
        The Differential entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute Differential entropy.

    See Also
    --------
    entropy_shannon, entropy_cumulativeresidual, entropy_kl

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal with Laplace Noise
      signal = nk.signal_simulate(duration=2, frequency=5, noise=0.1)

      # Compute Differential Entropy
      diffen, info = nk.entropy_differential(signal)
      diffen


    References
    -----------
    * `scipy.stats.differential_entropy()
      <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.differential_entropy.html>`_
    * https://en.wikipedia.org/wiki/Differential_entropy

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Check if string ('ABBA'), and convert each character to list (['A', 'B', 'B', 'A'])
    if not isinstance(signal, str):
        signal = list(signal)

    if "method" in kwargs:
        method = kwargs["method"]
        kwargs.pop("method")
    else:
        method = "vasicek"

    diffen = scipy.stats.differential_entropy(signal, method=method, base=base, **kwargs)

    return diffen, {"Method": method, "Base": base}
