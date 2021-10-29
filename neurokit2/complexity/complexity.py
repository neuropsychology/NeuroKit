import numpy as np
import pandas as pd

from .complexity_hjorth import complexity_hjorth
from .entropy_differential import entropy_differential
from .entropy_permutation import entropy_permutation
from .entropy_svd import entropy_svd
from .fractal_katz import fractal_katz
from .fractal_petrosian import fractal_petrosian
from .fractal_sevcik import fractal_sevcik
from .information_fisher import fisher_information


def complexity(signal, which=["fast"], delay=1, dimension=2):
    """Complexity Analysis

    This convenience function can be used to run a large number of complexity metrics.

    The categorization by "computation time" is based on our preliminary `benchmarking study
    <https://neurokit2.readthedocs.io/en/latest/studies/complexity_benchmark.html>`_.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    which : list
        What metrics to compute, based on their computation time. Currently, only 'fast' is supported.
    delay : int
        See for example :func:`entropy_permutation`.
    dimension : int
        See for example :func:`entropy_permutation`.

    Returns
    --------
    df : pd.DataFrame
        A dataframe with one row containing the results for each metric as columns.
    info : dict
        A dictionary containing additional information.

    See Also
    --------
    entropy_permutation, entropy_differential, entropy_svd, fractal_katz, fractal_petrosian,
    fractal_sevcik, fisher_information, complexity_hjorth

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> df, info = nk.complexity(signal, which = "fast")
    >>> df #doctest: +SKIP

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Sanitize which
    if isinstance(which, str):
        which = [which]

    # Initialize
    df = {}
    info = {}

    # Fast ======================================================================================
    if "fast" in which:

        # Fractal Dimension
        df["PFD"], info["PFD"] = fractal_petrosian(signal)
        df["KFD"], info["KFD"] = fractal_katz(signal)
        df["SFD"], info["SFD"] = fractal_sevcik(signal)

        # Entropy
        df["SVDEn"], info["SVDEn"] = entropy_svd(signal, delay=delay, dimension=dimension)
        df["PEn"], info["PEn"] = entropy_permutation(signal, dimension=dimension, delay=delay)
        df["DiffEn"], info["DiffEn"] = entropy_differential(signal)

        # Other
        df["FI"], info["FI"] = fisher_information(signal, delay=delay, dimension=dimension)
        df["Hjorth"], info["Hjorth"] = complexity_hjorth(signal)

    return pd.DataFrame.from_dict(df, orient="index").T, info
