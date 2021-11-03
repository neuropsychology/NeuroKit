import numpy as np
import pandas as pd

from .complexity_hjorth import complexity_hjorth
from .complexity_hurst import complexity_hurst
from .complexity_lempelziv import complexity_lempelziv
from .complexity_rr import complexity_rr
from .entropy_approximate import entropy_approximate
from .entropy_cumulative_residual import entropy_cumulative_residual
from .entropy_differential import entropy_differential
from .entropy_multiscale import entropy_multiscale
from .entropy_permutation import entropy_permutation
from .entropy_sample import entropy_sample
from .entropy_spectral import entropy_spectral
from .entropy_svd import entropy_svd
from .fractal_katz import fractal_katz
from .fractal_nld import fractal_nld
from .fractal_petrosian import fractal_petrosian
from .fractal_psdslope import fractal_psdslope
from .fractal_sda import fractal_sda
from .fractal_sevcik import fractal_sevcik
from .information_fisher import fisher_information


def complexity(signal, which=["fast", "medium", "slow"], delay=1, dimension=2, tolerance="default"):
    """Complexity Analysis

    This convenience function can be used to run a large number of complexity metrics. Does not
    include Recurrence Quantification Analysis (RQA, ``nk.complexity_rqa()``) which currently requires an additional
    dependency.

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
    fractal_sevcik, fisher_information, complexity_hjorth, complexity_rqa

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> df, info = nk.complexity(signal, which = ["fast", "medium"])
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
        df["DiffEn"], info["DiffEn"] = entropy_differential(signal)
        df["PEn"], info["PEn"] = entropy_permutation(signal, dimension=dimension, delay=delay)
        df["SpEn"], info["SpEn"] = entropy_spectral(signal)
        df["SVDEn"], info["SVDEn"] = entropy_svd(signal, delay=delay, dimension=dimension)

        # Other
        df["FI"], info["FI"] = fisher_information(signal, delay=delay, dimension=dimension)
        df["Hjorth"], info["Hjorth"] = complexity_hjorth(signal)
        df["RR"], info["RR"] = complexity_rr(signal)

    if "medium" in which:

        # Fractal Dimension
        df["NLD"], info["NLD"] = fractal_nld(signal)
        df["SDA"], info["SDA"] = fractal_sda(signal)
        df["PSDslope"], info["PSDslope"] = fractal_psdslope(signal)  # SR?

        # Entropy
        df["ApEn"], info["ApEn"] = entropy_approximate(
            signal, delay=delay, dimension=dimension, tolerance=tolerance
        )
        df["CREn"], info["CREn"] = entropy_cumulative_residual(signal)
        df["MSPEn"], info["MSPEn"] = entropy_permutation(
            signal, dimension=dimension, delay=delay, scale="default"
        )
        df["WPEn"], info["WPEn"] = entropy_permutation(
            signal, dimension=dimension, delay=delay, weighted=True
        )
        df["SampEn"], info["SampEn"] = entropy_sample(
            signal, dimension=dimension, delay=delay, tolerance=tolerance
        )
        df["MSE"], info["MSE"] = entropy_multiscale(
            signal, dimension=dimension, tolerance=tolerance
        )

        # Other
        df["Hurst"], info["Hurst"] = complexity_hurst(signal)
        df["LPZ"], info["LPZ"] = complexity_lempelziv(signal, dimension=dimension, delay=delay)
        df["PLPZ"], info["PLPZ"] = complexity_lempelziv(
            signal, dimension=dimension, delay=delay, permutation=True
        )

        # PSDSLOPE LZC PLZC

    return pd.DataFrame.from_dict(df, orient="index").T, info
