import numpy as np
import pandas as pd

from .complexity_hjorth import complexity_hjorth
from .complexity_hurst import complexity_hurst
from .complexity_lempelziv import complexity_lempelziv
from .complexity_lyapunov import complexity_lyapunov
from .complexity_rr import complexity_rr
from .entropy_approximate import entropy_approximate
from .entropy_cumulative_residual import entropy_cumulative_residual
from .entropy_differential import entropy_differential
from .entropy_fuzzy import entropy_fuzzy
from .entropy_multiscale import entropy_multiscale
from .entropy_permutation import entropy_permutation
from .entropy_range import entropy_range
from .entropy_sample import entropy_sample
from .entropy_shannon import entropy_shannon
from .entropy_spectral import entropy_spectral
from .entropy_svd import entropy_svd
from .fractal_correlation import fractal_correlation
from .fractal_dfa import fractal_dfa
from .fractal_higuchi import fractal_higuchi
from .fractal_katz import fractal_katz
from .fractal_nld import fractal_nld
from .fractal_petrosian import fractal_petrosian
from .fractal_psdslope import fractal_psdslope
from .fractal_sda import fractal_sda
from .fractal_sevcik import fractal_sevcik
from .information_fisher import fisher_information


def complexity(
    signal, which=["fast", "medium"], delay=1, dimension=2, tolerance="default", **kwargs
):
    """**Automated Complexity and Chaos Analysis**

    This function can be used to compute a large number of complexity metrics and features. For more
    control, you can run each function separately. Note that it does not include Recurrence
    Quantification Analysis (RQA, ``nk.complexity_rqa()``) which currently requires an additional
    dependency.

    The categorization by "computation time" is based on our preliminary `benchmarking study
    <https://neurokit2.readthedocs.io/en/latest/studies/complexity_benchmark.html>`_.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    which : list
        What metrics to compute, based on their computation time. Can be ``"fast"``, ``"medium"``,
        or ``"slow"``.
    delay : int
        See for example :func:`entropy_sample`.
    dimension : int
        See for example :func:`entropy_sample`.
    tolerance : float
        See for example :func:`entropy_sample`.

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
    * **Example 1**: Compute fast and medium-fast complexity metrics

    .. ipython:: python

      import neurokit2 as nk

      # Simulate a signal of 3 seconds
      signal = nk.signal_simulate(duration=3, frequency=[5, 10])

      # Fast metrics
      df, info = nk.complexity(signal, which = ["fast", "medium"])
      df

    * **Example 2**: Compute slow complexity metrics

    .. ipython:: python

      # Slow, with specific parameters for Higuchi and MFDFA
      df, info = nk.complexity(signal, which = "slow", k_max=6, q=range(-2, 2))
      df

    * **Example 3**: Compute complexity over time

    .. ipython:: python

      import numpy as np
      import pandas as pd
      import neurokit2 as nk

      # Create dynamically varying noise
      amount_noise = nk.signal_simulate(duration=2, frequency=0.9)
      amount_noise = nk.rescale(amount_noise, [0, 0.5])
      noise = np.random.uniform(0, 2, len(amount_noise)) * amount_noise

      # Add to simple signal
      signal = noise + nk.signal_simulate(duration=2, frequency=5)

      @savefig p_complexity1.png scale=100%
      nk.signal_plot(signal, sampling_rate = 1000)

    .. ipython:: python

      # Create function-wrappers that only return the index value
      pfd = lambda x: nk.fractal_petrosian(x)[0]
      kfd = lambda x: nk.fractal_katz(x)[0]
      sfd = lambda x: nk.fractal_sevcik(x)[0]
      svden = lambda x: nk.entropy_svd(x)[0]
      fisher = lambda x: -1 * nk.fisher_information(x)[0]  # FI is anticorrelated with complexity


      # Use them in a rolling window
      rolling_kfd = pd.Series(signal).rolling(500, min_periods = 300, center=True).apply(kfd)
      rolling_pfd = pd.Series(signal).rolling(500, min_periods = 300, center=True).apply(pfd)
      rolling_sfd = pd.Series(signal).rolling(500, min_periods = 300, center=True).apply(sfd)
      rolling_svden = pd.Series(signal).rolling(500, min_periods = 300, center=True).apply(svden)
      rolling_fisher = pd.Series(signal).rolling(500, min_periods = 300, center=True).apply(fisher)

      @savefig p_complexity2.png scale=100%
      nk.signal_plot([signal,
                      rolling_kfd.values,
                      rolling_pfd.values,
                      rolling_sfd.values,
                      rolling_svden.values,
                      rolling_fisher],
                     labels = ["Signal",
                               "Petrosian Fractal Dimension",
                               "Katz Fractal Dimension",
                               "Sevcik Fractal Dimension",
                               "SVD Entropy",
                               "Fisher Information"],
                     sampling_rate = 1000,
                     standardize = True)

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
        df["ShanEn"], info["ShanEn"] = entropy_shannon(signal)
        df["SpEn"], info["SpEn"] = entropy_spectral(signal)
        df["SVDEn"], info["SVDEn"] = entropy_svd(signal, delay=delay, dimension=dimension)

        # Other
        df["FI"], info["FI"] = fisher_information(signal, delay=delay, dimension=dimension)
        df["Hjorth"], info["Hjorth"] = complexity_hjorth(signal)
        df["RR"], info["RR"] = complexity_rr(signal)

    if "medium" in which:

        # Fractal Dimension
        df["NLD"], info["NLD"] = fractal_nld(signal)
        if len(signal) >= 1024:
            df["SDA"], info["SDA"] = fractal_sda(signal)
        df["PSDslope"], info["PSDslope"] = fractal_psdslope(signal)

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
        df["LZC"], info["LZC"] = complexity_lempelziv(signal, dimension=dimension, delay=delay)
        df["PLZC"], info["PLZC"] = complexity_lempelziv(
            signal, dimension=dimension, delay=delay, permutation=True
        )

    if "slow" in which:

        # Fractal Dimension
        df["CD"], info["CD"] = fractal_correlation(signal, delay=delay, dimension=dimension)
        df["HFD"], info["HFD"] = fractal_higuchi(signal, **kwargs)

        # Entropy
        df["FuzzyEn"], info["FuzzyEn"] = entropy_fuzzy(
            signal, dimension=dimension, delay=delay, tolerance=tolerance
        )
        df["FuzzyMSE"], info["FuzzyMSE"] = entropy_multiscale(
            signal, dimension=dimension, tolerance=tolerance, fuzzy=True
        )
        df["FuzzyRCMSE"], info["FuzzyRCMSE"] = entropy_multiscale(
            signal, dimension=dimension, tolerance=tolerance, refined=True, fuzzy=True
        )
        df["RCMSE"], info["RCMSE"] = entropy_multiscale(
            signal, dimension=dimension, tolerance=tolerance, refined=True
        )
        df["RangeEn"], info["RangeEn"] = entropy_range(
            signal, dimension=dimension, delay=delay, tolerance=tolerance
        )

        # Other
        df["DFA"], info["DFA"] = fractal_dfa(signal)
        _, info["MFDFA"] = fractal_dfa(signal, multifractal=True, **kwargs)
        df["MFDFA_ExpRange"] = info["MFDFA"]["ExpRange"]
        df["MFDFA_ExpMean"] = info["MFDFA"]["ExpMean"]
        df["MFDFA_DimRange"] = info["MFDFA"]["DimRange"]
        df["MFDFA_DimMean"] = info["MFDFA"]["DimMean"]
        df["LLE"], info["LLE"] = complexity_lyapunov(signal, dimension=dimension, delay=delay)

    # Prepare output
    df = pd.DataFrame.from_dict(df, orient="index").T  # Convert to dataframe
    df = df.reindex(sorted(df.columns), axis=1)  # Reorder alphabetically

    return df, info
