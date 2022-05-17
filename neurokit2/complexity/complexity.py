import numpy as np
import pandas as pd

from .complexity_hjorth import complexity_hjorth
from .complexity_lempelziv import complexity_lempelziv
from .complexity_lyapunov import complexity_lyapunov
from .complexity_relativeroughness import complexity_relativeroughness
from .entropy_approximate import entropy_approximate
from .entropy_cumulativeresidual import entropy_cumulativeresidual
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
from .fractal_hurst import fractal_hurst
from .fractal_katz import fractal_katz
from .fractal_nld import fractal_nld
from .fractal_petrosian import fractal_petrosian
from .fractal_psdslope import fractal_psdslope
from .fractal_sda import fractal_sda
from .fractal_sevcik import fractal_sevcik
from .information_fisher import fisher_information


def complexity(signal, which=["fast", "medium"], delay=1, dimension=2, tolerance="sd", **kwargs):
    """**Complexity and Chaos Analysis**

    Measuring the complexity of a signal refers to the quantification of various aspects related to
    concepts such as **chaos**, **entropy**, **unpredictability**, and **fractal dimension**.

    .. tip::

        We recommend checking our open-access `preprint <https://psyarxiv.com/f8k3x/>`_ for an
        introduction to **fractal physiology** and its application in neuroscience.

    There are many indices that have been developped and used to assess the complexity of signals,
    and all of them come with different specificities and limitations. While they should be used in
    an informed manner, it is also convenient to have a single function that can compute multiple
    indices at once.

    The ``nk.complexity()`` function can be used to compute a large number of complexity metrics
    and features. While this is great for exploratory analyses, we recommend running each function
    separately, to gain more control over the parameters and information that you get.

    .. warning::

        The indices included in this function will be subjected to change in future versions,
        depending on what the literature suggests. We recommend using this function only for quick
        exploratory analyses, but then replacing it by the calls to the individual functions.

    The categorization by "computation time" is based on our preliminary `benchmarking study
    <https://neuropsychology.github.io/NeuroKit/studies/complexity_benchmark.html>`_ results:

    .. figure:: ../../studies/complexity_benchmark/figures/computation_time-1.png
       :alt: Complexity Benchmark (Makowski).
       :target: https://neuropsychology.github.io/NeuroKit/studies/complexity_benchmark.html

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    which : list
        What metrics to compute, based on their computation time. Can be ``"fast"``, ``"medium"``,
        or ``"slow"``.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance` to estimate the optimal value for this parameter.

    Returns
    --------
    df : pd.DataFrame
        A dataframe with one row containing the results for each metric as columns.
    info : dict
        A dictionary containing additional information.

    See Also
    --------
    complexity_delay, complexity_dimension, complexity_tolerance

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
      df, info = nk.complexity(signal, which = "slow", k_max=6, q=range(-2, 3))
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
      @suppress
      plt.close()

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
      @suppress
      plt.close()

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
        df["RR"], info["RR"] = complexity_relativeroughness(signal)

    if "medium" in which:

        # Fractal Dimension
        df["NLD"], info["NLD"] = fractal_nld(signal, corrected=False)
        df["SDA"], info["SDA"] = fractal_sda(signal)
        df["PSDslope"], info["PSDslope"] = fractal_psdslope(signal)

        # Entropy
        df["CREn"], info["CREn"] = entropy_cumulativeresidual(signal)
        df["ApEn"], info["ApEn"] = entropy_approximate(
            signal, delay=delay, dimension=dimension, tolerance=tolerance
        )
        df["SampEn"], info["SampEn"] = entropy_sample(
            signal, dimension=dimension, delay=delay, tolerance=tolerance
        )
        df["MSEn"], info["MSEn"] = entropy_multiscale(
            signal, dimension=dimension, tolerance=tolerance, method="MSEn"
        )
        df["MSPEn"], info["MSPEn"] = entropy_permutation(
            signal, dimension=dimension, delay=delay, scale="default"
        )
        df["WPEn"], info["WPEn"] = entropy_permutation(
            signal, dimension=dimension, delay=delay, weighted=True
        )

        # Other
        df["Hurst"], info["Hurst"] = fractal_hurst(signal)
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
        df["RCMSEn"], info["RCMSEn"] = entropy_multiscale(
            signal, dimension=dimension, tolerance=tolerance, method="RCMSEn"
        )
        df["FuzzyMSEn"], info["FuzzyMSEn"] = entropy_multiscale(
            signal, dimension=dimension, tolerance=tolerance, fuzzy=True
        )
        df["FuzzyRCMSEn"], info["FuzzyRCMSEn"] = entropy_multiscale(
            signal, dimension=dimension, tolerance=tolerance, method="RCMSEn", fuzzy=True
        )
        df["RangeEn"], info["RangeEn"] = entropy_range(
            signal, dimension=dimension, delay=delay, tolerance=tolerance
        )

        # Other
        df["LLE"], info["LLE"] = complexity_lyapunov(signal, dimension=dimension, delay=delay)
        df["DFA"], info["DFA"] = fractal_dfa(signal)
        mfdfa, _ = fractal_dfa(signal, multifractal=True, **kwargs)
        for k in mfdfa.columns:
            df["MFDFA_" + k] = mfdfa[k].values[0]

    # Prepare output
    df = pd.DataFrame.from_dict(df, orient="index").T  # Convert to dataframe
    df = df.reindex(sorted(df.columns), axis=1)  # Reorder alphabetically

    return df, info
