import numpy as np
import pandas as pd

from .complexity_hjorth import complexity_hjorth
from .entropy_attention import entropy_attention
from .entropy_bubble import entropy_bubble
from .entropy_multiscale import entropy_multiscale
from .entropy_permutation import entropy_permutation
from .entropy_svd import entropy_svd
from .fractal_dfa import fractal_dfa
from .fractal_linelength import fractal_linelength


def complexity(signal, which="makowski2022", delay=1, dimension=2, tolerance="sd", **kwargs):
    """**Complexity and Chaos Analysis**

    Measuring the complexity of a signal refers to the quantification of various aspects related to
    concepts such as **chaos**, **entropy**, **unpredictability**, and **fractal dimension**.

    .. tip::

        We recommend checking our open-access `review <https://onlinelibrary.wiley.com/doi/10.1111/ejn.15800>`_ for an
        introduction to **fractal physiology** and its application in neuroscience.

    There are many indices that have been developed and used to assess the complexity of signals,
    and all of them come with different specificities and limitations. While they should be used in
    an informed manner, it is also convenient to have a single function that can compute multiple
    indices at once.

    The ``nk.complexity()`` function can be used to compute a useful subset of complexity metrics
    and features. While this is great for exploratory analyses, we recommend running each function
    separately, to gain more control over the parameters and information that you get.

    .. warning::

        The indices included in this function will be subjected to change in future versions,
        depending on what the literature suggests. We recommend using this function only for quick
        exploratory analyses, but then replacing it by the calls to the individual functions.

        Check-out our `open-access study <https://onlinelibrary.wiley.com/doi/10.1111/ejn.15800>`_
        explaining the selection of indices.

    The categorization by "computation time" is based on `our study
    <https://www.mdpi.com/1099-4300/24/8/1036>`_ results:

    .. figure:: https://raw.githubusercontent.com/DominiqueMakowski/ComplexityStructure/main/figures/time1-1.png
       :alt: Complexity Benchmark (Makowski).
       :target: https://www.mdpi.com/1099-4300/24/8/1036

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    which : list
        What metrics to compute. Can be "makowski2022".
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

      # Compute selection of complexity metrics (Makowski et al., 2022)
      df, info = nk.complexity(signal, which = "makowski2022")
      df

    * **Example 2**: Compute complexity over time

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

    References
    ----------
    * Lau, Z. J., Pham, T., Chen, S. H. A., & Makowski, D. (2022). Brain entropy, fractal
      dimensions and predictability: A review of complexity measures for EEG in healthy and
      neuropsychiatric populations. European Journal of Neuroscience, 1-23.
    * Makowski, D., Te, A. S., Pham, T., Lau, Z. J., & Chen, S. H. (2022). The Structure of Chaos:
      An Empirical Comparison of Fractal Physiology Complexity Indices Using NeuroKit2. Entropy, 24
      (8), 1036.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Initialize
    df = {}
    info = {}

    # Fast ======================================================================================
    if which in ["makowski2022", "makowski"]:

        df["LL"], info["LL"] = fractal_linelength(signal)
        df["Hjorth"], info["Hjorth"] = complexity_hjorth(signal)
        df["AttEn"], info["AttEn"] = entropy_attention(signal)
        df["SVDEn"], info["SVDEn"] = entropy_svd(signal, delay=delay, dimension=dimension)
        df["BubbEn"], info["BubbEn"] = entropy_bubble(
            signal, delay=delay, dimension=dimension, **kwargs
        )
        df["CWPEn"], info["CWPEn"] = entropy_permutation(
            signal,
            delay=delay,
            dimension=dimension,
            corrected=True,
            weighted=True,
            conditional=True,
            **kwargs
        )
        df["MSPEn"], info["MSPEn"] = entropy_multiscale(
            signal, dimension=dimension, method="MSPEn", **kwargs
        )

        mfdfa, _ = fractal_dfa(signal, multifractal=True, **kwargs)
        for k in mfdfa.columns:
            df["MFDFA_" + k] = mfdfa[k].values[0]

    # Prepare output
    df = pd.DataFrame.from_dict(df, orient="index").T  # Convert to dataframe
    df = df.reindex(sorted(df.columns), axis=1)  # Reorder alphabetically

    return df, info
