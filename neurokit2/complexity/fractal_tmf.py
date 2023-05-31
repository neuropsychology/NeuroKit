import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from packaging import version

from ..signal import signal_surrogate
from .fractal_dfa import fractal_dfa


def fractal_tmf(signal, n=40, show=False, **kwargs):
    """**Multifractal Nonlinearity (tMF)**

    The Multifractal Nonlinearity index (*t*\\MF) is the *t*\\-value resulting from the comparison
    of the multifractality of the signal (measured by the spectrum width, see
    :func:`.fractal_dfa`) with the multifractality of linearized
    :func:`surrogates <.signal_surrogate>` obtained by the IAAFT method (i.e., reshuffled series
    with comparable linear structure).

    This statistics grows larger the more the original series departs from the multifractality
    attributable to the linear structure of IAAFT surrogates. When p-value reaches significance, we
    can conclude that the signal's multifractality encodes processes that a linear contingency
    cannot.

    This index provides an extension of the assessment of multifractality, of which the
    multifractal spectrum is by itself a measure of heterogeneity, rather than interactivity.
    As such, it cannot alone be used to assess the specific presence of cascade-like interactivity
    in the time series, but must be compared to the spectrum of a sample of its surrogates.

    .. figure:: ../img/bell2019.jpg
       :alt: Figure from Bell et al. (2019).
       :target: https://doi.org/10.3389/fphys.2019.00998

    Both significantly negative and positive values can indicate interactivity, as any difference
    from the linear structure represented by the surrogates is an indication of nonlinear
    contingence. Indeed, if the degree of heterogeneity for the original series is significantly
    less than for the sample of linear surrogates, that is no less evidence of a failure of
    linearity than if the degree of heterogeneity is significantly greater.

    .. note::

        Help us review the implementation of this index by checking-it out and letting us know
        wether it is correct or not.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    n : int
        Number of surrogates. The literature uses values between 30 and 40.
    **kwargs : optional
        Other arguments to be passed to :func:`.fractal_dfa`.

    Returns
    -------
    float
        tMF index.
    info : dict
        A dictionary containing additional information, such as the p-value.

    See Also
    --------
    fractal_dfa, .signal_surrogate

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=1, sampling_rate=200, frequency=[5, 6, 12], noise=0.2)

      # Compute tMF
      @savefig p_fractal_tmf.png scale=100%
      tMF, info = nk.fractal_tmf(signal, n=100, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      tMF  # t-value
      info["p"]  # p-value


    References
    ----------
    * Ihlen, E. A., & Vereijken, B. (2013). Multifractal formalisms of human behavior. Human
      movement science, 32(4), 633-651.
    * Kelty-Stephen, D. G., Palatinus, K., Saltzman, E., & Dixon, J. A. (2013). A tutorial on
      multifractality, cascades, and interactivity for empirical time series in ecological science.
      Ecological Psychology, 25(1), 1-62.
    * Bell, C. A., Carver, N. S., Zbaracki, J. A., & Kelty-Stephen, D. G. (2019). Non-linear
      amplification of variability through interaction across scales supports greater accuracy in
      manual aiming: evidence from a multifractal analysis with comparisons to linear surrogates in
      the fitts task. Frontiers in physiology, 10, 998.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    info = {}

    w0 = fractal_dfa(signal, multifractal=True, show=False)[0]["Width"]

    w = np.zeros(n)
    for i in range(n):
        surro = signal_surrogate(signal, method="IAAFT")
        w[i] = float(fractal_dfa(surro, multifractal=True, show=False)[0]["Width"].iloc[0])

    # Run t-test
    # TODO: adjust in the future
    if version.parse(scipy.__version__) < version.parse("1.10.0"):
        t, p = scipy.stats.ttest_1samp(w, w0)
        t = t[0]
        t = t.item()
        info["p"] = p[0]
    else:
        t, info["p"] = scipy.stats.ttest_1samp(w, w0)

    if show is True:
        pd.Series(w).plot(kind="density", label="Width of surrogates")
        plt.axvline(x=w0.values, c="red", label="Width of original signal")
        plt.title(f"tMF = {t:.2f}, p = {info['p']:.2f}")
        plt.legend()

    return t, info
