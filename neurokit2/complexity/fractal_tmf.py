import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from ..misc import check_random_state
from ..signal import signal_surrogate
from .fractal_dfa import fractal_dfa


def fractal_tmf(signal, n=40, show=False, random_state=None, **kwargs):
    """**Multifractal Nonlinearity (tMF)**

    The Multifractal Nonlinearity index (*t*\\MF) is the *t*\\-value resulting from the comparison
    of the multifractality of the signal (measured by the spectrum width, see
    :func:`.fractal_dfa`) with the multifractality of linearized
    :func:`surrogates <.signal_surrogate>` obtained by the IAAFT method (i.e., reshuffled series
    with comparable linear structure). It is computed as a one-sample *t*-test (Bell et al., 2019):

    .. math::

      tMF = \\frac{W_{original} - mean(W_{surrogates})}{SE(W_{surrogates})}

    Positive values indicate that the original signal is more multifractal than its linear
    surrogates.

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

    .. note::

        The width is that of the MFDFA singularity spectrum (:func:`.fractal_dfa`), whereas Bell
        et al. (2019) use the Chhabra & Jensen (1989) direct method (keeping only the *q* whose
        fits have :math:`r \\geq 0.995`) and 30 surrogates. The values are therefore not directly
        comparable with theirs. We might add that method in the future.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    n : int
        Number of surrogates. The literature uses values between 30 and 40.
    show : bool
        If ``True``, plots the distribution of the surrogates' widths and the original width.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator used to create the surrogates. See for
        ``misc.check_random_state`` for further information.
    **kwargs : optional
        Other arguments to be passed to :func:`.fractal_dfa` (``multifractal=True`` is always used).

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
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    info = {}
    rng = check_random_state(random_state)
    kwargs = {k: v for k, v in kwargs.items() if k not in ["multifractal", "show"]}

    w0 = float(fractal_dfa(signal, multifractal=True, show=False, **kwargs)[0]["Width"].iloc[0])

    w = np.zeros(n)
    for i in range(n):
        surro = signal_surrogate(signal, method="IAAFT", random_state=rng)
        w[i] = float(fractal_dfa(surro, multifractal=True, show=False, **kwargs)[0]["Width"].iloc[0])

    # One-sample t-test of the original width against the surrogates' widths. ttest_1samp(w, w0)
    # returns (mean(w) - w0) / SE(w), so the sign is flipped to get (w0 - mean(w)) / SE(w)
    t, p = scipy.stats.ttest_1samp(w, w0)
    t = -float(np.squeeze(t))
    info["p"] = float(np.squeeze(p))
    info["Width"] = w0
    info["Width_Surrogates"] = w

    if show is True:
        pd.Series(w).plot(kind="density", label="Width of surrogates")
        plt.axvline(x=w0, c="red", label="Width of original signal")
        plt.title(f"tMF = {t:.2f}, p = {info['p']:.2f}")
        plt.legend()

    return t, info
