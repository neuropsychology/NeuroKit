# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from .entropy_approximate import entropy_approximate
from .optim_complexity_delay import complexity_delay
from .optim_complexity_dimension import complexity_dimension


def complexity_r(signal, method="maxApEn", delay=None, dimension=None, show=False):
    """Automated selection of the optimal tolerance (r) parameter for entropy measures

    The r parameter has a critical impact and is a major source of inconsistencies in the literature.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        If 'maxApEn', rmax where ApEn is max will be returned. If 'sd' (as in Standard Deviation),
        r = 0.2 * standard deviation of the signal will be returned.
    delay : int
        Only used if ``method='maxApEn'``. Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common to
        have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or to
        find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Only used if ``method='maxApEn'``. Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    show : bool
        If true and method is 'maxApEn', will plot the ApEn values for each value of r.

    Returns
    ----------
    float
        The optimal r as a similarity threshold. It corresponds to the filtering level - max absolute
        difference between segments.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> delay, _ = nk.complexity_delay(signal)
    >>> dimension, _ = nk.complexity_dimension(signal, delay=delay)
    >>> r, info = nk.complexity_r(signal, delay, dimension)
    >>> r #doctest: +SKIP


    References
    -----------
    - Lu, S., Chen, X., Kanters, J. K., Solomon, I. C., & Chon, K. H. (2008). Automatic selection of
      the threshold value r for approximate entropy. IEEE Transactions on Biomedical Engineering,
      55(8), 1966-1972.
    """
    # Method
    method = method.lower()
    if method in ["traditional", "sd", "std"]:
        r = 0.2 * np.std(signal, ddof=1)
        info = {"Method": method}
    elif method in ["maxapen", "optimize"]:
        r, info = _optimize_r(signal, delay=delay, dimension=dimension, show=show)
        info.update({"Method": method})
    return r, info


# =============================================================================
# Internals
# =============================================================================
def _optimize_r(signal, delay=None, dimension=None, show=False):

    if not delay:
        delay = complexity_delay(signal, delay_max=100, method="fraser1986")
    if not dimension:
        dimension = complexity_dimension(signal, delay=delay, dimension_max=20, show=True)

    modulator = np.arange(0.02, 0.8, 0.02)
    r_range = modulator * np.std(signal, ddof=1)

    ApEn = np.zeros_like(r_range)
    for i, r in enumerate(r_range):
        ApEn[i] = entropy_approximate(signal, delay=delay, dimension=dimension, r=r_range[i])[0]

    r = r_range[np.argmax(ApEn)]

    if show is True:
        _optimize_r_plot(r, r_range, ApEn, ax=None)

    return r, {"Values": r_range, "Scores": ApEn}


def _optimize_r_plot(r, r_range, ApEn, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.set_title("Optimization of Tolerence Threshold (r)")
    ax.set_xlabel("Tolerence threshold $r$")
    ax.set_ylabel("Approximate Entropy $ApEn$")
    ax.plot(r_range, ApEn, "bo-", label="$ApEn$", color="#80059c")
    ax.axvline(x=r, color="#E91E63", label="Optimal r: " + str(np.round(r, 3)))
    ax.legend(loc="upper right")

    return fig
