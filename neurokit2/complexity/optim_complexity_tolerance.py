# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from .entropy_approximate import entropy_approximate
from .optim_complexity_delay import complexity_delay
from .optim_complexity_dimension import complexity_dimension


def complexity_tolerance(
    signal, method="maxApEn", r_range=None, delay=None, dimension=None, show=False
):
    """Automated selection of the optimal tolerance (r) parameter for entropy measures

    The tolerance r is essentially a threshold value by which to consider two points as similar. This parameter has a critical impact and is a major source of inconsistencies in the literature.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        If 'maxApEn', different values of tolerance will be tested and the one where ApEn is
        maximized will be selected and returned. If 'sd' (as in Standard Deviation),
        r = 0.2 * standard deviation of the signal will be returned.
    r_range : Union[list, int]
        Only used if ``method='maxApEn'``. The range of tolerance values to test. If an integer, will be set to ``np.linspace(0.02, 0.8, r_range) * np.std(signal, ddof=1)``. If ``None``, will be set to ``40``. You can set a lower number for faster results.
    delay : int
        Only used if ``method='maxApEn'``. See ``entropy_approximate()``.
    dimension : int
        Only used if ``method='maxApEn'``. See ``entropy_approximate()``.
    show : bool
        If true and method is 'maxApEn', will plot the ApEn values for each value of r.

    See Also
    --------
    entropy_approximate, complexity_delay, complexity_dimension

    Returns
    ----------
    float
        The optimal tolerance value.
    dict
        A dictionary with the values of r and the corresponding ApEn values (when method='maxApEn').

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> # Fast
    >>> r, info = nk.complexity_tolerance(signal, method = 'SD')
    >>> r
    0.07072836242007384
    >>>
    >>> # Slow
    >>> r, info = nk.complexity_tolerance(signal, delay=8, dimension=6, method = 'maxApEn', show=True)
    >>> r #doctest: +SKIP
    0.014145672484014769
    >>>
    >>> # Narrower range
    >>> r, info = nk.complexity_tolerance(signal, delay=8, dimension=6, method = 'maxApEn', r_range=np.linspace(0.002, 0.1, 30), show=True)
    >>>


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
        r, info = _optimize_tolerance_maxapen(
            signal, r_range=r_range, delay=delay, dimension=dimension, show=show
        )
        if show is True:
            _optimize_tolerance_plot(r, info["Values"], info["Scores"])
        info.update({"Method": method})
    return r, info


# =============================================================================
# Internals
# =============================================================================
def _optimize_tolerance_maxapen(signal, r_range=None, delay=None, dimension=None, show=False):

    # Optimize missing parameters
    if delay is None or dimension is None:
        raise ValueError("If method='maxApEn', both delay and dimension must be specified.")

    if r_range is None:
        r_range = 40
    if isinstance(r_range, int):
        r_range = np.linspace(0.02, 0.8, r_range) * np.std(signal, ddof=1)

    ApEn = np.zeros_like(r_range)
    for i, r in enumerate(r_range):
        ApEn[i], _ = entropy_approximate(
            signal, delay=delay, dimension=dimension, tolerance=r_range[i]
        )

    return r_range[np.argmax(ApEn)], {"Values": r_range, "Scores": ApEn}


def _optimize_tolerance_plot(r, r_range, ApEn, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.set_title("Optimization of Tolerence Threshold (r)")
    ax.set_xlabel("Tolerence threshold $r$")
    ax.set_ylabel("Approximate Entropy $ApEn$")
    ax.plot(r_range, ApEn, "o-", label="$ApEn$", color="#80059c")
    ax.axvline(x=r, color="#E91E63", label="Optimal r: " + str(np.round(r, 3)))
    ax.legend(loc="upper right")

    return fig
