import numpy as np

from ..events import events_plot
from ..misc import as_vector
from .signal_plot import signal_plot


def signal_changepoints(signal, change="meanvar", penalty=None, show=False):
    """**Change Point Detection**

    Only the PELT method is implemented for now.

    Parameters
    -----------
    signal : Union[list, np.array, pd.Series]
        Vector of values.
    change : str
        Can be one of ``"meanvar"`` (default), ``"mean"`` or ``"var"``.
    penalty : float
        The algorithm penalty. Defaults to the Schwarz Information Criterion: ``np.log(len(signal))``
        for ``"mean"``, twice that for ``"var"`` and three times that for ``"meanvar"``.
    show : bool
        If ``True``, plots the signal with markers of changepoints. Defaults to ``False``.

    Returns
    -------
    Array
        Values indicating the samples at which the changepoints occur.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.emg_simulate(burst_number=3)
      @savefig p_signal_changepoints1.png scale=100%
      nk.signal_changepoints(signal, change="var", show=True)
      @suppress
      plt.close()


    References
    ----------
    * Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a
      linear computational cost. Journal of the American Statistical Association, 107(500), 1590-1598.

    """
    signal = as_vector(signal).astype(float)
    changepoints = _signal_changepoints_pelt(signal, change=change, penalty=penalty)

    if show is True:
        if len(changepoints) > 0:
            events_plot(changepoints, signal)
        else:
            signal_plot(signal)

    return changepoints


def _signal_changepoints_pelt(signal, change="meanvar", penalty=None):
    """PELT algorithm to find change points in a signal.

    Adapted from: https://github.com/ruipgil/changepy https://github.com/deepcharles/ruptures
    https://github.com/STOR-i/Changepoints.jl https://github.com/rkillick/changepoint/

    """
    # Initialize
    length = len(signal)
    if len(np.unique(signal)) <= 1:
        return np.array([], dtype=int)
    if penalty is None:
        penalty = {"mean": 1, "var": 2}.get(change.lower(), 3) * np.log(length)
    if change.lower() == "var":
        cost = _signal_changepoints_cost_var(signal)
    elif change.lower() == "mean":
        cost = _signal_changepoints_cost_mean(signal)
    else:
        cost = _signal_changepoints_cost_meanvar(signal)

    # Run algorithm
    F = np.zeros(length + 1)
    R = np.array([0], dtype=int)
    pruned = np.array([], dtype=int)
    candidates = np.zeros(length + 1, dtype=int)

    F[0] = -penalty  # pylint: disable=E1130
    F[1] = np.inf

    for tstar in range(2, length + 1):
        cpt_cands = R
        seg_costs = np.array([cost(cpt_cands[i], tstar) for i in range(len(cpt_cands))])

        F_cost = F[cpt_cands] + seg_costs
        F[tstar] = np.nanmin(F_cost) + penalty
        tau = np.nanargmin(F_cost)
        candidates[tstar] = cpt_cands[tau]

        R = np.append(cpt_cands[~np.isin(cpt_cands, pruned)], tstar - 1)
        pruned = cpt_cands[F_cost >= F[tstar]]

    changepoints = []
    last = candidates[-1]
    while last > 0:
        changepoints.append(last)
        last = candidates[last]
    return np.array(changepoints[::-1], dtype=int)


# =============================================================================
# Cost functions
# =============================================================================
def _signal_changepoints_cost_mean(signal):
    """Cost function for a normally distributed signal with a changing mean."""
    i_variance = 1 / (2 * np.var(signal))
    cmm = [0.0]
    cmm.extend(np.cumsum(signal))

    cmm2 = [0.0]
    cmm2.extend(np.cumsum(np.square(signal)))

    def cost(start, end):
        cmm2_diff = cmm2[end] - cmm2[start]
        cmm_diff = pow(cmm[end] - cmm[start], 2)
        i_diff = end - start
        diff = cmm2_diff - cmm_diff / i_diff
        return diff * i_variance

    return cost


def _signal_changepoints_cost_var(signal):
    """Cost function for a normally distributed signal with a changing variance."""
    floor = 1e-10 * np.var(signal)
    cumm = [0.0]
    cumm.extend(np.cumsum(np.power(np.abs(signal - np.mean(signal)), 2)))

    def cost(s, t):
        dist = float(t - s)
        diff = cumm[t] - cumm[s]
        return dist * np.log(max(diff / dist, floor))

    return cost


def _signal_changepoints_cost_meanvar(signal):
    """Cost function for a normally distributed signal with a changing mean and variance."""
    floor = 1e-10 * np.var(signal)
    signal = np.hstack(([0.0], np.array(signal)))

    cumm = np.cumsum(signal)
    cumm_sq = np.cumsum([val**2 for val in signal])

    def cost(s, t):
        ts_i = 1.0 / (t - s)
        mu = (cumm[t] - cumm[s]) * ts_i
        sig = max((cumm_sq[t] - cumm_sq[s]) * ts_i - mu**2, floor)
        return (t - s) * (np.log(sig) + 1)

    return cost
