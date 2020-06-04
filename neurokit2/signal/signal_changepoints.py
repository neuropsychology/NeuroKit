import numpy as np

from ..events import events_plot
from ..misc import as_vector


def signal_changepoints(signal, change="meanvar", penalty=None, show=False):
    """
    Change Point Detection.

    Only the PELT method is implemented for now.

    Parameters
    -----------
    signal : list, array or Series
        Vector of values.
    signal : list, array or Series
        Vector of values.
    penalty : float
        The algorithm penalty. Default to ``np.log(len(signal))``.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.emg_simulate(burst_number=3)
    >>> fig = nk.signal_changepoints(signal, change="var", show=True)
    >>> fig #doctest: +SKIP

    References
    ----------
    - Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association, 107(500), 1590-1598.

    """
    signal = as_vector(signal)
    changepoints = _signal_changepoints_pelt(signal, change=change, penalty=penalty)

    if show is True:
        events_plot(changepoints, signal)

    return changepoints


def _signal_changepoints_pelt(signal, change="meanvar", penalty=None):
    """
    PELT algorithm to find change points in a signal.

    Adapted from: https://github.com/ruipgil/changepy https://github.com/deepcharles/ruptures
    https://github.com/STOR-i/Changepoints.jl https://github.com/rkillick/changepoint/

    """
    # Initialize
    length = len(signal)
    if penalty is None:
        penalty = np.log(length)
    if change.lower() == "var":
        cost = _signal_changepoints_cost_var(signal)
    elif change.lower() == "mean":
        cost = _signal_changepoints_cost_mean(signal)
    else:
        cost = _signal_changepoints_cost_meanvar(signal)

    # Run algorithm
    F = np.zeros(length + 1)
    R = np.array([0], dtype=np.int)
    candidates = np.zeros(length + 1, dtype=np.int)

    F[0] = -penalty

    for tstar in range(2, length + 1):
        cpt_cands = R
        seg_costs = np.zeros(len(cpt_cands))
        for i in range(0, len(cpt_cands)):
            seg_costs[i] = cost(cpt_cands[i], tstar)

        F_cost = F[cpt_cands] + seg_costs
        F[tstar] = np.min(F_cost) + penalty
        tau = np.argmin(F_cost)
        candidates[tstar] = cpt_cands[tau]

        ineq_prune = [val < F[tstar] for val in F_cost]
        R = [cpt_cands[j] for j, val in enumerate(ineq_prune) if val]
        R.append(tstar - 1)
        R = np.array(R, dtype=np.int)

    last = candidates[-1]
    changepoints = [last]
    while last > 0:
        last = candidates[last]
        changepoints.append(last)

    return np.sort(changepoints)


# =============================================================================
# Cost functions
# =============================================================================
def _signal_changepoints_cost_mean(signal):
    """
    Cost function for a normally distributed signal with a changing mean.
    """
    i_variance_2 = 1 / (np.var(signal) ** 2)
    cmm = [0.0]
    cmm.extend(np.cumsum(signal))

    cmm2 = [0.0]
    cmm2.extend(np.cumsum(np.abs(signal)))

    def cost(start, end):
        cmm2_diff = cmm2[end] - cmm2[start]
        cmm_diff = pow(cmm[end] - cmm[start], 2)
        i_diff = end - start
        diff = cmm2_diff - cmm_diff
        return (diff / i_diff) * i_variance_2

    return cost


def _signal_changepoints_cost_var(signal):
    """
    Cost function for a normally distributed signal with a changing variance.
    """
    cumm = [0.0]
    cumm.extend(np.cumsum(np.power(np.abs(signal - np.mean(signal)), 2)))

    def cost(s, t):
        dist = float(t - s)
        diff = cumm[t] - cumm[s]
        return dist * np.log(diff / dist)

    return cost


def _signal_changepoints_cost_meanvar(signal):
    """
    Cost function for a normally distributed signal with a changing mean and variance.
    """
    signal = np.hstack(([0.0], np.array(signal)))

    cumm = np.cumsum(signal)
    cumm_sq = np.cumsum([val ** 2 for val in signal])

    def cost(s, t):
        ts_i = 1.0 / (t - s)
        mu = (cumm[t] - cumm[s]) * ts_i
        sig = (cumm_sq[t] - cumm_sq[s]) * ts_i - mu ** 2
        sig_i = 1.0 / sig
        return (
            (t - s) * np.log(sig)
            + (cumm_sq[t] - cumm_sq[s]) * sig_i
            - 2 * (cumm[t] - cumm[s]) * mu * sig_i
            + ((t - s) * mu ** 2) * sig_i
        )

    return cost
