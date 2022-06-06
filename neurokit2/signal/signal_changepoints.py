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
        The algorithm penalty. Defaults to ``np.log(len(signal))``.
    show : bool
        Defaults to ``False``.

    Returns
    -------
    Array
        Values indicating the samples at which the changepoints occur.
    Fig
        Figure of plot of signal with markers of changepoints.

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
    signal = as_vector(signal)
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
    if penalty is None:
        penalty = np.log(length)  # pylint: disable=E1111
    if change.lower() == "var":
        cost = _signal_changepoints_cost_var(signal)
    elif change.lower() == "mean":
        cost = _signal_changepoints_cost_mean(signal)
    else:
        cost = _signal_changepoints_cost_meanvar(signal)

    # Run algorithm
    F = np.zeros(length + 1)
    R = np.array([0], dtype=int)
    candidates = np.zeros(length + 1, dtype=int)

    F[0] = -penalty  # pylint: disable=E1130

    for tstar in range(2, length + 1):
        cpt_cands = R
        seg_costs = np.array([cost(cpt_cands[i], tstar) for i in range(len(cpt_cands))])

        F_cost = F[cpt_cands] + seg_costs
        F[tstar] = np.nanmin(F_cost) + penalty
        tau = np.nanargmin(F_cost)
        candidates[tstar] = cpt_cands[tau]

        ineq_prune = [val < F[tstar] for val in F_cost]
        R = [cpt_cands[j] for j, val in enumerate(ineq_prune) if val]
        R.append(tstar - 1)
        R = np.array(R, dtype=int)

    changepoints = np.sort(np.unique(candidates[candidates]))
    changepoints = changepoints[changepoints > 0]
    return changepoints


# =============================================================================
# Cost functions
# =============================================================================
def _signal_changepoints_cost_mean(signal):
    """Cost function for a normally distributed signal with a changing mean."""
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
    """Cost function for a normally distributed signal with a changing variance."""
    cumm = [0.0]
    cumm.extend(np.cumsum(np.power(np.abs(signal - np.mean(signal)), 2)))

    def cost(s, t):
        dist = float(t - s)
        diff = cumm[t] - cumm[s]
        return dist * np.log(diff / dist)

    return cost


def _signal_changepoints_cost_meanvar(signal):
    """Cost function for a normally distributed signal with a changing mean and variance."""
    signal = np.hstack(([0.0], np.array(signal)))

    cumm = np.cumsum(signal)
    cumm_sq = np.cumsum([val ** 2 for val in signal])

    def cost(s, t):
        ts_i = 1.0 / (t - s)
        mu = (cumm[t] - cumm[s]) * ts_i
        sig = (cumm_sq[t] - cumm_sq[s]) * ts_i - mu ** 2
        sig_i = 1.0 / sig

        if sig <= 0:
            return np.nan
        else:
            return (
                (t - s) * np.log(sig)
                + (cumm_sq[t] - cumm_sq[s]) * sig_i
                - 2 * (cumm[t] - cumm[s]) * mu * sig_i
                + ((t - s) * mu ** 2) * sig_i
            )

    return cost
