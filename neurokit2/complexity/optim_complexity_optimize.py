import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np

from .entropy_approximate import entropy_approximate
from .optim_complexity_delay import (
    _embedding_delay_metric,
    _embedding_delay_plot,
    _embedding_delay_select,
)
from .optim_complexity_dimension import (
    _embedding_dimension_afn,
    _embedding_dimension_ffn,
    _embedding_dimension_plot,
)
from .optim_complexity_tolerance import _optimize_tolerance_plot


def complexity_optimize(
    signal,
    delay_max=50,
    delay_method="fraser1986",
    dimension_max=10,
    dimension_method="afnn",
    tolerance_method="maxApEn",
    show=False,
    **kwargs
):
    """**Joint-estimation of optimal complexity parameters**

    The selection of the parameters *Dimension* and *Delay* is a challenge. One approach is to
    select them (semi) independently (as dimension selection often requires the delay) from each
    other, using :func:`complexity_delay` and :func:`complexity_dimension`.

    Estimate optimal complexity parameters Dimension (m), Time Delay (tau) and tolerance (r).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay_max : int
        See :func:`complexity_delay`.
    delay_method : str
        See :func:`complexity_delay`.
    dimension_max : int
        See :func:`complexity_dimension`.
    dimension_method : str
        See :func:`complexity_dimension`.
    tolerance_method : str
        See :func:`complexity_tolerance`.
    show : bool
        Defaults to ``False``.

    Returns
    -------
    optimal_dimension : int
        Optimal dimension.
    optimal_delay : int
        Optimal time delay.
    optimal_tolerance : int
        Optimal tolerance

    See Also
    ------------
    complexity_delay, complexity_dimension, complexity_tolerance

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, frequency=[5, 7], noise=0.01)
      parameters = nk.complexity_optimize(signal, show=True)
      parameters

    References
    -----------
    * Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based
      method for determining the optimal embedding parameters of a signal. In 2003 IEEE
      International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.
      (ICASSP'03). (Vol. 6, pp. VI-29). IEEE.
    * Camplani, M., & Cannas, B. (2009). The role of the embedding dimension and time delay in time
      series forecasting. IFAC Proceedings Volumes, 42(7), 316-320.
    * Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1994). Reconstruction expansion as a
      geometry-based framework for choosing proper delay times. Physica-Section D, 73(1), 82-98.
    * Cao, L. (1997). Practical method for determining the minimum embedding dimension of a scalar
      time series. Physica D: Nonlinear Phenomena, 110(1-2), 43-50.
    * Lu, S., Chen, X., Kanters, J. K., Solomon, I. C., & Chon, K. H. (2008). Automatic selection of
      the threshold value r for approximate entropy. IEEE Transactions on Biomedical Engineering,
      55(8), 1966-1972.

    """

    out = {}

    # Optimize delay
    tau_sequence, metric, metric_values, out["Delay"] = _complexity_delay(
        signal, delay_max=delay_max, method=delay_method
    )

    # Optimize dimension
    dimension_seq, optimize_indices, out["Dimension"] = _complexity_dimension(
        signal, delay=out["Delay"], dimension_max=dimension_max, method=dimension_method, **kwargs
    )

    # Optimize r
    tolerance_method = tolerance_method.lower()
    if tolerance_method in ["traditional"]:
        out["Tolerance"] = 0.2 * np.std(signal, ddof=1)
    if tolerance_method in ["maxapen", "optimize"]:
        r_range, ApEn, out["Tolerance"] = _complexity_tolerance(
            signal, delay=out["Delay"], dimension=out["Dimension"]
        )

    if show is True:
        if tolerance_method in ["traditional"]:
            raise ValueError(
                "NeuroKit error: complexity_optimize():"
                "show is not available for current tolerance_method"
            )
        if tolerance_method in ["maxapen", "optimize"]:
            _complexity_plot(
                signal,
                out,
                tau_sequence,
                metric,
                metric_values,
                dimension_seq[:-1],
                optimize_indices,
                r_range,
                ApEn,
                dimension_method=dimension_method,
            )

    return out


# =============================================================================
# Plot
# =============================================================================


def _complexity_plot(
    signal,
    out,
    tau_sequence,
    metric,
    metric_values,
    dimension_seq,
    optimize_indices,
    r_range,
    ApEn,
    dimension_method="afnn",
):

    # Prepare figure
    fig = plt.figure(constrained_layout=False)
    spec = matplotlib.gridspec.GridSpec(
        ncols=2, nrows=3, height_ratios=[1, 1, 1], width_ratios=[1 - 1.2 / np.pi, 1.2 / np.pi]
    )

    ax_tau = fig.add_subplot(spec[0, :-1])
    ax_dim = fig.add_subplot(spec[1, :-1])
    ax_r = fig.add_subplot(spec[2, :-1])

    if out["Dimension"] > 2:
        plot_type = "3D"
        ax_attractor = fig.add_subplot(spec[:, -1], projection="3d")
    else:
        plot_type = "2D"
        ax_attractor = fig.add_subplot(spec[:, -1])

    fig.suptitle("Otimization of Complexity Parameters", fontweight="bold", fontsize=16)
    plt.tight_layout(h_pad=0.4, w_pad=0.05)

    # Plot tau optimization
    # Plot Attractor
    _embedding_delay_plot(
        signal,
        metric_values=metric_values,
        tau_sequence=tau_sequence,
        tau=out["Delay"],
        metric=metric,
        ax0=ax_tau,
        ax1=ax_attractor,
        plot=plot_type,
    )

    # Plot dimension optimization
    if dimension_method.lower() in ["afnn"]:
        _embedding_dimension_plot(
            method=dimension_method,
            dimension_seq=dimension_seq,
            min_dimension=out["Dimension"],
            E1=optimize_indices[0],
            E2=optimize_indices[1],
            ax=ax_dim,
        )
    if dimension_method.lower() in ["fnn"]:
        _embedding_dimension_plot(
            method=dimension_method,
            dimension_seq=dimension_seq,
            min_dimension=out["Dimension"],
            f1=optimize_indices[0],
            f2=optimize_indices[1],
            f3=optimize_indices[2],
            ax=ax_dim,
        )

    # Plot r optimization
    _optimize_tolerance_plot(out["Tolerance"], {"Values": r_range, "Scores": ApEn}, ax=ax_r)

    return fig


# =============================================================================
# Internals
# ==============================================================================


def _complexity_delay(signal, delay_max=100, method="fraser1986"):

    # Initalize vectors
    if isinstance(delay_max, int):
        tau_sequence = np.arange(1, delay_max)
    else:
        tau_sequence = np.array(delay_max)

    # Get metric
    # Method
    method = method.lower()
    if method in ["fraser", "fraser1986", "tdmi"]:
        metric = "Mutual Information"
        algorithm = "first local minimum"
    elif method in ["theiler", "theiler1990"]:
        metric = "Autocorrelation"
        algorithm = "first 1/e crossing"
    elif method in ["casdagli", "casdagli1991"]:
        metric = "Autocorrelation"
        algorithm = "first zero crossing"
    elif method in ["rosenstein", "rosenstein1993", "adfd"]:
        metric = "Displacement"
        algorithm = "closest to 40% of the slope"
    else:
        raise ValueError("NeuroKit error: complexity_delay(): 'method' not recognized.")
    metric_values = _embedding_delay_metric(signal, tau_sequence, metric=metric)
    # Get optimal tau
    optimal = _embedding_delay_select(metric_values, algorithm=algorithm)
    if ~np.isnan(optimal):
        tau = tau_sequence[optimal]
    else:
        raise ValueError(
            "NeuroKit error: No optimal time delay is found."
            " Consider using a higher `delay_max`."
        )

    return tau_sequence, metric, metric_values, tau


def _complexity_dimension(
    signal, delay=1, dimension_max=20, method="afnn", R=10.0, A=2.0, **kwargs
):

    # Initalize vectors
    if isinstance(dimension_max, int):
        dimension_seq = np.arange(1, dimension_max + 1)
    else:
        dimension_seq = np.array(dimension_max)

    # Method
    method = method.lower()
    if method in ["afnn"]:
        E, Es = _embedding_dimension_afn(
            signal, dimension_seq=dimension_seq, delay=delay, show=False, **kwargs
        )
        E1 = E[1:] / E[:-1]
        E2 = Es[1:] / Es[:-1]
        min_dimension = [i for i, x in enumerate(E1 >= 0.85 * np.max(E1)) if x][0] + 1
        optimize_indices = [E1, E2]
        return dimension_seq, optimize_indices, min_dimension

    if method in ["fnn"]:
        f1, f2, f3 = _embedding_dimension_ffn(
            signal, dimension_seq=dimension_seq, delay=delay, R=R, A=A, **kwargs
        )
        min_dimension = [i for i, x in enumerate(f3 <= 1.85 * np.min(f3[np.nonzero(f3)])) if x][0]
        optimize_indices = [f1, f2, f3]
        return dimension_seq, optimize_indices, min_dimension
    else:
        raise ValueError("NeuroKit error: complexity_dimension(): 'method' not recognized.")


def _complexity_tolerance(signal, delay=None, dimension=None):

    modulator = np.arange(0.02, 0.8, 0.02)
    r_range = modulator * np.std(signal, ddof=1)
    ApEn = np.zeros_like(r_range)
    for i, r in enumerate(r_range):
        ApEn[i] = entropy_approximate(
            signal, delay=delay, dimension=dimension, tolerance=r_range[i]
        )[0]
    r = r_range[np.argmax(ApEn)]

    return r_range, ApEn, r
