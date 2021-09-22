# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..misc import NeuroKitWarning
from .utils import _sanitize_multichannel


def fractal_higuchi(signal, kmax="default", show=False):
    """
    Computes Higuchi's Fractal Dimension (HFD) by reconstructing k-max number of new
    data sets. For each reconstructed data set, curve length is computed and plotted
    against its corresponding k value on a log-log scale. HFD equates to the slope obtained
    from fitting a least-squares method.

    Values should fall between 1 and 2. For more information about k parameter selection, see
    the papers referenced below.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series, np.ndarray, pd.DataFrame]
        The signal (i.e., a time series) in the form of a vector of values or in
        the form of an n-dimensional array (with a shape of len(channels) x len(samples))
        or dataframe.
    kmax : str or int
        Maximum number of interval times (should be greater than or equal to 2).
        If "default", then the optimal kmax is computed based on the point at which HFD values plateau
        for a range of kmax values.
    show : bool
        Visualise the slope of the curve for the selected kmax value.

    Returns
    ----------
    slope : float
        Higuchi's fractal dimension of the single time series or the mean HFD
        across the channels of an n-dimensional time series.
    parameters : dict
        A dictionary containing additional information regarding the parameters used
        to compute Higuchi's fractal dimension and the individual HFD values of each
        channel if an n-dimensional time series is passed.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.data('bio_eventrelated_100hz')['ECG']
    >>>
    >>> hfd, parameters = nk.fractal_higuchi(signal, kmax=5, show=True)
    >>> hfd #doctest: +SKIP
    >>> hfd, parameters = nk.fractal_higuchi(signal, kmax="default", show=True)
    >>> hfd #doctest: +SKIP

    Reference
    ----------
    - Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
    Physica D: Nonlinear Phenomena, 31(2), 277-283.

    - Vega, C. F., & Noel, J. (2015, June). Parameters analyzed of Higuchi's fractal dimension for EEG brain signals.
    In 2015 Signal Processing Symposium (SPSympo) (pp. 1-5). IEEE. https://ieeexplore.ieee.org/document/7168285
    """

    # Sanitize maximum value of k
    k_end = 60
    k_end = _cleanse_k(signal, kmax, k_end)

    # Obtain optimal k if default
    if kmax == "default":
        if signal.ndim > 1:
            # n-dimensional
            signal = _sanitize_multichannel(signal)
            k_max = []
            slope_values = []
            for index, colname in enumerate(signal):
                k, k_range, slope = _fractal_higuchi_optimal_k(signal[colname], k_start=2, k_end=k_end)
                k_max.append(k)
                slope_values.append(slope)
        else:
            # one single time series
            k_max, k_range, slope_values = _fractal_higuchi_optimal_k(signal, k_start=2, k_end=k_end)
    else:
        # no optimizing
        k_max = k_end

    # Make sure kmax >= 2
    if np.mean(k_max) <= 2:
        raise ValueError(
            "NeuroKit error: fractal_higuchi(): The optimal `kmax` detected as less than or equal to 2. "
            "Please manually input a `kmax` value of more than 2."
        )

    # Compute slope
    if signal.ndim > 1 and kmax == 'default':
        # pass average value of kmax if optimized in n-dimensional series
        slope, intercept, k_values, average_values = _fractal_higuchi_slope(signal, int(np.mean(k_max)))
    else:
        slope, intercept, k_values, average_values = _fractal_higuchi_slope(signal, k_max)

    # Plot
    if show:
        # show only slope plot if kmax not automated
        if kmax != "default":
            _fractal_higuchi_plot(k_values, average_values, k_max, slope, intercept, ax=None)
        # show both slope plot and kmax optimizing plot
        else:
            fig = plt.figure(constrained_layout=False)
            fig.suptitle('Higuchi Fractal Dimension (HFD)')
            spec = matplotlib.gridspec.GridSpec(
                    ncols=1, nrows=2
                )
            ax_slope = fig.add_subplot(spec[0, 0])
            _fractal_higuchi_plot(k_values, average_values, k_max, slope, intercept, ax=ax_slope)
            ax_kmax = fig.add_subplot(spec[1, 0])
            _fractal_higuchi_optimal_k_plot(k_range, slope_values, k_max, signal, ax=ax_kmax)

    parameters = {'kmax': k_max}

    return slope, parameters


# =============================================================================
# Utilities
# =============================================================================


def _fractal_higuchi_slope(signal, kmax):

    if signal.ndim > 1:
        N = signal.shape[0]
    else:
        N = signal.size

    average_values = []
    # Compute length of the curve, Lm(k)
    for k in range(1, kmax + 1):
        sets = []
        for m in range(1, k + 1):
            n_max = int(np.floor((N - m) / k))
            normalization = (N - 1) / (n_max * k)
            Lm_k = np.sum(np.abs(np.diff(signal[m-1::k], n=1))) * normalization
            Lm_k = Lm_k / k
            sets.append(Lm_k)
        # Compute average value over k sets of Lm(k)
        L_k = np.sum(sets) / k
        average_values.append(L_k)

    # Slope of best-fit line through points
    k_values = np.arange(1, kmax + 1)
    slope, intercept = - np.polyfit(np.log(k_values), np.log(average_values), 1)

    return slope, intercept, k_values, average_values


def _fractal_higuchi_plot(k_values, average_values, kmax, slope, intercept, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle('Higuchi Fractal Dimension (HFD)')
    else:
        fig = None

    # if multiple optimal k values passed (from n-dimensional time series), get mean optimal k
    if isinstance(kmax, list):
        kmax = int(np.mean(kmax))
    kmax_val = str(kmax)
    slope_val = str(np.round(slope, 2))
    ax.set_title("Least-squares linear best-fit curve for $k_{max}$ = " + kmax_val +
                 ", slope = " + slope_val)
    ax.set_ylabel(r"$ln$(L(k))")
    ax.set_xlabel(r"$ln$(1/k)")
    colors = plt.cm.plasma(np.linspace(0, 1, len(k_values)))

    # Label all values unless len(k_values) > 10 then label only min and max k_max
    if len(k_values) < 10:
        for i in range(0, len(k_values)):
            ax.scatter(-np.log(k_values[i]), np.log(average_values[i]), color=colors[i],
                   marker='o', zorder=2, label="k = {}".format(i+1))
    else:
        for i in range(0, len(k_values)):
            ax.scatter(-np.log(k_values[i]), np.log(average_values[i]), color=colors[i],
                   marker='o', zorder=2, label="_no_legend_")
        ax.plot([], label="k = {}".format(np.min(k_values)),
                c=colors[0])
        ax.plot([], label="k = {}".format(np.max(k_values)),
                c=colors[-1])

    fit_values = [slope * i + -intercept for i in -np.log(k_values)]
    ax.plot(-np.log(k_values), fit_values, color="#FF9800", zorder=1)
    ax.legend(loc="lower right")

    return fig


# =============================================================================
# Compute kmax
# =============================================================================
def _cleanse_k(signal, kmax, k_end):

    N = len(signal)
    # upper limit for k value
    max_k = int(np.floor(N / 2))  # so that normalizing factor is positive

    if isinstance(kmax, int):
        k_end = kmax

    if max_k < k_end:
        return max_k
    else:
        return k_end


def _fractal_higuchi_optimal_k(signal, k_start=2, k_end=60):
    """
    Optimize the kmax parameter.

    HFD values are plotted against a range of kmax and the point at which the values plateau is
    considered the saturation point and subsequently selected as the kmax value.
    """
    k_range = np.arange(k_start, k_end + 1)
    slope_values = []

    if signal.ndim > 1:
        # n-dimensional
        kmax = []
        for index, colname in enumerate(signal):
            # initiate slope values to be stored per signal
            slope_values_channel = []
            for _, k in enumerate(k_range):
                slope, _, _, _ = _fractal_higuchi_slope(signal[colname], kmax=k)
                slope_values_channel.append(slope)

            slope_values.append(slope_values_channel)

            # Obtain saturation point of slope
            # first step: set optimal where slope is max
            optimal_k = np.argmax(slope_values)
            if optimal_k > 0.7 * len(k_range):
                # second step: set optimal where slope is approaching the max
                optimal_k = [i for i, x in enumerate(slope_values >= 0.95 * np.max(slope_values)) if x][0]
            kmax.append(k_range[optimal_k])

    else:
        # if one single time series
        for _, k in enumerate(k_range):
            slope, _, _, _ = _fractal_higuchi_slope(signal, kmax=k)
            slope_values.append(slope)

        # Obtain saturation point of slope
        # first step: set optimal where slope is max
        optimal_k = np.argmax(slope_values)
        if optimal_k > 0.7 * len(k_range):
            # second step: set optimal where slope is approaching the max
            optimal_k = [i for i, x in enumerate(slope_values >= 0.95 * np.max(slope_values)) if x][0]
        kmax = k_range[optimal_k]

        # If no plateau
        if isinstance(kmax, int):
            if kmax <= 2:
                warn(
                    "The optimal kmax value detected is 2 or less. There may be no plateau in this case. "
                    "You can inspect the plot by set `show=True`. HFD is returned for kmax value of 20.",
                    category=NeuroKitWarning
                )
                kmax = 20
        elif isinstance(kmax, list):  # if lists of kmax (from n-dim signal)
            if any([i <= 2 for i in kmax]):
                warn(
                    "The optimal kmax value detected is 2 or less. There may be no plateau in this case. "
                    "You can inspect the plot by set `show=True`. HFD is returned for kmax value of 20.",
                    category=NeuroKitWarning
                )
                below_indices = np.where(np.array(kmax) <= 16)[0]
                for i in below_indices:
                    kmax[i] = 20

    return kmax, k_range, slope_values


def _fractal_higuchi_optimal_k_plot(k_range, slope_values, optimal_k, signal, ax=None):

    # Prepare plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.set_title("Optimization of $k_{max}$ parameter")
    ax.set_xlabel("$k_{max}$ values")
    ax.set_ylabel("Higuchi Fractal Dimension (HFD) values")
    colors = plt.cm.PuBu(np.linspace(0, 1, len(k_range)))

    # Loop if n-dimensional signal
    if signal.ndim > 1:
        color_channels = plt.cm.Purples(np.linspace(0.5, 1, signal.shape[1]))
        for k, slope, index in zip(optimal_k, slope_values, range(0, signal.shape[1])):
            ax.plot(k_range, slope, color=color_channels[index], zorder=1, label=signal.columns[index])
            for i, j in enumerate(k_range):
                ax.scatter(k_range[i], slope[i], color=colors[i],
                           marker='o', zorder=2)
        ax.axvline(x=np.mean(optimal_k).astype(int), color="#E91E63",
                   label="Mean Optimal $k_{max}$: " + str(k))
        ax.legend(loc="upper right")
    else:
        # if single time series
        ax.plot(k_range, slope_values, color="#2196F3", zorder=1)    
        for i, j in enumerate(k_range):
            ax.scatter(k_range[i], slope_values[i], color=colors[i],
                       marker='o', zorder=2)
        ax.axvline(x=optimal_k, color="#E91E63", label="Optimal $k_{max}$: " + str(optimal_k))
        ax.legend(loc="upper right")

    return fig
