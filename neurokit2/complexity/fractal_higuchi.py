# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def fractal_higuchi(signal, kmax=5, show=True):
    """
    Computes Higuchi's Fractal Dimension (HFD).
    Value should fall between 1 and 2. For more information about k parameter selection, see
    the papers referenced below.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    kmax : int
        Maximum number of interval times (should be greater than or equal to 2).
    show : bool
        Visualise plot.

    Returns
    ----------
    slope
        Higuchi's fractal dimension.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5, noise=10)
    >>>
    >>> hfd = nk.fractal_higuchi(signal, kmax=5)
    >>> hfd #doctest: +SKIP

    References
    ----------
    - Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
    Physica D: Nonlinear Phenomena, 31(2), 277-283.

    - Vega, C. F., & Noel, J. (2015, June). Parameters analyzed of Higuchi's fractal dimension for EEG brain signals.
    In 2015 Signal Processing Symposium (SPSympo) (pp. 1-5). IEEE. https://ieeexplore.ieee.org/document/7168285
    """
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
    slope, _ = - np.polyfit(np.log(k_values), np.log(average_values), 1)

    if show:
        _fractal_higuchi_plot(k_values, average_values, kmax, slope)
        
    return slope


def _fractal_higuchi_plot(k_values, average_values, kmax, slope):

    kmax = str(kmax)
    slope = str(np.round(slope, 2))
    fig, ax = plt.subplots()
    ax.set_title("Least-squares linear best-fit curve for $k_{max}$ = " + kmax + ", slope = " + slope)
    ax.set_ylabel(r"$ln$(L(k))")
    ax.set_xlabel(r"$ln$(1/k)")
    colors = plt.cm.plasma(np.linspace(0, 1, len(k_values)))

    for i in range(0, len(k_values)):
        ax.scatter(-np.log(k_values[i]), np.log(average_values[i]), color=colors[i],
               marker='o', zorder=2, label="k = {}".format(i))
    ax.plot(-np.log(k_values), np.log(average_values), color="#FF9800", zorder=1)
    ax.legend(loc="lower right")

    return fig

# def _fractal_higuchi_optimal_k(k_first=2, k_end=60):
#     """
#     Optimize the kmax parameter.
    
#     HFD values are plotted against a range of kmax and the point at which the values plateau is
#     considered the saturation point and subsequently selected as the kmax value.
#     """

#     k_range = np.arange(k_first, k_end + 1)
#     slope_values = []
#     for i in k_range:
#         slope = nk.fractal_higuchi(signal, kmax=i)
#         slope_values.append(slope)

#     # Obtain saturation point of slope
    

#     # Plot
#     fig, ax = plt.subplots()
#     ax.set_title("Optimization of $k_max$ parameter")
#     ax.set_xlabel("$k_max$ values")
#     ax.set_ylabel("Higuchi Fractal Dimension (HFD) values")

#     ax.plot(k_range, slope_values, color="#FF5722")
#     ax.legend(loc="upper right")

