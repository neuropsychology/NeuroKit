# -*- coding: utf-8 -*-
import warnings

import numpy as np
import scipy.stats


def density_bandwidth(x, method="KernSmooth", resolution=401):
    """**Bandwidth Selection for Density Estimation**

    Bandwidth selector for :func:`.density` estimation. See ``bw_method`` argument in
    :func:`.scipy.stats.gaussian_kde`.

    The ``"KernSmooth"`` method is adapted from the ``dpik()`` function from the *KernSmooth* R
    package. In this case, it estimates the optimal AMISE bandwidth using the direct plug-in method
    with 2 levels for the Parzen-Rosenblatt estimator with Gaussian kernel.

    Parameters
    -----------
    x : Union[list, np.array, pd.Series]
        A vector of values.
    method : float
        The bandwidth of the kernel. The larger the values, the smoother the estimation. Can be a
        number, or ``"scott"`` or ``"silverman"``
        (see ``bw_method`` argument in :func:`.scipy.stats.gaussian_kde`), or ``"KernSmooth"``.
    resolution : int
        Only when ``method="KernSmooth"``. The number of equally-spaced points over which binning
        is performed to obtain kernel functional approximation (see ``gridsize`` argument in ``KernSmooth::dpik()``).

    Returns
    -------
    float
        Bandwidth value.

    See Also
    --------
    density

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      x = np.random.normal(0, 1, size=100)
      bw = nk.density_bandwidth(x)
      bw

      nk.density_bandwidth(x, method="scott")
      nk.density_bandwidth(x, method=1)

      @savefig p_density_bandwidth.png scale=100%
      x, y = nk.density(signal, bandwidth=bw, show=True)
      @suppress
      plt.close()

    References
    ----------
    * Jones, W. M. (1995). Kernel Smoothing, Chapman & Hall.

    """
    if isinstance(method, str):
        method = method.lower()
    if isinstance(method, (float, int)) or method != "kernsmooth":
        return scipy.stats.gaussian_kde(x, bw_method=method).factor

    n = len(x)

    stdev = np.nanstd(x, ddof=1)
    iqr = np.diff(np.percentile(x, [25, 75]))[0] / 1.349
    scalest = min(stdev, iqr)

    data_scaled = (x - np.nanmean(x)) / scalest
    min_scaled = np.nanmin(data_scaled)
    max_scaled = np.nanmax(data_scaled)

    gcounts = _density_linearbinning(
        x=data_scaled,
        gpoints=np.linspace(min_scaled, max_scaled, resolution),
        truncate=True,
    )

    alpha = (2 * np.sqrt(2) ** 9 / (7 * n)) ** (1 / 9)
    psi6hat = _density_bkfe(gcounts, 6, alpha, min_scaled, max_scaled)

    alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1 / 7)
    psi4hat = _density_bkfe(gcounts, 4, alpha, min_scaled, max_scaled)

    delta_0 = 1 / ((4 * np.pi) ** (1 / 10))
    output = scalest * delta_0 * (1 / (psi4hat * n)) ** (1 / 5)
    return output


def _density_linearbinning(x, gpoints, truncate=True):
    """
    Linear binning. Adapted from KernSmooth R package.
    """
    n = len(x)
    M = gpoints.shape[0]
    a = gpoints[0]
    b = gpoints[-1]

    # initialization of gcounts:
    gcounts = np.zeros(M)
    Delta = (b - a) / (M - 1)

    for i in range(n):
        lxi = ((x[i] - a) / Delta) + 1
        li = int(lxi)

        rem = lxi - li

        if (li >= 1) and (li < M):
            gcounts[li - 1] = gcounts[li - 1] + 1 - rem
            gcounts[li] = gcounts[li] + rem

        elif (li < 1) and (truncate is False):
            gcounts[0] = gcounts[0] + 1

        elif (li >= M) and (truncate is False):
            gcounts[M - 1] = gcounts[M - 1] + 1

    return gcounts


def _density_bkfe(gcounts, drv, h, a, b):
    """
    'bkfe' function adapted from KernSmooth R package.
    """
    resol = len(gcounts)

    # Set the sample size and bin width
    n = np.nansum(gcounts)
    delta = (b - a) / (resol - 1)

    # Obtain kernel weights
    tau = drv + 4
    L = min(int(tau * h / delta), resol)
    if L == 0:
        warnings.warn(
            "WARNING : Binning grid too coarse for current (small) bandwidth: consider increasing 'resolution'"
        )
    lvec = np.arange(L + 1)
    arg = lvec * delta / h

    dnorm = np.exp(-np.square(arg) / 2) / np.sqrt(2 * np.pi)
    kappam = dnorm / h ** (drv + 1)
    hmold0 = 1
    hmold1 = arg
    hmnew = 1
    if drv >= 2:
        for i in np.arange(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1  # Compute mth degree Hermite polynomial
            hmold1 = hmnew  # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    P = 2 ** (int(np.log(resol + L + 1) / np.log(2)) + 1)
    kappam = np.concatenate((kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]), axis=0)
    Gcounts = np.concatenate((gcounts, np.zeros(P - resol)), axis=0)
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    gcounter = gcounts * np.fft.ifft(kappam * Gcounts).real[0:resol]

    return np.nansum(gcounter) / n**2
