import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .entropy_shannon import entropy_shannon


def entropy_phase(signal, delay=1, k=4, show=False, **kwargs):
    """**Phase Entropy (PhasEn)**

    Phase entropy (PhasEn or PhEn) has been developed by quantifying the distribution of the signal
    across *k* parts of a two-dimensional phase space referred to as a second order difference
    plot (SODP). It builds on the concept of :func:`Grid Entropy <entropy_grid>`, that uses
    :func:`Poincaré plot <.hrv_nonlinear>` as its basis.

    The SODP is divided into *k* sectors of equal angle around the origin. The value of each sector
    is the sum of the angles (between 0 and :math:`2\\pi`) of the points that it contains, and the
    :func:`Shannon entropy <entropy_shannon>` of their relative values is divided by
    :math:`\\log(k)`, its maximum, so that PhasEn lies between 0 and 1 (Rohila & Sharma, 2019).

    .. figure:: ../img/rohila2019.png
       :alt: Figure from Rohila et al. (2019).
       :target: https://doi.org/10.1088/1361-6579/ab499e

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    k : int
        The number of sections that the SODP is divided into. It is a coarse graining parameter
        that defines how fine the grid is. It is recommended to use even-numbered (preferably
        multiples of 4) partitions for sake of symmetry.
    show : bool
        Plot the Second Order Difference Plot (SODP).
    **kwargs : optional
        Other keyword arguments, such as the logarithmic ``base`` to use for
        :func:`entropy_shannon` (which does not change the normalized PhasEn).

    Returns
    -------
    phasen : float
        Phase Entropy
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute Phase Entropy
      @savefig p_entropy_phase1.png scale=100%
      phasen, info = nk.entropy_phase(signal, k=4, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      phasen

    .. ipython:: python

      @savefig p_entropy_phase2.png scale=100%
      phasen, info = nk.entropy_phase(signal, k=8, show=True)
      @suppress
      plt.close()

    References
    ----------
    * Rohila, A., & Sharma, A. (2019). Phase entropy: A new complexity measure for heart rate
      variability. Physiological Measurement, 40(10), 105006.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    info = {"k": k, "Delay": delay}
    signal = np.asarray(signal, dtype=float)

    # 1. Compute SODP axes
    y = signal[2 * delay :] - signal[delay:-delay]
    x = signal[delay:-delay] - signal[: -2 * delay]

    # 2. Compute the slope (angle theta, between 0 and 2pi) of each scatter point from the origin
    # (the points on the axes included; a point at the origin gets 0 and so does not contribute)
    theta = np.arctan2(y, x) % (2 * np.pi)

    # 3. The entire plot is divided into k sections having an angle span of 2pi/k radians each
    angles = np.linspace(0, 2 * np.pi, k + 1)
    sectors = np.minimum(theta // (2 * np.pi / k), k - 1).astype(int)

    # 4. The cumulative slope of each sector is obtained by adding the slope of each scatter point
    # within that sector
    # 5. The probability distribution of the slopes in each sector is computed
    freq = np.bincount(sectors, weights=theta, minlength=k)
    freq = freq / np.sum(freq)

    # 6. the Shannon entropy computed from the distribution p(i)
    phasen, _ = entropy_shannon(freq=freq, **kwargs)

    # Normalize by the maximum entropy (all sectors equal), in the same logarithmic base
    phasen = phasen / (np.log(k) / np.log(kwargs.get("base", 2)))

    if show is True:
        limx = np.ceil(np.max(np.abs([y, x])))
        Ys = np.sin(angles) * limx * np.sqrt(2)
        Xs = np.cos(angles) * limx * np.sqrt(2)
        resampled_cmap = plt.get_cmap("jet").resampled(k)
        colors = resampled_cmap(np.linspace(0, 1, k))

        plt.figure()
        for i in range(k):
            plt.plot(x[sectors == i], y[sectors == i], ".", color=tuple(colors[i, :]))
        plt.plot(np.vstack((np.zeros(k + 1), Xs)), np.vstack((np.zeros(k + 1), Ys)), color="red")
        plt.axis([-limx, limx, -limx, limx])
        plt.gca().set_aspect("equal", "box")
        plt.xlabel(r"$X(n +  \tau) - X(n)$")
        plt.ylabel(r"$X(n + 2 \tau) - X(n + \tau)$")
        plt.xticks([-limx, 0, limx])
        plt.yticks([-limx, 0, limx])
        plt.title("Second Order Difference Plot (SODP)")

    return phasen, info
