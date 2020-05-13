import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.spatial

from .embedding import embedding


def embedding_concurrent(signal, delay_max=100, dimension_max=20, surrogate_iter=5):
    """
    Estimate optimal Dimension (m) and optimal Time Delay (tau) using
    Differential Entropy b method.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    delay_max : int
        The maximum time delay (Tau) to test.
    dimension_max : int
        The maximum embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order') to test.
    surrogate_iter : int
        The maximum surrogates generated using the iAAFT method.

    Returns
    -------
    optimal_dimension : int
        Optimal dimension.
    optimal_delay : int
        Optimal time delay.

    See Also
    ------------
    embedding_dimension, embedding_delay

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Artifical example
    >>> signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)
    >>> optimal_dimension, optimal_delay = nk.embedding_concurrent(signal, delay_max=100, dimension_max=20, surrogate_iter=5)
    >>>
    >>> # Realistic example
    >>> ecg = nk.ecg_simulate(duration=60*6, sampling_rate=150)
    >>> signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=150)[0], sampling_rate=150)
    >>> optimal_dimension, optimal_delay = nk.embedding_concurrent(signal, delay_max=500, dimension_max=10, surrogate_iter=5)

    References
    -----------
    - Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based method for determining the optimal embedding parameters of a signal. In 2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.(ICASSP'03). (Vol. 6, pp. VI-29). IEEE.
    """

    # Initalize vectors
    if isinstance(delay_max, int):
        tau_sequence = np.arange(1, delay_max)
    else:
        tau_sequence = np.array(delay_max)
    if isinstance(dimension_max, int):
        dimension_seq = np.arange(1, dimension_max + 1)
    else:
        dimension_seq = np.array(dimension_max)

    N = len(signal)

    surrogate_list = []
    optimal = {}

    for dimension in dimension_seq:
        optimal[dimension] = []
        # Calculate differential entropy for each embedded
        for tau in tau_sequence:
            signal_embedded = embedding(signal, delay=tau, dimension=dimension)
            signal_entropy = _differential_entropy(signal_embedded, k=1)

            # calculate average of surrogates entropy
            for inter in range(surrogate_iter):
                surrogate, i, rmsd = iaaft(signal)
                surrogate_embedded = embedding(surrogate, delay=tau, dimension=dimension)
                surrogate_entropy = _differential_entropy(surrogate_embedded, k=1)
                surrogate_list.append(surrogate_entropy)
                surrogate_entropy_average = sum(surrogate_list) / len(surrogate_list)

            # entropy ratio for each set of d and tau
            entropy_ratio = signal_entropy / surrogate_entropy_average + (dimension*np.log(N)) / N
            optimal[dimension].append(entropy_ratio)

    # optimal dimension and tau is where entropy_ratio is minimum
    optimal_df = pd.DataFrame.from_dict(optimal)
    optimal_delay, optimal_dimension = np.unravel_index(np.nanargmin(optimal_df.values), optimal_df.shape)

    optimal_delay = optimal_delay + 1  # accounts for zero indexing

    return optimal_dimension, optimal_delay

# =============================================================================
# Internals
# =============================================================================

def iaaft(signal, max_iter=1000, atol=1e-8, rtol=1e-10):
    """
    Return iterative amplitude adjusted Fourier transform surrogates.
    Returns phase randomized, amplitude adjusted (IAAFT) surrogates with
    the same power spectrum (to a very high accuracy) and distribution
    as the original data using an iterative scheme.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    max_iter : int
        Maximum iterations to be performed while checking for
        convergence. Convergence can be achieved before maximum interation.
    atol : float
        Absolute tolerance for checking convergence.
    rtol : float
        Relative tolerance for checking convergence. If both atol and rtol are
        set to zero,  the iterations end only when the RMSD stops changing or
        when maximum iteration is reached.

    Returns
    -------
    surrogate : array
        Surrogate series with (almost) the same power spectrum and
        distribution.
    i : int
        Number of iterations that have been performed.
    rmsd : float
        Root-mean-square deviation (RMSD) between the absolute squares
        of the Fourier amplitudes of the surrogate series and that of
        the original series.

    References
    -----
    Schreiber, T., & Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. Physical review letters, 77(4), 635.
    `entropy_estimators` <https://github.com/paulbrodersen/entropy_estimators>`_
    """
    # Calculate "true" Fourier amplitudes and sort the series
    amplitudes = np.abs(np.fft.rfft(signal))
    sort = np.sort(signal)

    # Previous and current error
    previous_error, current_error = (-1, 1)

    # Start with a random permutation
    t = np.fft.rfft(np.random.permutation(signal))

    for i in range(max_iter):
        # Match power spectrum
        s = np.real(np.fft.irfft(amplitudes * t / np.abs(t), n=len(signal)))

        # Match distribution by rank ordering
        surrogate = sort[np.argsort(np.argsort(s))]

        t = np.fft.rfft(surrogate)
        current_error = np.sqrt(np.mean((amplitudes ** 2 - np.abs(t) ** 2) ** 2))

        # Check convergence
        if abs(current_error - previous_error) <= atol + rtol * abs(previous_error):
            break
        else:
            previous_error = current_error

    # Normalize error w.r.t. mean of the "true" power spectrum.
    rmsd = current_error / np.mean(amplitudes ** 2)
    return surrogate, i, rmsd


def _differential_entropy(x, k=1, norm='max', min_dist=0.):
    """
    Estimates the entropy H of a random variable x based on
    the kth-nearest neighbour distances between point samples.

    Parameters:
    ----------
    x: (n, d) ndarray
        n samples from a d-dimensional multivariate distribution
    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution
    norm: 'euclidean' or 'max'
        p-norm used when computing k-nearest neighbour distances
    min_dist: float (default 0.)
        minimum distance between data points;
        smaller distances will be capped using this value
    Returns:
    --------
    h: float
        entropy H(X)

    References
    -----
    Kozachenko, L., & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9–16.
    `NoLiTSA` <https://github.com/manu-mannattil/nolitsa>`_
    """

    n, d = x.shape

    if norm == 'max':  # max norm
        p = np.inf
        log_c_d = 0  # volume of the d-dimensional unit ball
    elif norm == 'euclidean':  # euclidean norm
        p = 2
        log_c_d = (d / 2.) * np.log(np.pi) - np.log(scipy.special.gamma(d / 2. + 1))
    else:
        raise ValueError("NeuroKit error: differential_entropy(): 'method' "
                         "not recognized.")

    kdtree = scipy.spatial.cKDTree(x)

    # Query all points -- k+1 as query point also in initial set
    distances, _ = kdtree.query(x, k + 1, eps=0, p=p)
    distances = distances[:, -1]

    # Enforce non-zero distances
    distances[distances < min_dist] = min_dist

    sum_log_dist = np.sum(np.log(2*distances))  # 2*radius=diameter
    h = -scipy.special.digamma(k) + scipy.special.digamma(n) + log_c_d + (d / float(n)) * sum_log_dist

    return h
