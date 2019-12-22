# https://github.com/gilestrolab/pyrem/blob/master/src/pyrem/univariate.py

r"""
==================================================
Feature computation for univariate time series
==================================================
This sub-module provides routines for computing features on univariate time series.
Many functions are improved version of PyEEG [PYEEG]_ functions. Be careful,
some functions will give different results compared to PyEEG as the maths have been changed to match original definitions.
Have a look at the documentation notes/ source code to know more.
Here a list of the functions that were reimplemented:
* Approximate entropy :func:`~pyrem.univariate.ap_entropy` [RIC00]_
* Fisher information :func:`~pyrem.univariate.fisher_info` [PYEEG]_
* Higuchi fractal dimension  :func:`~pyrem.univariate.hfd` [HIG88]_
* Hjorth parameters :func:`~pyrem.univariate.hjorth` [HJO70]_
* Petrosian fractal dimension :func:`~pyrem.univariate.pfd` [PET95]_
* Sample entropy :func:`~pyrem.univariate.samp_entropy` [RIC00]_
* Singular value decomposition entropy :func:`~pyrem.univariate.svd_entropy` [PYEEG]_
* Spectral entropy :func:`~pyrem.univariate.spectral_entropy` [PYEEG]_
.. [PET95]  A. Petrosian, Kolmogorov complexity of finite sequences and recognition of different preictal EEG patterns, in ,
    Proceedings of the Eighth IEEE Symposium on Computer-Based Medical Systems, 1995, 1995, pp. 212-217.
.. [PYEEG] F. S. Bao, X. Liu, and C. Zhang, PyEEG: An Open Source Python Module for EEG/MEG Feature Extraction,
    Computational Intelligence and Neuroscience, vol. 2011, p. e406391, Mar. 2011.
.. [HJO70] B. Hjorth, EEG analysis based on time domain properties,
    Electroencephalography and Clinical Neurophysiology, vol. 29, no. 3, pp. 306-310, Sep. 1970.
.. [COS05] M. Costa, A. L. Goldberger, and C.-K. Peng, "Multiscale entropy analysis of biological signals," Phys. Rev. E, vol. 71, no. 2, p. 021906, Feb. 2005.
.. [RIC00] J. S. Richman and J. R. Moorman, "Physiological time-series analysis using approximate entropy and sample entropy,"
    American Journal of Physiology - Heart and Circulatory Physiology, vol. 278, no. 6, pp. H2039-H2049, Jun. 2000.
.. [HIG88] T. Higuchi, "Approach to an irregular time series on the basis of the fractal theory," Physica D: Nonlinear Phenomena, vol. 31, no. 2, pp. 277-283, Jun. 1988.
"""


__author__ = 'quentin'
import numpy as np


def _embed_seq(X,tau,de):

    N =len(X)

    if de * tau > N:
        raise ValueError("Cannot build such a matrix, because D * Tau > N")

    if tau<1:
        raise ValueError("Tau has to be at least 1")


    Y=np.zeros((de, N - (de - 1) * tau))

    for i in range(de):
        Y[i] = X[i *tau : i*tau + Y.shape[1] ]

    return Y.T

def _make_cmp(X, M, R, in_range_i, in_range_j):
     #Then we make Cmp
    N = len(X)

    Emp = _embed_seq(X, 1, M + 1)
    inrange_cmp = np.abs(Emp[in_range_i,-1] - Emp[in_range_j,-1]) <= R

    in_range_cmp_i = in_range_i[inrange_cmp]


    Cmp = np.bincount(in_range_cmp_i, minlength=N-M)
    in_range_cmp_j = in_range_j[inrange_cmp]
    Cmp += np.bincount(in_range_cmp_j, minlength=N-M)

    return Cmp.astype(np.float)

def _coarse_grainning(a, tau):
    """
    Coarse grainning for multiscale (sample) entropy.
    """

    if tau ==1:
        return a
    length_out = a.size / tau

    n_dropped = a.size % tau
    mat = a[0:a.size - n_dropped].reshape((tau, length_out))
    return np.mean(mat, axis=0)

def _make_cm(X,M,R):
    N = len(X)

    # we pregenerate all indices

    i_idx,j_idx  = np.triu_indices(N - M)

    # We start by making Cm
    Em = _embed_seq(X, 1, M)
    dif =  np.abs(Em[i_idx] - Em[j_idx])
    max_dist = np.max(dif, 1)
    inrange_cm = max_dist <= R


    in_range_i = i_idx[inrange_cm]
    in_range_j = j_idx[inrange_cm]


    Cm = np.bincount(in_range_i, minlength=N-M+1)
    Cm += np.bincount(in_range_j, minlength=N-M+1)

    inrange_last = np.max(np.abs(Em[:-1] - Em[-1]),1) <= R
    Cm[inrange_last] += 1
    # all matches + self match
    Cm[-1] += np.sum(inrange_last) + 1

    return Cm.astype(np.float), in_range_i, in_range_j

def pfd(a):
    r"""
    Compute Petrosian Fractal Dimension of a time series [PET95]_.
    It is defined by:
    .. math::
        \frac{log(N)}{log(N) + log(\frac{N}{N+0.4N_{\delta}})}
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which implemented an apparently erroneous formulae:
        .. math::
            \frac{log(N)}{log(N) + log(\frac{N}{N}+0.4N_{\delta})}
    Where:
    :math:`N` is the length of the time series, and
    :math:`N_{\delta}` is the number of sign changes.
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :return: the Petrosian Fractal Dimension; a scalar.
    :rtype: float
    Example:
    >>> import pyrem as pr
    >>> import numpy as np
    >>> # generate white noise:
    >>> noise = np.random.normal(size=int(1e4))
    >>> pr.univariate.pdf(noise)
    """

    diff = np.diff(a)
    # x[i] * x[i-1] for i in t0 -> tmax
    prod = diff[1:-1] * diff[0:-2]

    # Number of sign changes in derivative of the signal
    N_delta = np.sum(prod < 0)
    n = len(a)

    return np.log(n)/(np.log(n)+np.log(n/(n+0.4*N_delta)))

def hjorth(a):
    r"""
    Compute Hjorth parameters [HJO70]_.
    .. math::
        Activity = m_0 = \sigma_{a}^2
    .. math::
        Complexity = m_2 = \sigma_{d}/ \sigma_{a}
    .. math::
        Morbidity = m_4 =  \frac{\sigma_{dd}/ \sigma_{d}}{m_2}
    Where:
    :math:`\sigma_{x}^2` is the mean power of a signal :math:`x`. That is, its variance, if it's mean is zero.
    :math:`a`, :math:`d` and :math:`dd` represent the original signal, its first and second derivatives, respectively.
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which appear to uses a non normalised (by the length of the signal) definition of the activity:
        .. math::
            \sigma_{a}^2 = \sum{\mathbf{x}[i]^2}
        As opposed to
        .. math::
            \sigma_{a}^2 = \frac{1}{n}\sum{\mathbf{x}[i]^2}
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :return: activity, complexity and morbidity
    :rtype: tuple(float, float, float)
    Example:
    >>> import pyrem as pr
    >>> import numpy as np
    >>> # generate white noise:
    >>> noise = np.random.normal(size=int(1e4))
    >>> activity, complexity, morbidity = pr.univariate.hjorth(noise)
    """

    first_deriv = np.diff(a)
    second_deriv = np.diff(a,2)

    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity

def svd_entropy(a, tau, de):
    r"""
    Compute the Singular Value Decomposition entropy of a signal with embedding dimension "de" and delay "tau" [PYEEG]_.
    .. note::
        **Difference with PyEEG:**
        The result differs from PyEEG implementation because :math:`log_2` is used (as opposed to natural logarithm in PyEEG code),
        according to the definition in their paper [PYEEG]_ (eq. 9):
        .. math::
            H_{SVD} = -\sum{\bar\sigma{}_i log_2 \bar\sigma{}_i}
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :param tau: the delay
    :type tau: int
    :param de: the embedding dimension
    :type de: int
    :return: the SVD entropy, a scalar
    :rtype: float
    """

    mat =  _embed_seq(a, tau, de)
    W = np.linalg.svd(mat, compute_uv = False)
    W /= sum(W) # normalize singular values
    return -1*sum(W * np.log2(W))

def fisher_info(a, tau, de):
    r"""
    Compute the Fisher information of a signal with embedding dimension "de" and delay "tau" [PYEEG]_.
    Vectorised (i.e. faster) version of the eponymous PyEEG function.
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :param tau: the delay
    :type tau: int
    :param de: the embedding dimension
    :type de: int
    :return: the Fisher information, a scalar
    :rtype: float
    """

    mat =  _embed_seq(a, tau, de)
    W = np.linalg.svd(mat, compute_uv = False)
    W /= sum(W) # normalize singular values
    FI_v = (W[1:] - W[:-1]) **2 / W[:-1]

    return np.sum(FI_v)

def ap_entropy(a, m, R):
    r"""
    Compute the approximate entropy of a signal with embedding dimension "de" and delay "tau" [PYEEG]_.
    Vectorised version of the PyEEG function. Faster than PyEEG, but still critically slow.
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :param m: the scale
    :type m: int
    :param R: The tolerance
    :type R: float`
    :return: the approximate entropy, a scalar
    :rtype: float
    """

    N = len(a)
    Cm, in_range_i, in_range_j = _make_cm(a,m,R)

    Cmp = _make_cmp(a, m, R, in_range_i, in_range_j)

    Cm /= float((N - m +1 ))
    Cmp /= float(N - m)

    Phi_m, Phi_mp = np.sum(np.log(Cm)),  np.sum(np.log(Cmp))
    Ap_En = (Phi_m - Phi_mp) / (N - m)
    return Ap_En

def samp_entropy(a, m, r, tau=1, relative_r=True):
    r"""
    Compute the sample entropy [RIC00]_ of a signal with embedding dimension `de` and delay `tau` [PYEEG]_.
    Vectorised version of the eponymous PyEEG function.
    In addition, this function can also be used to vary tau and therefore compute Multi-Scale Entropy(MSE) [COS05]_ by
    coarse grainning the time series (see example bellow).
    By default, r is expressed as relatively to the standard deviation of the signal.
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :param m: the scale
    :type m: int
    :param r: The tolerance
    :type r: float
    :param tau: The scale for coarse grainning.
    :type tau: int
    :param relative_r: whether the argument r is relative to the standard deviation. If false, an absolute value should be given for r.
    :type relative_r: true
    :return: the approximate entropy, a scalar
    :rtype: float
    Example:
    >>> import pyrem as pr
    >>> import numpy as np
    >>> # generate white noise:
    >>> noise = np.random.normal(size=int(1e4))
    >>> pr.univariate.samp_entropy(noise, m=2, r=1.5)
    >>> # now we can do that for multiple scales (MSE):
    >>> [pr.univariate.samp_entropy(noise, m=2, r=1.5, tau=tau) for tau in range(1, 5)]
    """


    coarse_a = _coarse_grainning(a, tau)
    if relative_r:
        coarse_a /= np.std(coarse_a)
    embsp = _embed_seq(coarse_a, 1 , m + 1)
    embsp_last = embsp[:,-1]
    embs_mini = embsp[:, :-1]


    # Buffers are preallocated chunks of memory storing temporary results.
    # see the `out` argument in numpy *ufun* documentation

    dist_buffer = np.zeros(embsp.shape[0] - 1, dtype=np.float32)
    subtract_buffer = np.zeros((dist_buffer.size ,m), dtype=np.float32)
    in_range_buffer = np.zeros_like(dist_buffer, dtype=np.bool)
    sum_cm, sum_cmp = 0.0, 0.0

    # we iterate through all templates (rows), except last one.
    for i,template in enumerate(embs_mini[:-1]):

        # these are just views to the buffer arrays. to store intermediary matrices
        dist_b_view = dist_buffer[i:]
        sub_b_view = subtract_buffer[i:]
        range_b_view = in_range_buffer[i:]
        embsp_view = embsp_last[i+1:]

        # substract the template from each subsequent row of the embedded matrix
        np.subtract(embs_mini[i+1:],  template, out=sub_b_view)
        # Absolute distance
        np.abs(sub_b_view, out=sub_b_view)
        # Maximal absolute difference between a scroll and a template is the distance
        np.max(sub_b_view, axis=1, out=dist_b_view)
        # we compare this distance to a tolerance r
        np.less_equal(dist_b_view, r, out= range_b_view)
        # score one for this template for each match
        in_range_sum = np.sum(range_b_view)
        sum_cm  += in_range_sum

        ### reuse the buffers for last column
        dist_b_view = dist_buffer[:in_range_sum]

        where = np.flatnonzero(range_b_view)
        dist_b_view= np.take(embsp_view,where,out=dist_b_view)
        range_b_view = in_range_buffer[range_b_view]
        # score one to TODO for each match of the last element
        dist_b_view -= embsp_last[i]
        np.abs(dist_b_view, out=dist_b_view)
        np.less_equal(dist_b_view, r, out=range_b_view)
        sum_cmp += np.sum(range_b_view)

    if sum_cm == 0 or sum_cmp ==0:
        return np.NaN
    return np.log(sum_cm/sum_cmp)

def spectral_entropy(a, sampling_freq, bands=None):

    r"""
    Compute spectral entropy of a  signal with respect to frequency bands.
    The power spectrum is computed through fft. Then, it is normalised and assimilated to a probability density function.
    The entropy of the signal :math:`x` can be expressed by:
    .. math::
        H(x) =  -\sum_{f=0}^{f = f_s/2} PSD(f) log_2[PSD(f)]
    Where:
    :math:`PSD` is the normalised power spectrum (Power Spectrum Density), and
    :math:`f_s` is the sampling frequency
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :param sampling_freq: the sampling frequency
    :type sampling_freq:  float
    :param bands: a list of numbers delimiting the bins of the frequency bands. If None the entropy is computed over the whole range of the DFT (from 0 to :math:`f_s/2`)
    :return: the spectral entropy; a scalar
    """



    psd = np.abs(np.fft.rfft(a))**2
    psd /= np.sum(psd) # psd as a pdf (normalised to one)

    if bands is None:
        power_per_band= psd[psd>0]
    else:
        freqs = np.fft.rfftfreq(a.size, 1/float(sampling_freq))
        bands = np.asarray(bands)

        freq_limits_low = np.concatenate([[0.0],bands])
        freq_limits_up = np.concatenate([bands, [np.Inf]])

        power_per_band = [np.sum(psd[np.bitwise_and(freqs >= low, freqs<up)])
                for low,up in zip(freq_limits_low, freq_limits_up)]

        power_per_band= power_per_band[ power_per_band > 0]

    return - np.sum(power_per_band * np.log2(power_per_band))



def hfd(a, k_max):

    r"""
    Compute Higuchi Fractal Dimension of a time series.
    Vectorised version of the eponymous [PYEEG]_ function.
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which appears to have implemented an erroneous formulae.
        [HIG88]_ defines the normalisation factor as:
        .. math::
            \frac{N-1}{[\frac{N-m}{k} ]\dot{} k}
        [PYEEG]_ implementation uses:
        .. math::
            \frac{N-1}{[\frac{N-m}{k}]}
        The latter does *not* give the expected fractal dimension of approximately `1.50` for brownian motion (see example bellow).
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :param k_max: the maximal value of k
    :type k_max: int
    :return: Higuchi's fractal dimension; a scalar
    :rtype: float
    Example from [HIG88]_. This should produce a result close to `1.50`:
    >>> import numpy as np
    >>> import pyrem as pr
    >>> i = np.arange(2 ** 15) +1001
    >>> z = np.random.normal(size=int(2 ** 15) + 1001)
    >>> y = np.array([np.sum(z[1:j]) for j in i])
    >>> pr.univariate.hfd(y,2**8)
    """

    L = []
    x = []
    N = a.size


    # TODO this could be used to pregenerate k and m idxs ... but memory pblem?
    # km_idxs = np.triu_indices(k_max - 1)
    # km_idxs = k_max - np.flipud(np.column_stack(km_idxs)) -1
    # km_idxs[:,1] -= 1
    #

    for k in range(1,k_max):
        Lk = 0
        for m in range(0,k):
            #we pregenerate all idxs
            idxs = np.arange(1,int(np.floor((N-m)/k)),dtype=np.int32)

            Lmk = np.sum(np.abs(a[m+idxs*k] - a[m+k*(idxs-1)]))
            Lmk = (Lmk*(N - 1)/(((N - m)/ k)* k)) / k
            Lk += Lmk


        L.append(np.log(Lk/(m+1)))
        x.append([np.log(1.0/ k), 1])

    (p, r1, r2, s)=np.linalg.lstsq(x, L)
    return p[0]




def dfa(X, Ave = None, L = None, sampling= 1):
    """
    WIP on this function. It is basically copied and pasted from [PYEEG]_, without verification of the maths or unittests.
    """
    X = np.array(X)
    if Ave is None:
        Ave = np.mean(X)
    Y = np.cumsum(X)
    Y -= Ave
    if not L:
        max_power = np.int(np.log2(len(X)))-4
        L = X.size / 2 ** np.arange(4,max_power)
    if len(L)<2:
        raise Exception("Too few values for L. Time series too short?")
    F = np.zeros(len(L)) # F(n) of different given box length n

    for i,n in enumerate(L):
        sampled = 0
        for j in range(0,len(X) -n ,n):

            if np.random.rand() < sampling:
                F[i] += np.polyfit(np.arange(j,j+n), Y[j:j+n],1, full=True)[1]
                sampled += 1
        if sampled > 0:
            F[i] /= float(sampled)

    LF = np.array([(l,f) for l,f in zip(L,F) if l>0]).T

    F = np.sqrt(LF[1])
    Alpha = np.polyfit(np.log(LF[0]), np.log(F),1)[0]
    return Alpha

def hurst(signal):
    """
    **Experimental**/untested implementation taken from:
    http://drtomstarke.com/index.php/calculation-of-the-hurst-exponent-to-test-for-trend-and-mean-reversion/
    Use at your own risks.
    """
    tau = []; lagvec = []

    #  Step through the different lags
    for lag in range(2,20):

    #  produce price difference with lag
        pp = np.subtract(signal[lag:],signal[:-lag])

    #  Write the different lags into a vector
        lagvec.append(lag)

    #  Calculate the variance of the difference vector
        tau.append(np.std(pp))

    #  linear fit to double-log graph (gives power)
    m = np.polyfit(np.log10(lagvec),np.log10(tau),1)

    # calculate hurst
    hurst = m[0]

    return hurst
