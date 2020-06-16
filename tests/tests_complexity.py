import numpy as np
import pandas as pd
import neurokit2 as nk
import nolds

from pyentrp import entropy as pyentrp

"""
For the testing of complexity, we test our implementations against existing and established ones.
However, some of these other implementations are not really packaged in a way
SO THAT we can easily import them. Thus, we directly copied their content in this file
(below the tests).
"""


# =============================================================================
# Some sanity checks
# =============================================================================
def test_complexity_sanity():

    signal = np.cos(np.linspace(start=0, stop=30, num=1000))

    # Entropy
    assert np.allclose(nk.entropy_fuzzy(signal), nk.entropy_sample(signal, fuzzy=True), atol=0.000001)

    # Fractal
    assert np.allclose(nk.fractal_dfa(signal, windows=np.array([4, 8, 12, 20])), 2.1009048365682133, atol=0.000001)
    assert np.allclose(nk.fractal_dfa(signal), 1.957966586191164, atol=0.000001)
    assert np.allclose(nk.fractal_dfa(signal, multifractal=True), 1.957966586191164, atol=0.000001)

    assert np.allclose(nk.fractal_correlation(signal), 0.7884473170763334, atol=0.000001)
    assert np.allclose(nk.fractal_correlation(signal, r="nolds"), nolds.corr_dim(signal, 2), atol=0.0001)


# =============================================================================
# Comparison against R
# =============================================================================
"""
R code:

library(TSEntropies)
library(pracma)

signal <- read.csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/bio_eventrelated_100hz.csv")$RSP
r <- 0.2 * sd(signal)

# ApEn --------------------------------------------------------------------

TSEntropies::ApEn(signal, dim=2, lag=1, r=r)
0.04383386
TSEntropies::ApEn(signal, dim=3, lag=2, r=1)
0.0004269369
pracma::approx_entropy(signal[1:200], edim=2, r=r, elag=1)
0.03632554

# SampEn ------------------------------------------------------------------

TSEntropies::SampEn(signal[1:300], dim=2, lag=1, r=r)
0.04777648
TSEntropies::FastSampEn(signal[1:300], dim=2, lag=1, r=r)
0.003490405
pracma::sample_entropy(signal[1:300], edim=2, tau=1, r=r)
0.03784376
pracma::sample_entropy(signal[1:300], edim=3, tau=2, r=r)
0.09185509
"""


def test_complexity_vs_R():

    signal = pd.read_csv(
        "https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/bio_eventrelated_100hz.csv"
    )["RSP"].values
    r = 0.2 * np.std(signal, ddof=1)

    # ApEn
    apen = nk.entropy_approximate(signal, dimension=2, r=r)
    assert np.allclose(apen, 0.04383386, atol=0.0001)
    apen = nk.entropy_approximate(signal, dimension=3, delay=2, r=1)
    assert np.allclose(apen, 0.0004269369, atol=0.0001)
    apen = nk.entropy_approximate(signal[0:200], dimension=2, delay=1, r=r)
    assert np.allclose(apen, 0.03632554, atol=0.0001)

    # SampEn
    sampen = nk.entropy_sample(signal[0:300], dimension=2, r=r)
    assert np.allclose(sampen, nk.entropy_sample(signal[0:300], dimension=2, r=r, distance="infinity"), atol=0.001)
    assert np.allclose(sampen, 0.03784376, atol=0.001)
    sampen = nk.entropy_sample(signal[0:300], dimension=3, delay=2, r=r)
    assert np.allclose(sampen, 0.09185509, atol=0.01)


# =============================================================================
# Comparison against Python implementations
# =============================================================================


def test_complexity_vs_Python():

    signal = np.cos(np.linspace(start=0, stop=30, num=100))

    # Shannon
    shannon = nk.entropy_shannon(signal)
    #    assert scipy.stats.entropy(shannon, pd.Series(signal).value_counts())
    assert np.allclose(shannon - pyentrp.shannon_entropy(signal), 0)

    # Approximate
    assert np.allclose(nk.entropy_approximate(signal), 0.17364897858477146)
    assert np.allclose(
        nk.entropy_approximate(signal, dimension=2, r=0.2 * np.std(signal, ddof=1)) - entropy_app_entropy(signal, 2), 0
    )

    assert nk.entropy_approximate(signal, dimension=2, r=0.2 * np.std(signal, ddof=1)) != pyeeg_ap_entropy(
        signal, 2, 0.2 * np.std(signal, ddof=1)
    )

    # Sample
    assert np.allclose(
        nk.entropy_sample(signal, dimension=2, r=0.2 * np.std(signal, ddof=1)) - entropy_sample_entropy(signal, 2), 0
    )
    assert np.allclose(nk.entropy_sample(signal, dimension=2, r=0.2) - nolds.sampen(signal, 2, 0.2), 0)
    assert np.allclose(nk.entropy_sample(signal, dimension=2, r=0.2) - entro_py_sampen(signal, 2, 0.2, scale=False), 0)
    assert np.allclose(nk.entropy_sample(signal, dimension=2, r=0.2) - pyeeg_samp_entropy(signal, 2, 0.2), 0)

    #    import sampen
    #    sampen.sampen2(signal[0:300], mm=2, r=r)

    assert nk.entropy_sample(signal, dimension=2, r=0.2) != pyentrp.sample_entropy(signal, 2, 0.2)[1]
    assert (
        nk.entropy_sample(signal, dimension=2, r=0.2 * np.sqrt(np.var(signal)))
        != MultiscaleEntropy_sample_entropy(signal, 2, 0.2)[0.2][2]
    )

    # MSE
    #    assert nk.entropy_multiscale(signal, 2, 0.2*np.sqrt(np.var(signal))) != np.trapz(MultiscaleEntropy_mse(signal, [i+1 for i in range(10)], 2, 0.2, return_type="list"))
    #    assert nk.entropy_multiscale(signal, 2, 0.2*np.std(signal, ddof=1)) != np.trapz(pyentrp.multiscale_entropy(signal, 2, 0.2, 10))

    # Fuzzy
    assert np.allclose(
        nk.entropy_fuzzy(signal, dimension=2, r=0.2, delay=1) - entro_py_fuzzyen(signal, 2, 0.2, 1, scale=False), 0
    )

    # DFA
    assert nk.fractal_dfa(signal, windows=np.array([4, 8, 12, 20])) != nolds.dfa(
        signal, nvals=[4, 8, 12, 20], fit_exp="poly"
    )


# =============================================================================
# Wikipedia
# =============================================================================
def wikipedia_sampen(signal, m=2, r=1):
    N = len(signal)
    B = 0.0
    A = 0.0

    # Split time series and save all templates of length m
    xmi = np.array([signal[i : i + m] for i in range(N - m)])
    xmj = np.array([signal[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([signal[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)


# =============================================================================
# Pyeeg
# =============================================================================


def pyeeg_embed_seq(time_series, tau, embedding_dimension):
    if not type(time_series) == np.ndarray:
        typed_time_series = np.asarray(time_series)
    else:
        typed_time_series = time_series

    shape = (typed_time_series.size - tau * (embedding_dimension - 1), embedding_dimension)

    strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)

    return np.lib.stride_tricks.as_strided(typed_time_series, shape=shape, strides=strides)


def pyeeg_bin_power(X, Band, Fs):
    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(C[int(np.floor(Freq / Fs * len(X))) : int(np.floor(Next_Freq / Fs * len(X)))])
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio


def pyeeg_ap_entropy(X, M, R):
    N = len(X)

    Em = pyeeg_embed_seq(X, 1, M)
    A = np.tile(Em, (len(Em), 1, 1))
    B = np.transpose(A, [1, 0, 2])
    D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
    InRange = np.max(D, axis=2) <= R

    # Probability that random M-sequences are in range
    Cm = InRange.mean(axis=0)

    # M+1-sequences in range if M-sequences are in range & last values are close
    Dp = np.abs(np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T)

    Cmp = np.logical_and(Dp <= R, InRange[:-1, :-1]).mean(axis=0)

    Phi_m, Phi_mp = np.sum(np.log(Cm)), np.sum(np.log(Cmp))

    Ap_En = (Phi_m - Phi_mp) / (N - M)

    return Ap_En


def pyeeg_samp_entropy(X, M, R):
    N = len(X)

    Em = pyeeg_embed_seq(X, 1, M)[:-1]
    A = np.tile(Em, (len(Em), 1, 1))
    B = np.transpose(A, [1, 0, 2])
    D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
    InRange = np.max(D, axis=2) <= R
    np.fill_diagonal(InRange, 0)  # Don't count self-matches

    Cm = InRange.sum(axis=0)  # Probability that random M-sequences are in range
    Dp = np.abs(np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T)

    Cmp = np.logical_and(Dp <= R, InRange).sum(axis=0)

    # Avoid taking log(0)
    Samp_En = np.log(np.sum(Cm + 1e-100) / np.sum(Cmp + 1e-100))

    return Samp_En


# =============================================================================
# Entropy
# =============================================================================


from sklearn.neighbors import KDTree


def entropy_embed(x, order=3, delay=1):
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay : i * delay + Y.shape[1]]
    return Y.T


def entropy_app_samp_entropy(x, order, metric="chebyshev", approximate=True):
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError(
            "The given metric (%s) is not valid. The valid " "metric names are: %s" % (metric, _all_metrics)
        )
    phi = np.zeros(2)
    r = 0.2 * np.std(x, axis=-1, ddof=1)

    # compute phi(order, r)
    _emb_data1 = entropy_embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r, count_only=True).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = entropy_embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r, count_only=True).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


def entropy_app_entropy(x, order=2, metric="chebyshev"):
    phi = entropy_app_samp_entropy(x, order=order, metric=metric, approximate=True)
    return np.subtract(phi[0], phi[1])


def entropy_sample_entropy(x, order=2, metric="chebyshev"):
    x = np.asarray(x, dtype=np.float64)
    phi = entropy_app_samp_entropy(x, order=order, metric=metric, approximate=False)
    return -np.log(np.divide(phi[1], phi[0]))


# =============================================================================
# entro-py
# =============================================================================


def entro_py_sampen(x, dim, r, scale=True):
    return entro_py_entropy(x, dim, r, scale=scale)


def entro_py_cross_sampen(x1, x2, dim, r, scale=True):
    return entro_py_entropy([x1, x2], dim, r, scale)


def entro_py_fuzzyen(x, dim, r, n, scale=True):
    return entro_py_entropy(x, dim, r, n=n, scale=scale, remove_baseline=True)


def entro_py_cross_fuzzyen(x1, x2, dim, r, n, scale=True):
    return entro_py_entropy([x1, x2], dim, r, n, scale=scale, remove_baseline=True)


def entro_py_pattern_mat(x, m):
    x = np.asarray(x).ravel()
    if m == 1:
        return x
    else:
        N = len(x)
        patterns = np.zeros((m, N - m + 1))
        for i in range(m):
            patterns[i, :] = x[i : N - m + i + 1]
        return patterns


def entro_py_entropy(x, dim, r, n=1, scale=True, remove_baseline=False):
    fuzzy = True if remove_baseline else False
    cross = True if type(x) == list else False
    N = len(x[0]) if cross else len(x)

    if scale:
        if cross:
            x = [entro_py_scale(np.copy(x[0])), entro_py_scale(np.copy(x[1]))]
        else:
            x = entro_py_scale(np.copy(x))

    phi = [0, 0]  # phi(m), phi(m+1)
    for j in [0, 1]:
        m = dim + j
        npat = N - dim  # https://github.com/ixjlyons/entro-py/pull/2/files
        if cross:
            #            patterns = [entro_py_pattern_mat(x[0], m), entro_py_pattern_mat(x[1], m)]
            patterns = [
                entro_py_pattern_mat(x[0], m)[:, :npat],
                entro_py_pattern_mat(x[1], m)[:, :npat],
            ]  # https://github.com/ixjlyons/entro-py/pull/2/files
        else:
            #            patterns = entro_py_pattern_mat(x, m)
            patterns = entro_py_pattern_mat(x, m)[:, :npat]

        if remove_baseline:
            if cross:
                patterns[0] = entro_py_remove_baseline(patterns[0], axis=0)
                patterns[1] = entro_py_remove_baseline(patterns[1], axis=0)
            else:
                patterns = entro_py_remove_baseline(patterns, axis=0)

        #        count = np.zeros(N-m)  # https://github.com/ixjlyons/entro-py/pull/2/files
        #        for i in range(N-m):  # https://github.com/ixjlyons/entro-py/pull/2/files
        count = np.zeros(npat)
        for i in range(npat):
            if cross:
                if m == 1:
                    sub = patterns[1][i]
                else:
                    sub = patterns[1][:, [i]]
                dist = np.max(np.abs(patterns[0] - sub), axis=0)
            else:
                if m == 1:
                    sub = patterns[i]
                else:
                    sub = patterns[:, [i]]
                dist = np.max(np.abs(patterns - sub), axis=0)

            if fuzzy:
                sim = np.exp(-np.power(dist, n) / r)
            else:
                sim = dist < r

            count[i] = np.sum(sim) - 1

        #        phi[j] = np.mean(count) / (N-m-1)
        phi[j] = np.mean(count) / (N - dim - 1)  # https://github.com/ixjlyons/entro-py/pull/2/files

    return np.log(phi[0] / phi[1])


def entro_py_scale(x, axis=None):
    x = entro_py_remove_baseline(x, axis=axis)
    x /= np.std(x, ddof=1, axis=axis, keepdims=True)
    return x


def entro_py_remove_baseline(x, axis=None):
    x -= np.mean(x, axis=axis, keepdims=True)
    return x


# =============================================================================
# MultiscaleEntropy https://github.com/reatank/MultiscaleEntropy/blob/master/MultiscaleEntropy/mse.py
# =============================================================================

import math
from collections.abc import Iterable


def MultiscaleEntropy_init_return_type(return_type):
    if return_type == "dict":
        return {}
    else:
        return []


def MultiscaleEntropy_check_type(x, num_type, name):
    if isinstance(x, num_type):
        tmp = [x]
    elif not isinstance(x, Iterable):
        raise ValueError(name + " should be a " + num_type.__name__ + " or an iterator of " + num_type.__name__)
    else:
        tmp = []
        for i in x:
            tmp.append(i)
            if not isinstance(i, num_type):
                raise ValueError(name + " should be a " + num_type.__name__ + " or an iterator of " + num_type.__name__)
    return tmp


# sum of seperate intervals of x
def MultiscaleEntropy_coarse_grain(x, scale_factor):
    x = np.array(x)
    x_len = len(x)
    if x_len % scale_factor:
        padded_len = (1 + int(x_len / scale_factor)) * scale_factor
    else:
        padded_len = x_len
    tmp_x = np.zeros(padded_len)
    tmp_x[:x_len] = x
    tmp_x = np.reshape(tmp_x, (int(padded_len / scale_factor), scale_factor))
    ans = np.reshape(np.sum(tmp_x, axis=1), (-1)) / scale_factor

    return ans


def MultiscaleEntropy_sample_entropy(x, m=[2], r=[0.15], sd=None, return_type="dict", safe_mode=False):
    """[Sample Entropy, the threshold will be r*sd]

    Arguments:
        x {[input signal]} -- [an iterator of numbers]

    Keyword Arguments:
        m {list} -- [m in sample entropy] (default: {[2]})
        r {list} -- [r in sample entropy] (default: {[0.15]})
        sd {number} -- [standard derivation of x, if None, will be calculated] (default: {None})
        return_type {str} -- [can be dict or list] (default: {'dict'})
        safe_mode {bool} -- [if set True, type checking will be skipped] (default: {False})

    Raises:
        ValueError -- [some values too big]

    Returns:
        [dict or list as return_type indicates] -- [if dict, nest as [scale_factor][m][r] for each value of m, r; if list, nest as [i][j] for lengths of m, r]
    """
    # type checking
    if not safe_mode:
        m = MultiscaleEntropy_check_type(m, int, "m")
        r = MultiscaleEntropy_check_type(r, float, "r")
        if not (sd == None) and not (isinstance(sd, float) or isinstance(sd, int)):
            raise ValueError("sd should be a number")
    try:
        x = np.array(x)
    except:
        raise ValueError("x should be a sequence of numbers")
    # value checking
    if len(x) < max(m):
        raise ValueError("the max m is bigger than x's length")

    # initialization
    if sd == None:
        sd = np.sqrt(np.var(x))
    ans = MultiscaleEntropy_init_return_type(return_type)

    # calculation
    for i, rr in enumerate(r):
        threshold = rr * sd
        if return_type == "dict":
            ans[rr] = MultiscaleEntropy_init_return_type(return_type)
        else:
            ans.append(MultiscaleEntropy_init_return_type(return_type))
        count = {}
        tmp_m = []
        for mm in m:
            tmp_m.append(mm)
            tmp_m.append(mm + 1)
        tmp_m = list(set(tmp_m))
        for mm in tmp_m:
            count[mm] = 0

        for j in range(1, len(x) - min(m) + 1):
            cont = 0
            for inc in range(0, len(x) - j):
                if abs(x[inc] - x[j + inc]) < threshold:
                    cont += 1
                elif cont > 0:
                    for mm in tmp_m:
                        tmp = cont - mm + 1
                        count[mm] += tmp if tmp > 0 else 0
                    cont = 0
            if cont > 0:
                for mm in tmp_m:
                    tmp = cont - mm + 1
                    count[mm] += tmp if tmp > 0 else 0
        for mm in m:
            if count[mm + 1] == 0 or count[mm] == 0:
                t = len(x) - mm + 1
                tmp = -math.log(1 / (t * (t - 1)))
            else:
                tmp = -math.log(count[mm + 1] / count[mm])
            if return_type == "dict":
                ans[rr][mm] = tmp
            else:
                ans[i].append(tmp)
    return ans


def MultiscaleEntropy_mse(
    x, scale_factor=[i for i in range(1, 21)], m=[2], r=[0.15], return_type="dict", safe_mode=False
):
    """[Multiscale Entropy]

    Arguments:
        x {[input signal]} -- [an iterator of numbers]

    Keyword Arguments:
        scale_factor {list} -- [scale factors of coarse graining] (default: {[i for i in range(1,21)]})
        m {list} -- [m in sample entropy] (default: {[2]})
        r {list} -- [r in sample entropy] (default: {[0.15]})
        return_type {str} -- [can be dict or list] (default: {'dict'})
        safe_mode {bool} -- [if set True, type checking will be skipped] (default: {False})

    Raises:
        ValueError -- [some values too big]

    Returns:
        [dict or list as return_type indicates] -- [if dict, nest as [scale_factor][m][r] for each value of scale_factor, m, r; if list nest as [i][j][k] for lengths of scale_factor, m, r]
    """
    # type checking
    if not safe_mode:
        m = MultiscaleEntropy_check_type(m, int, "m")
        r = MultiscaleEntropy_check_type(r, float, "r")
        scale_factor = MultiscaleEntropy_check_type(scale_factor, int, "scale_factor")
    try:
        x = np.array(x)
    except:
        print("x should be a sequence of numbers")
    # value checking
    if max(scale_factor) > len(x):
        raise ValueError("the max scale_factor is bigger than x's length")

    # calculation
    sd = np.sqrt(np.var(x))
    ms_en = MultiscaleEntropy_init_return_type(return_type)
    for s_f in scale_factor:
        y = MultiscaleEntropy_coarse_grain(x, s_f)
        if return_type == "dict":
            ms_en[s_f] = MultiscaleEntropy_sample_entropy(y, m, r, sd, "dict", True)
        else:
            ms_en.append(MultiscaleEntropy_sample_entropy(y, m, r, sd, "list", True))

    if return_type == "list":
        ms_en = [i[0] for i in ms_en]
        ms_en = [i[0] for i in ms_en]
    return ms_en
