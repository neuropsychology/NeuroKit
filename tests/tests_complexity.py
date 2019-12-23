import numpy as np
import pandas as pd
import neurokit2 as nk

import nolds

from pyentrp import entropy as pyentrp

"""
For the testing of complexity, we test our implementations against existing and established ones.
However, as some of these other implementations are not really packaged in a way
that we can easily import them. Thus, we directly copied their content in this file
(below the tests).
"""


# =============================================================================
# Complexity
# =============================================================================


def test_complexity():

    signal = np.cos(np.linspace(start=0, stop=30, num=100))


    # Shannon
    assert nk.entropy_shannon(signal) == pyentrp.shannon_entropy(signal)


    # Approximate
    assert nk.entropy_approximate(signal, 2, 0.2*np.std(signal, ddof=1)) == entropy_app_entropy(signal, 2)

    assert nk.entropy_approximate(signal, 2, 0.2*np.std(signal, ddof=1)) != pyeeg_ap_entropy(signal, 2, 0.2*np.std(signal, ddof=1))


    # Sample
    assert nk.entropy_sample(signal, 2, 0.2*np.std(signal, ddof=1)) == entropy_sample_entropy(signal, 2)
    assert nk.entropy_sample(signal, 2, 0.2) == nolds.sampen(signal, 2, 0.2)

    assert nk.entropy_sample(signal, 2, 0.2) != pyentrp.sample_entropy(signal, 2, 0.2)[1]
    assert nk.entropy_sample(signal, 2, 0.2) != pyeeg_samp_entropy(signal, 2, 0.2)
    assert nk.entropy_sample(signal, 2, 0.2) != entro_py_sampen(signal, 2, 0.2, scale=False)


    # Fuzzy
    assert nk.entropy_fuzzy(signal, 2, 0.2) == entro_py_fuzzyen(signal, 2, 0.2, 1, scale=False)




# =============================================================================
# Pyeeg
# =============================================================================


def pyeeg_embed_seq(time_series, tau, embedding_dimension):
    if not type(time_series) == np.ndarray:
        typed_time_series = np.asarray(time_series)
    else:
        typed_time_series = time_series

    shape = (
        typed_time_series.size - tau * (embedding_dimension - 1),
        embedding_dimension
    )

    strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)

    return np.lib.stride_tricks.as_strided(
        typed_time_series,
        shape=shape,
        strides=strides
    )




def pyeeg_bin_power(X, Band, Fs):
    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[int(np.floor(Freq / Fs * len(X))):
                int(np.floor(Next_Freq / Fs * len(X)))]
        )
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
    Dp = np.abs(
        np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T
    )

    Cmp = np.logical_and(Dp <= R, InRange[:-1, :-1]).mean(axis=0)

    Phi_m, Phi_mp = np.sum(np.log(Cm)), np.sum(np.log(Cmp))

    Ap_En = (Phi_m - Phi_mp) / (N - M)

    return Ap_En


def pyeeg_samp_entropy(X, M, R):
    N = len(X)

    Em = pyeeg_embed_seq(X, 1, M)
    A = np.tile(Em, (len(Em), 1, 1))
    B = np.transpose(A, [1, 0, 2])
    D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
    InRange = np.max(D, axis=2) <= R
    np.fill_diagonal(InRange, 0)  # Don't count self-matches

    Cm = InRange.sum(axis=0)  # Probability that random M-sequences are in range
    Dp = np.abs(
        np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T
    )

    Cmp = np.logical_and(Dp <= R, InRange[:-1, :-1]).sum(axis=0)

    # Avoid taking log(0)
    Samp_En = np.log(np.sum(Cm + 1e-100) / np.sum(Cmp + 1e-100))

    return Samp_En

# =============================================================================
# Entropy
# =============================================================================


from math import log, floor
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
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T



def entropy_app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    phi = np.zeros(2)
    r = 0.2 * np.std(x, axis=-1, ddof=1)

    # compute phi(order, r)
    _emb_data1 = entropy_embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = entropy_embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi




def entropy_app_entropy(x, order=2, metric='chebyshev'):
    phi = entropy_app_samp_entropy(x, order=order, metric=metric, approximate=True)
    return np.subtract(phi[0], phi[1])



def entropy_sample_entropy(x, order=2, metric='chebyshev'):
    x = np.asarray(x, dtype=np.float64)
    phi = entropy_app_samp_entropy(x, order=order, metric=metric,
                            approximate=False)
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
        patterns = np.zeros((m, N-m+1))
        for i in range(m):
            patterns[i, :] = x[i:N-m+i+1]
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
        if cross:
            patterns = [entro_py_pattern_mat(x[0], m), entro_py_pattern_mat(x[1], m)]
        else:
            patterns = entro_py_pattern_mat(x, m)

        if remove_baseline:
            if cross:
                patterns[0] = entro_py_remove_baseline(patterns[0], axis=0)
                patterns[1] = entro_py_remove_baseline(patterns[1], axis=0)
            else:
                patterns = entro_py_remove_baseline(patterns, axis=0)

        count = np.zeros(N-m)
        for i in range(N-m):
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
                sim = dist <= r

            count[i] = np.sum(sim) - 1

        phi[j] = np.mean(count) / (N-m-1)

    return np.log(phi[0] / phi[1])


def entro_py_scale(x, axis=None):
    x = entro_py_remove_baseline(x, axis=axis)
    x /= np.std(x, ddof=1, axis=axis, keepdims=True)
    return x


def entro_py_remove_baseline(x, axis=None):
    x -= np.mean(x, axis=axis, keepdims=True)
    return x