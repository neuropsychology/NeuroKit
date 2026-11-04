from collections.abc import Iterable

import antropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sklearn.neighbors
from pyentrp import entropy as pyentrp

# import EntropyHub
import neurokit2 as nk
from neurokit2.complexity.optim_complexity_optimize import _complexity_dimension


# For the testing of complexity, we test our implementations against existing and established ones.
# However, some of these other implementations are not really packaged in a way
# SO THAT we can easily import them. Thus, we directly copied their content in this file
# (below the tests).


# =============================================================================
# Some sanity checks
# =============================================================================
def test_complexity_sanity():
    signal = np.cos(np.linspace(start=0, stop=30, num=1000))
    mdfa_q = [-5, -3, -1, 1, 3, 5]

    # Entropy
    assert np.allclose(
        nk.entropy_fuzzy(signal)[0],
        nk.entropy_sample(signal, fuzzy=True)[0],
        atol=0.000001,
    )

    # Fractal
    fractal_dfa, parameters = nk.fractal_dfa(signal, scale=np.array([4, 8, 12, 20]))
    assert parameters["Fluctuations"].shape == (4, 1)
    # Regression value (updated when the last overlapping window started being included)
    assert np.allclose(fractal_dfa, 2.10277697, atol=0.0001)

    _, parameters = nk.fractal_dfa(signal, multifractal=True, q=mdfa_q)
    # Default scales: from 10 to 100 samples, about 10% apart
    assert parameters["Fluctuations"].shape == (25, len(mdfa_q))

    assert np.allclose(nk.fractal_correlation(signal)[0], 0.7930504156910694, atol=0.000001)
    assert np.allclose(nk.fractal_correlation(signal, radius="nolds")[0], 0.8356790165119761, atol=0.000001)


def test_fractal_correlation_known_values():
    # Two repeated points have dimension 0
    alternating = np.tile([0.0, 1.0], 250)
    assert np.isclose(nk.fractal_correlation(alternating, radius="nolds")[0], 0, atol=1e-10)

    # A straight line has dimension 1
    ramp = np.arange(2000.0)
    assert np.isclose(nk.fractal_correlation(ramp, radius="nolds")[0], 1, atol=0.05)

    # Uniform noise fills the embedding space
    noise = np.random.default_rng(0).uniform(size=3000)
    for dimension in (1, 2, 3):
        cd = nk.fractal_correlation(noise, dimension=dimension, radius="nolds")[0]
        assert np.isclose(cd, dimension, atol=0.15)

    # Scaling, shifting and flipping don't change the dimension
    sine = np.sin(np.linspace(0, 10 * np.pi, 2000, endpoint=False))
    expected = nk.fractal_correlation(sine, radius="nolds")[0]
    for transformed in (sine * 1000, sine + 50, -sine):
        assert np.isclose(nk.fractal_correlation(transformed, radius="nolds")[0], expected)


def test_fractal_correlation_default_radius_scale_invariance():
    # The correlation dimension must not depend on the units of the signal
    signal = np.sin(np.linspace(0, 10 * np.pi, 2000, endpoint=False))
    expected = nk.fractal_correlation(signal)[0]
    assert np.isclose(nk.fractal_correlation(signal * 1000)[0], expected)


def test_fractal_correlation_constant_signal():
    # Dimension of zero check
    for radius in (64, "nolds", "Corr_Dim", "boon2008", [0.5, 1, 2]):
        assert nk.fractal_correlation(np.ones(100), radius=radius)[0] == 0


def test_fractal_correlation_radius_list():
    signal = np.sin(np.linspace(0, 10 * np.pi, 500, endpoint=False))
    expected, info = nk.fractal_correlation(signal, radius=16)
    assert np.isclose(nk.fractal_correlation(signal, radius=list(info["Radius"]))[0], expected)


def test_fractal_correlation_theiler():
    import scipy.spatial

    def cd_reference(x, delay, dimension, radius, theiler=0):
        # Grassberger & Procaccia (1983), leaving out the pairs closer than `theiler` samples in
        # time (Theiler, 1986)
        emb = nk.complexity_embedding(x, delay=delay, dimension=dimension)
        d = scipy.spatial.distance.pdist(emb)
        i, j = np.triu_indices(len(emb), 1)
        d = d[(j - i) > theiler]
        c = np.array([np.mean(d < r) for r in radius])
        return np.polyfit(np.log2(radius), np.log2(c), 1)[0]

    # Smoothed noise (oversampled, as EEG)
    rng = np.random.default_rng(42)
    x = np.convolve(rng.normal(size=1009), np.ones(10) / 10, mode="valid")
    cd, info = nk.fractal_correlation(x, delay=7, dimension=3, radius=16)
    assert np.allclose(cd, cd_reference(x, 7, 3, info["Radius"]))
    for theiler in [7, 28]:
        cd_w, info_w = nk.fractal_correlation(x, delay=7, dimension=3, radius=16, theiler=theiler)
        # The radii do not depend on the Theiler window
        assert np.allclose(info_w["Radius"], info["Radius"])
        assert np.allclose(cd_w, cd_reference(x, 7, 3, info["Radius"], theiler=theiler))
    # The window removes the time-close pairs, which raises the estimate of an oversampled signal
    assert nk.fractal_correlation(x, delay=7, dimension=3, radius=16, theiler=28)[0] > cd


def test_expspace_range():
    # Must span from start to stop, including when start is 0
    for base in (1, 2):
        assert np.allclose(nk.expspace(10, 100, 3, out=float, base=base)[[0, -1]], [10, 100])
        assert np.allclose(nk.expspace(0, 1, 3, out=float, base=base)[[0, -1]], [0, 1])
        # Spaced evenly on a log scale, whatever the units
        assert np.allclose(nk.expspace(1e-3, 1e-1, 3, out=float, base=base), [1e-3, 1e-2, 1e-1])


def test_fractal_dfa_scale_invariance():
    # Regression test for issue #1208: DFA's scaling exponents are invariant
    # under constant rescaling of the signal. White noise is monofractal with
    # h(q) = 0.5 for every q, so rescaled copies of the same noise must give
    # the same answer regardless of the amplitude units.
    scale = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256])
    q = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    signal = np.random.default_rng(0).standard_normal(20000)

    reference = None
    for multiplier in (1e0, 1e-2, 1e-4, 1e-5, 1e-10):
        _, info = nk.fractal_dfa(signal * multiplier, scale=scale, q=q, multifractal=True, show=False)
        h = np.asarray(info["h"], float)
        assert np.all(np.isfinite(h))
        if reference is None:
            reference = h
        else:
            # The fix makes h exactly invariant under rescaling (up to float
            # rounding), so pin it down tightly: a loose atol would let smaller
            # versions of the same bug through (see #1208).
            np.testing.assert_allclose(h, reference, rtol=1e-8)


# =============================================================================
# Comparison against R
# =============================================================================


# R code:
#
# library(TSEntropies)
# library(pracma)
#
# signal <- read.csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/bio_eventrelated_100hz.csv")$RSP
# r <- 0.2 * sd(signal)

# ApEn --------------------------------------------------------------------

# TSEntropies::ApEn(signal, dim=2, lag=1, r=r)
# 0.04383386
# TSEntropies::ApEn(signal, dim=3, lag=2, r=1)
# 0.0004269369
# pracma::approx_entropy(signal[1:200], edim=2, r=r, elag=1)
# 0.03632554

# SampEn ------------------------------------------------------------------

# TSEntropies::SampEn(signal[1:300], dim=2, lag=1, r=r)
# 0.04777648
# TSEntropies::FastSampEn(signal[1:300], dim=2, lag=1, r=r)
# 0.003490405
# pracma::sample_entropy(signal[1:300], edim=2, tau=1, r=r)
# 0.03784376
# pracma::sample_entropy(signal[1:300], edim=3, tau=2, r=r)
# 0.09185509


def test_complexity_vs_R():
    signal = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/bio_eventrelated_100hz.csv")[
        "RSP"
    ].values
    r = 0.2 * np.std(signal, ddof=1)

    # ApEn
    apen = nk.entropy_approximate(signal, dimension=2, tolerance=r)[0]
    assert np.allclose(apen, 0.04383386, atol=0.0001)
    apen = nk.entropy_approximate(signal, dimension=3, delay=2, tolerance=1)[0]
    assert np.allclose(apen, 0.0004269369, atol=0.0001)
    apen = nk.entropy_approximate(signal[0:200], dimension=2, delay=1, tolerance=r)[0]
    assert np.allclose(apen, 0.03632554, atol=0.0001)

    # SampEn
    sampen = nk.entropy_sample(signal[0:300], dimension=2, tolerance=r)[0]
    assert np.allclose(
        sampen,
        nk.entropy_sample(signal[0:300], dimension=2, tolerance=r, distance="infinity")[0],
        atol=0.001,
    )
    assert np.allclose(sampen, 0.03784376, atol=0.001)
    sampen = nk.entropy_sample(signal[0:300], dimension=3, delay=2, tolerance=r)[0]
    assert np.allclose(sampen, 0.09185509, atol=0.01)


# =============================================================================
# Comparison against Python implementations
# =============================================================================


def test_complexity_vs_Python():
    signal = np.cos(np.linspace(start=0, stop=30, num=100))
    tolerance = 0.2 * np.std(signal, ddof=1)

    # Shannon
    shannon = nk.entropy_shannon(signal)[0]
    #    assert scipy.stats.entropy(shannon, pd.Series(signal).value_counts())
    assert np.allclose(shannon, pyentrp.shannon_entropy(signal))

    # Approximate
    assert np.allclose(nk.entropy_approximate(signal)[0], 0.17364897858477146)

    # EntropyHub doens't work because of PyEMD
    # assert np.allclose(
    #     nk.entropy_approximate(signal, dimension=2, tolerance=tolerance)[0],
    #     EntropyHub.ApEn(signal, m=2, tau=1, r=tolerance)[0][2],
    # )
    assert np.allclose(
        nk.entropy_approximate(signal, dimension=2, tolerance=tolerance)[0],
        entropy_app_entropy(signal, 2),
    )

    assert nk.entropy_approximate(signal, dimension=2, tolerance=tolerance)[0] != pyeeg_ap_entropy(signal, 2, tolerance)

    # Sample
    assert np.allclose(
        nk.entropy_sample(signal, dimension=2, tolerance=tolerance)[0],
        entropy_sample_entropy(signal, 2),
    )
    assert np.allclose(
        nk.entropy_sample(signal, dimension=2, tolerance=0.2)[0],
        entro_py_sampen(signal, 2, 0.2, scale=False),
    )
    assert np.allclose(
        nk.entropy_sample(signal, dimension=2, tolerance=0.2)[0],
        pyeeg_samp_entropy(signal, 2, 0.2),
    )
    # assert np.allclose(
    #     nk.entropy_sample(signal, dimension=2, tolerance=0.2)[0],
    #     EntropyHub.SampEn(signal, m=2, tau=1, r=0.2)[0][2],
    # )

    #    import sampen
    #    sampen.sampen2(signal[0:300], mm=2, r=r)

    assert nk.entropy_sample(signal, dimension=2, tolerance=0.2)[0] != pyentrp.sample_entropy(signal, 2, 0.2)[1]
    assert (
        nk.entropy_sample(signal, dimension=2, tolerance=0.2 * np.sqrt(np.var(signal)))[0]
        != MultiscaleEntropy_sample_entropy(signal, 2, 0.2)[0.2][2]
    )

    # MSE
    #    assert nk.entropy_multiscale(signal, 2, 0.2*np.sqrt(np.var(signal)))
    #           != np.trapz(MultiscaleEntropy_mse(signal, [i+1 for i in range(10)], 2, 0.2, return_type="list"))
    #    assert nk.entropy_multiscale(signal, 2, 0.2*np.std(signal, ddof=1))
    #           != np.trapz(pyentrp.multiscale_entropy(signal, 2, 0.2, 10))

    # Fuzzy
    assert np.allclose(
        nk.entropy_fuzzy(signal, dimension=2, tolerance=0.2, delay=1)[0] - entro_py_fuzzyen(signal, 2, 0.2, 1, scale=False),
        0,
    )

    # Lempel Ziv Complexity
    threshold = np.nanmedian(signal)
    binary = np.zeros(len(signal))
    binary[signal > threshold] = 1
    assert np.allclose(
        nk.complexity_lempelziv(signal, symbolize="median", normalize=True)[0]
        - antropy.lziv_complexity(binary, normalize=True),
        0,
    )

    # Katz
    assert np.allclose(nk.fractal_katz(signal)[0] - antropy.katz_fd(signal), 0)


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
# entropy_estimators (https://github.com/paulbrodersen/entropy_estimators)
# =============================================================================

# import numpy as np
# from entropy_estimators import continuous
#
# x = np.random.randn(10000)
#
# # I don't know what this compute though
# continuous.get_h_mvn(x)
# continuous.get_h(x, k=5)


# =============================================================================
# Pyeeg
# =============================================================================


def pyeeg_embed_seq(time_series, tau, embedding_dimension):
    if not isinstance(time_series, np.ndarray):
        typed_time_series = np.asarray(time_series)
    else:
        typed_time_series = time_series

    shape = (
        typed_time_series.size - tau * (embedding_dimension - 1),
        embedding_dimension,
    )

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

    # Probability that random M-sequences are in range
    Cm = InRange.sum(axis=0)
    Dp = np.abs(np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T)

    Cmp = np.logical_and(Dp <= R, InRange).sum(axis=0)

    # Avoid taking log(0)
    Samp_En = np.log(np.sum(Cm + 1e-100) / np.sum(Cmp + 1e-100))

    return Samp_En


# =============================================================================
# Entropy
# =============================================================================


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
    phi = np.zeros(2)
    r = 0.2 * np.std(x, axis=-1, ddof=1)

    # compute phi(order, r)
    _emb_data1 = entropy_embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = sklearn.neighbors.KDTree(emb_data1, metric=metric).query_radius(emb_data1, r, count_only=True).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = entropy_embed(x, order + 1, 1)
    count2 = sklearn.neighbors.KDTree(emb_data2, metric=metric).query_radius(emb_data2, r, count_only=True).astype(np.float64)
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
    fuzzy = remove_baseline
    cross = isinstance(x, list)
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
        # https://github.com/ixjlyons/entro-py/pull/2/files
        phi[j] = np.mean(count) / (N - dim - 1)

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
        [dict or list as return_type indicates] -- [if dict, nest as [scale_factor][m][r] for each value of m, r;
                                                    if list, nest as [i][j] for lengths of m, r]

    """
    # type checking
    if not safe_mode:
        m = MultiscaleEntropy_check_type(m, int, "m")
        r = MultiscaleEntropy_check_type(r, float, "r")
        if sd is not None and not isinstance(sd, (float, int)):
            raise ValueError("sd should be a number")
    try:
        x = np.array(x)
    except Exception as exc:
        raise ValueError("x should be a sequence of numbers") from exc
    # value checking
    if len(x) < max(m):
        raise ValueError("the max m is bigger than x's length")

    # initialization
    if sd is None:
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
                tmp = -np.log(1 / (t * (t - 1)))
            else:
                tmp = -np.log(count[mm + 1] / count[mm])
            if return_type == "dict":
                ans[rr][mm] = tmp
            else:
                ans[i].append(tmp)
    return ans


def MultiscaleEntropy_mse(
    x,
    scale_factor=list(range(1, 21)),
    m=[2],
    r=[0.15],
    return_type="dict",
    safe_mode=False,
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
        [dict or list as return_type indicates] -- [if dict, nest as [scale_factor][m][r] for each value of scale_factor, m, r;
                                                    if list nest as [i][j][k] for lengths of scale_factor, m, r]

    """
    # type checking
    if not safe_mode:
        m = MultiscaleEntropy_check_type(m, int, "m")
        r = MultiscaleEntropy_check_type(r, float, "r")
        scale_factor = MultiscaleEntropy_check_type(scale_factor, int, "scale_factor")
    try:
        x = np.array(x)
    except Exception as exc:
        raise ValueError("x should be a sequence of numbers") from exc
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


def test_entropy_approximate_sign():
    # Pincus (1991): ApEn = Phi(m) - Phi(m+1), signed. When no embedded vector has a neighbour
    # other than itself, Phi(m) = -log(N - (m - 1) * delay) and Phi(m+1) = -log(N - m * delay),
    # so ApEn = log((N - m * delay) / (N - (m - 1) * delay)) < 0 (self-match bias).
    rng = np.random.default_rng(42)
    signal = rng.normal(size=500)
    n, m = len(signal), 5
    apen = nk.entropy_approximate(signal, delay=1, dimension=m, tolerance=0.1)[0]
    assert apen < 0
    assert np.allclose(apen, np.log((n - m) / (n - m + 1)))

    # Non-degenerate case stays positive and matches the reference implementation
    apen = nk.entropy_approximate(signal, delay=1, dimension=2, tolerance=0.2)[0]
    assert apen > 0


def test_entropy_sample_delay():
    # Richman & Moorman (2000): the same N - m * delay templates at m and m + 1, whatever the delay
    rng = np.random.default_rng(42)
    signal = np.convolve(rng.normal(size=1009), np.ones(10) / 10, mode="valid")
    signal = (signal - signal.mean()) / signal.std(ddof=1)

    def sampen_reference(x, delay, dimension, tolerance):
        emb = nk.complexity_embedding(x, delay=delay, dimension=dimension + 1)

        def n_pairs(vectors):
            d = np.max(np.abs(vectors[:, None, :] - vectors[None, :, :]), axis=2)
            return (np.sum(d <= tolerance) - len(vectors)) / 2

        return -np.log(n_pairs(emb) / n_pairs(emb[:, :-1]))

    for delay in [1, 7]:
        assert np.allclose(
            nk.entropy_sample(signal, delay=delay, dimension=5, tolerance=0.7)[0],
            sampen_reference(signal, delay, 5, 0.7),
        )


def test_complexity_lempelziv_count():
    from neurokit2.complexity.complexity_lempelziv import _complexity_lempelziv_count

    def lz76_words(symbols):
        # LZ76 parsing (Lempel & Ziv, 1976) from its definition: each word is the shortest part of
        # the rest of the sequence that cannot be copied from what precedes its last symbol
        s = list(symbols)
        n, c, p = len(s), 0, 0

        def copyable(word, end):
            return any(s[q : q + len(word)] == word for q in range(end - len(word) + 1))

        while p < n:
            k = 1
            while p + k <= n and copyable(s[p : p + k], p + k - 1):
                k += 1
            c += 1
            p += k
        return c

    # Published parsings (Kaspar & Schuster, 1987)
    for string in ["0001101001000101", "1001111011000010"]:
        assert _complexity_lempelziv_count(np.array(list(string)).astype(int)) == (6, 16)

    # Binary sequences
    rng = np.random.default_rng(42)
    for length in [1, 2, 3, 10, 50, 200]:
        symbols = (rng.random(length) < 0.5).astype(int)
        assert _complexity_lempelziv_count(symbols) == (lz76_words(symbols), length)

    # Permutation LZC with more than 10 possible patterns (dimension >= 4)
    signal = nk.signal_simulate(duration=4, sampling_rate=250, frequency=[5, 10], noise=0.5, random_state=42)
    for dimension in [3, 4, 5]:
        _, info = nk.complexity_lempelziv(signal, dimension=dimension, permutation=True)
        patterns = nk.complexity_ordinalpatterns(signal, dimension=dimension)[1]["Uniques"]
        assert info["Complexity_Kolmogorov"] == lz76_words(patterns)
        assert info["Permutation"] is True and info["Dimension"] == dimension and info["Delay"] == 1

    # One sample
    assert nk.complexity_lempelziv([1.0])[0] == 0


def test_fractal_petrosian():
    def pfd_reference(n, n_delta):
        # Petrosian (1995): N is the number of samples, N_delta the number of sign changes
        return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))

    signal = nk.signal_simulate(duration=2, sampling_rate=250, frequency=[5, 12], noise=0.5, random_state=42)
    d = np.diff(signal)
    assert np.allclose(nk.fractal_petrosian(signal)[0], pfd_reference(len(signal), np.sum(d[1:] * d[:-1] < 0)))

    # Flat segments are not sign changes: monotonic staircases and a constant signal give 1
    up = np.repeat(np.arange(10.0), 2)
    assert np.allclose(nk.fractal_petrosian(up)[0], 1)
    assert np.allclose(nk.fractal_petrosian(up[::-1].copy())[0], 1)
    assert np.allclose(nk.fractal_petrosian([3, 2, 2, 1])[0], 1)
    assert np.allclose(nk.fractal_petrosian(np.ones(20))[0], 1)

    # A plateau at an extremum is one sign change
    assert np.array_equal(nk.complexity_symbolize([0, 1, 1, 0], method="C"), [0, 0, 1])
    assert np.allclose(nk.fractal_petrosian([0, 1, 1, 0])[0], pfd_reference(4, 1))

    # Without zero differences, method "C" is the sign of the differences
    rng = np.random.default_rng(42)
    x = rng.normal(size=1000)
    assert np.array_equal(nk.complexity_symbolize(x, method="C"), np.signbit(np.diff(x)).astype(int))


def test_fractal_dfa_windows():
    def dfa_reference(signal, scale, overlap=True, order=1):
        # Peng et al. (1994), with every window that fits in the signal
        profile = np.cumsum(signal - np.mean(signal))
        n = len(profile)
        fluctuations = []
        for window in scale:
            x = np.arange(window)
            variances = []
            for start in range(0, n - window + 1, window // 2 if overlap else window):
                segment = profile[start : start + window]
                variances.append(np.var(segment - np.polyval(np.polyfit(x, segment, order), x)))
            fluctuations.append(np.sqrt(np.mean(variances)))
        return np.polyfit(np.log2(scale), np.log2(fluctuations), 1)[0]

    rng = np.random.default_rng(42)
    for signal in [rng.normal(size=2000), np.cumsum(rng.normal(size=1537))]:
        scale = np.array([10, 14, 20, 28, 40, 50])
        for overlap in [True, False]:
            for order in [1, 2]:
                assert np.allclose(
                    nk.fractal_dfa(signal, scale=scale, overlap=overlap, order=order)[0],
                    dfa_reference(signal, scale, overlap=overlap, order=order),
                )

    # Known exponents: 0.5 for white noise, 1.5 for a Brownian motion
    noise = rng.normal(size=10000)
    assert np.allclose(nk.fractal_dfa(noise)[0], 0.5, atol=0.1)
    assert np.allclose(nk.fractal_dfa(np.cumsum(noise))[0], 1.5, atol=0.1)


def test_fractal_mfdfa():
    def mfdfa_reference(signal, scale, q, overlap=True):
        # Kantelhardt et al. (2002), with the logarithmic average for q = 0
        profile = np.cumsum(signal - np.mean(signal))
        fluctuations = np.zeros((len(scale), len(q)))
        for i, window in enumerate(scale):
            x = np.arange(window)
            f2 = []
            for start in range(0, len(profile) - window + 1, window // 2 if overlap else window):
                segment = profile[start : start + window]
                f2.append(np.mean((segment - np.polyval(np.polyfit(x, segment, 1), x)) ** 2))
            f2 = np.array(f2)
            for j, qj in enumerate(q):
                if qj == 0:
                    fluctuations[i, j] = np.exp(0.5 * np.mean(np.log(f2)))
                else:
                    fluctuations[i, j] = np.mean(f2 ** (qj / 2)) ** (1 / qj)
        return fluctuations

    rng = np.random.default_rng(3)
    q = np.array([-5.0, -3, -1, 0, 1, 2, 3, 5])
    scale = np.array([10, 16, 25, 40, 63, 100])
    for signal in [rng.normal(size=2000), np.cumsum(rng.normal(size=2000))]:
        for overlap in [True, False]:
            _, info = nk.fractal_dfa(signal, scale=scale, q=q, multifractal=True, overlap=overlap)
            np.testing.assert_allclose(info["Fluctuations"], mfdfa_reference(signal, scale, q, overlap), rtol=1e-9)

    # Binomial multifractal cascade (Kantelhardt et al., 2002), whose h(q) is known. Its smallest
    # segment variances are 1e-12 times the largest: they must not be discarded, as they set the
    # negative q exponents (which were about 1.2 instead of 1.6-1.8)
    a, nmax = 0.75, 16
    ones = np.array([bin(k).count("1") for k in range(2**nmax)])
    cascade = a**ones * (1 - a) ** (nmax - ones)
    q = np.array([-5.0, -3, -1])
    h_theory = 1 / q - np.log(a**q + (1 - a) ** q) / (q * np.log(2))
    scale = np.unique(np.round(2 ** np.linspace(4, 12, 25)).astype(int))
    _, info = nk.fractal_dfa(cascade, scale=scale, q=q, multifractal=True, order=2)
    np.testing.assert_allclose(info["h"], h_theory, atol=0.1)

    # Several q values close to 0, and unsorted q
    signal = rng.normal(size=3000)
    _, info = nk.fractal_dfa(signal, q=[-1, -0.05, 0, 0.05, 1], multifractal=True)
    assert np.all(info["Fluctuations"][:, 1] == info["Fluctuations"][:, 2])
    assert np.all(info["Fluctuations"][:, 2] == info["Fluctuations"][:, 3])
    out1, _ = nk.fractal_dfa(signal, q=[-3, -1, 1, 3], multifractal=True)
    out2, _ = nk.fractal_dfa(signal, q=[1, -3, 3, -1], multifractal=True)
    pd.testing.assert_frame_equal(out1, out2)

    # Exactly linear segments of the profile (runs of identical values) are left out
    signal[1000:1200] = 0
    _, info = nk.fractal_dfa(signal, scale=np.array([10, 20, 40, 80]), q=[-5, 2], multifractal=True)
    assert np.allclose(info["h"], 0.5, atol=0.15)

    # Spectrum features of a symmetric spectrum (H linear in q)
    from neurokit2.complexity.fractal_dfa import _singularity_spectrum

    q = np.arange(-5.0, 6)
    out = _singularity_spectrum(q=q, slopes=0.7 - 0.02 * q)
    assert np.isclose(out["Peak"], 0.7)
    assert np.isclose(out["Asymmetry"], 0.5)
    # Faini et al. (2021): sum of the squared increments of h between consecutive q
    assert np.isclose(out["Increment"], 10 * 0.02**2)
    # Orozco-Duque et al. (2015, eq. 17): squared second differences of h, over 2 * |q_max| + 2
    out = _singularity_spectrum(q=q, slopes=0.7 - 0.01 * q**2)
    assert np.isclose(out["Fluctuation"], 9 * 0.02**2 / 12)

    # Default scales: evenly log-spaced from 10 samples to a tenth of the signal, about 10% apart
    from neurokit2.complexity.fractal_dfa import _fractal_dfa_findscales

    scale = _fractal_dfa_findscales(10000)
    assert len(scale) == 49
    assert np.all(np.abs(scale / np.geomspace(10, 1000, 49) - 1) <= 0.05)

    # max. DFA: the scales up to the knee are kept, and returned
    signal = nk.signal_simulate(duration=20, frequency=[5, 7, 10, 14], noise=0.05, random_state=1)
    scale = nk.expspace(10, 400, 30, base=2).astype(int)
    _, info = nk.fractal_dfa(signal, scale=scale)
    knee = nk.find_knee(np.log2(info["Fluctuations"][:, 0]), x=np.log2(scale), verbose=False)
    _, info = nk.fractal_dfa(signal, scale=scale, maxdfa=True)
    assert np.isclose(np.log2(info["scale"][-1]), knee)
    assert len(info["scale"]) == len(info["Fluctuations"])


def test_complexity_delay_methods():
    # Sine of period 42 samples: its autocorrelation is cos(2 * pi * tau / 42)
    period = 42
    signal = np.sin(2 * np.pi * np.arange(4000) / period)
    autocor = nk.signal_autocor(signal)[0]

    # First lag at or below 0 (10.5 -> 11), 1/e (7.98 -> 8) and 1 - 1/e (5.92 -> 6)
    for method, expected in [("casdagli1991", 11), ("theiler1990", 8), ("rosenstein1993", 6)]:
        delay, info = nk.complexity_delay(signal, delay_max=50, method=method)
        assert delay == expected
        # The scores are the autocorrelation at the tested lags (they used to start at lag 0)
        assert np.allclose(info["Scores"], autocor[1:51])

    # A custom sequence of delays is honoured
    delay, info = nk.complexity_delay(signal, delay_max=np.arange(5, 30), method="casdagli1991")
    assert delay == 11
    assert np.allclose(info["Scores"], autocor[5:30])

    # Rosenstein (1994): the slope of the average displacement at 40% of its initial value,
    # i.e., cos(pi * tau / period) = 0.4 -> tau = 15.5
    delay, info = nk.complexity_delay(signal, delay_max=50, method="rosenstein1994")
    slope = np.diff(np.concatenate([[0], info["Scores"]]))
    assert delay == np.where(slope <= 0.4 * slope[0])[0][0] + 1
    assert 14 <= delay <= 18

    # Mutual information: first local minimum at a quarter of the period, also for a noise-free
    # signal with delay_max beyond the period (the number of bins used to diverge there)
    assert nk.complexity_delay(signal, delay_max=100, method="fraser1986")[0] in [10, 11]

    # SPAR: a third of the period
    assert nk.complexity_delay(signal, method="spar")[0] == 14

    # C-C method (Kim, 1999): usable, returns the first local minimum of its curve
    lorenz = nk.complexity_simulate(duration=4, sampling_rate=250, method="lorenz")
    delay, info = nk.complexity_delay(lorenz, delay_max=30, method="kim1999")
    i = np.where(info["Values"] == delay)[0][0]
    assert 1 <= delay <= 30
    assert info["Scores"][i] <= info["Scores"][i - 1] and info["Scores"][i] <= info["Scores"][i + 1]


def test_complexity_dimension_methods():
    # A noisy sine (a circle): 2 dimensions
    rng = np.random.default_rng(3)
    sine = np.sin(2 * np.pi * np.arange(2000) / 41.3) + 1e-4 * rng.normal(size=2000)
    dimension, info = nk.complexity_dimension(sine, delay=10, dimension_max=6, method="fnn")
    assert dimension == 2
    assert np.all(info["f3"][1:] < 0.02)

    # Quantized signals (e.g., RR intervals in integer milliseconds) are handled: identical
    # vectors are skipped rather than counted as true neighbours (fnn) or making everything NaN
    # (afnn), and the result is one of the tested dimensions
    rr = np.round(60000 / (70 + 5 * np.cumsum(rng.normal(size=600)) / 10))
    for method in ["fnn", "afnn"]:
        dimension, info = nk.complexity_dimension(rr, delay=1, dimension_max=8, method=method)
        assert dimension in info["Values"]
    # A signal whose vectors are all identical to others gives NaN with a warning
    for method in ["fnn", "afnn"]:
        with pytest.warns(nk.misc.NeuroKitWarning):
            dimension, _ = nk.complexity_dimension(np.tile([0.0, 1.0, 2.0], 100), delay=1, dimension_max=3, method=method)
        assert np.isnan(dimension)

    # afnn with a custom list of dimensions returns the dimension itself (not its index + 1)
    lorenz = nk.complexity_simulate(duration=4, sampling_rate=250, method="lorenz")
    full, info = nk.complexity_dimension(lorenz, delay=5, dimension_max=8, method="afnn")
    custom, _ = nk.complexity_dimension(lorenz, delay=5, dimension_max=[2, 3, 4, 5, 6, 7, 8], method="afnn")
    assert full == custom
    assert full == info["Values"][np.argmax(info["E1"] >= 0.85 * np.nanmax(info["E1"]))]


def test_complexity_tolerance_methods():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=1000)
    sd1, sd2 = np.std(np.diff(signal), ddof=1), np.std(signal, ddof=1)

    # Chon (2009): r in units of the SD, divided by the fourth root of N / 1000, with the
    # published coefficients for m = 2 to 7
    for dimension, x, y in [(2, -0.036, 0.26), (3, -0.08, 0.46), (7, -0.2, 1.0)]:
        expected = sd2 * (x + y * np.sqrt(sd1 / sd2)) / (len(signal) / 1000) ** 0.25
        assert np.allclose(nk.complexity_tolerance(signal, method="chon2009", dimension=dimension)[0], expected)
    # Scales with the signal
    r1 = nk.complexity_tolerance(signal, method="chon2009", dimension=2)[0]
    r100 = nk.complexity_tolerance(100 * signal, method="chon2009", dimension=2)[0]
    assert np.allclose(r100, 100 * r1)
    # And with the length (N = 4000 -> divided by sqrt(2))
    r4000 = nk.complexity_tolerance(np.tile(signal, 4), method="chon2009", dimension=2)[0]
    assert np.allclose(r4000, r1 / np.sqrt(2), rtol=0.05)

    # numpy integers (as returned by complexity_dimension()) are accepted
    for method in ["nolds", "chon2009", "makowski"]:
        assert np.allclose(
            nk.complexity_tolerance(signal, method=method, dimension=np.int64(3))[0],
            nk.complexity_tolerance(signal, method=method, dimension=3)[0],
        )

    # Plotting does not modify the returned scores
    for method in ["recurrence", "neighbours"]:
        _, info = nk.complexity_tolerance(signal[:300], method=method, delay=1, dimension=2, show=True)
        assert np.max(info["Scores"]) <= 1
    for method in ["sd", "nolds", "chon2009", "makowski", "singh2016"]:
        nk.complexity_tolerance(signal[:300], method=method, delay=1, dimension=2, show=True)
    plt.close("all")


def test_complexity_rqa():
    import scipy.stats

    from neurokit2.complexity.complexity_rqa import _complexity_rqa_features

    def runs(sequence):
        lengths, run = [], 0
        for value in list(sequence) + [False]:
            if value:
                run += 1
            elif run > 0:
                lengths.append(run)
                run = 0
        return lengths

    def entropy(lengths):
        return scipy.stats.entropy(np.unique(lengths, return_counts=True)[1]) if len(lengths) > 0 else 0

    def rqa_reference(rc, lmin=2):
        # Marwan et al. (2007): whole matrix, line of identity (LOI) excluded
        rc = np.asarray(rc, dtype=bool)
        rec = rc.copy()
        np.fill_diagonal(rec, False)
        n = len(rc)
        diag = np.array([x for k in range(1 - n, n) if k != 0 for x in runs(np.diagonal(rec, k))], dtype=int)
        vert = np.array([x for column in rec.T for x in runs(column)], dtype=int)
        # White lines: between two consecutive recurrence points of a column (the LOI included)
        white = np.concatenate([np.diff(np.flatnonzero(column)) - 1 for column in rc.T]).astype(int)
        diag, vert, white = diag[diag >= lmin], vert[vert >= lmin], white[white >= lmin]
        out = {
            "RecurrenceRate": rec.sum() / (n**2 - n),
            "Determinism": diag.sum() / rec.sum(),
            "L": diag.mean() if len(diag) > 0 else 0,
            "Divergence": 1 / diag.max() if len(diag) > 0 else np.nan,
            "LEn": entropy(diag),
            "Laminarity": vert.sum() / rec.sum(),
            "TrappingTime": vert.mean() if len(vert) > 0 else 0,
            "VMax": vert.max() if len(vert) > 0 else 0,
            "VEn": entropy(vert),
            "W": white.mean() if len(white) > 0 else 0,
            "WMax": white.max() if len(white) > 0 else 0,
            "WEn": entropy(white),
        }
        out["LamiDet"] = out["Laminarity"] / out["Determinism"] if out["Determinism"] > 0 else np.nan
        # Tomashin et al. (2022, approach 3): log-log slope over the first 64 diagonals
        lag = np.arange(1, min(n, 65))
        rate = np.array([np.diagonal(rc, k).mean() for k in lag])
        lag, rate = lag[rate > 0], rate[rate > 0]
        out["DiagRec"] = np.polyfit(np.log2(lag), np.log2(rate), 1)[0] if len(lag) > 2 else np.nan
        return out

    rng = np.random.default_rng(42)
    matrices = []
    for p in [0.05, 0.3, 0.7]:
        rc = np.triu(rng.uniform(size=(60, 60)) < p, k=1)
        matrices.append(rc | rc.T | np.eye(60, dtype=bool))
    # Time-irreversible signals (Hénon map, sawtooth), where the vertical lines below and above the
    # LOI differ
    henon = np.zeros(300)
    henon[0], y = 0.1, 0.1
    for i in range(1, 300):
        henon[i], y = 1 - 1.4 * henon[i - 1] ** 2 + y, 0.3 * henon[i - 1]
    sawtooth = (np.arange(300) % 20) / 20 + rng.normal(0, 0.02, 300)
    for signal in [henon, sawtooth, rng.normal(size=300)]:
        matrices.append(nk.recurrence_matrix(signal, dimension=2, tolerance=0.15 * np.ptp(signal))[0])

    for rc in matrices:
        for lmin in [1, 2, 3]:
            out = _complexity_rqa_features(rc, min_linelength=lmin)
            for key, value in rqa_reference(rc, lmin=lmin).items():
                assert np.isclose(out[key], value, equal_nan=True), (key, lmin)

    # White lines that reach the border of the matrix are left out: here, only the two lines
    # between the LOI and the recurrences (0, 5) and (5, 0) have a recurrence point at both ends
    rc = np.eye(6, dtype=bool)
    rc[0, 5] = rc[5, 0] = True
    out = _complexity_rqa_features(rc)
    assert out["W"] == 4 and out["WMax"] == 4 and out["WEn"] == 0

    # Same features with complexity_rqa(), and LamiDet is returned
    signal = nk.signal_simulate(duration=4, sampling_rate=100, frequency=[5, 6, 7], noise=0.2, random_state=3)
    rqa, info = nk.complexity_rqa(signal, tolerance=1)
    assert rqa.shape == (1, 15)
    assert np.isclose(rqa["LamiDet"][0], rqa["Laminarity"][0] / rqa["Determinism"][0])
    for key, value in rqa_reference(info["Recurrence_Matrix"]).items():
        assert np.isclose(rqa[key][0], value), key
    # No recurrences: the ratios are undefined
    rqa, _ = nk.complexity_rqa(signal, tolerance=0)
    assert rqa["RecurrenceRate"][0] == 0
    assert rqa[["Determinism", "Laminarity", "LamiDet", "DeteRec"]].isna().all(axis=None)

    # The default tolerance gives a recurrence rate of 1% (in any dimension), whereas 0.2 SD,
    # designed for the Chebyshev distance of SampEn, leaves almost no recurrences
    for dimension in [3, 7]:
        rqa, info = nk.complexity_rqa(signal, dimension=dimension)
        assert info["Tolerance"] == nk.complexity_tolerance(signal, "recurrence", delay=1, dimension=dimension)[0]
        assert 0.01 <= rqa["RecurrenceRate"][0] < 0.0101
        assert nk.complexity_rqa(signal, dimension=dimension, tolerance="sd")[0]["RecurrenceRate"][0] < 0.002


def test_complexity_rqa_distance():
    import scipy.spatial

    signal = nk.signal_simulate(duration=4, sampling_rate=100, frequency=[6, 10, 21], noise=0.5, random_state=2)
    signal = nk.standardize(signal)
    m, r = 3, 0.5
    embedded = nk.complexity_embedding(signal, delay=1, dimension=m)
    for distance, metric in [("euclidean", "euclidean"), ("chebyshev", "chebyshev"), ("manhattan", "cityblock")]:
        rc, d = nk.recurrence_matrix(signal, dimension=m, tolerance=r, distance=distance)
        np.testing.assert_allclose(d, scipy.spatial.distance.cdist(embedded, embedded, metric))
        assert np.array_equal(rc, d <= r)
        # The default tolerance gives 1% of recurrences with each distance
        rqa, info = nk.complexity_rqa(signal, dimension=m, distance=distance)
        assert info["Distance"] == distance
        assert info["Tolerance"] == nk.complexity_tolerance(signal, "recurrence", delay=1, dimension=m, distance=distance)[0]
        assert 0.01 <= rqa["RecurrenceRate"][0] < 0.0101

    # With the Chebyshev distance (and a delay of 1), the recurrence matrix in dimension m is the one
    # of dimension 1 restricted to its diagonal lines of at least m points...
    rc1 = nk.recurrence_matrix(signal, dimension=1, tolerance=r, distance="chebyshev")[0].astype(bool)
    rc = nk.recurrence_matrix(signal, dimension=m, tolerance=r, distance="chebyshev")[0].astype(bool)
    n = len(rc)
    expected = np.ones((n, n), dtype=bool)
    for k in range(m):
        expected &= rc1[k : k + n, k : k + n]
    assert np.array_equal(rc, expected)
    # ... and SampEn is minus the log of the probability that a recurrence point is followed by
    # another one on its diagonal (among the first N - m points, the templates of SampEn)
    np.fill_diagonal(rc, False)
    successor = (rc[:-1, :-1] & rc[1:, 1:]).sum() / rc[:-1, :-1].sum()
    assert np.isclose(-np.log(successor), nk.entropy_sample(signal, dimension=m, tolerance=r)[0])

    with pytest.raises(ValueError, match="distance"):
        nk.complexity_rqa(signal, distance="cosine")


def test_complexity_tolerance_recurrence():
    import scipy.spatial

    rng = np.random.default_rng(7)
    for signal in [rng.normal(size=400), np.round(rng.normal(size=400).cumsum())]:
        d = scipy.spatial.distance.pdist(nk.complexity_embedding(signal, delay=2, dimension=4))
        r, info = nk.complexity_tolerance(signal, method="recurrence", delay=2, dimension=4)
        # The smallest distance that gives a recurrence rate of at least 1%
        assert np.mean(d <= r) >= 0.01
        assert np.mean(d < r) < 0.01
        # The curve is the recurrence rate at each tolerance of the range
        assert len(info["Values"]) == 50
        np.testing.assert_allclose(info["Scores"], [np.mean(d <= x) for x in info["Values"]])

        # With a custom range, the value whose recurrence rate is the closest to 1%
        r_range = np.quantile(d, [0.3, 0.002, 0.05, 0.012])
        r, info = nk.complexity_tolerance(signal, method="recurrence", delay=2, dimension=4, r_range=r_range)
        np.testing.assert_allclose(info["Scores"], [np.mean(d <= x) for x in r_range])
        assert r == r_range[3]


def test_entropy_spectral_sampling_rate():
    signal = nk.signal_simulate(duration=10, sampling_rate=100, frequency=[5, 30], noise=0)

    # The sampling rate is forwarded to signal_psd() instead of raising a TypeError
    spen, _ = nk.entropy_spectral(signal, sampling_rate=100)
    assert np.isclose(spen, nk.entropy_spectral(signal)[0])

    # With a restricted frequency range, the band is defined in Hz of the actual signal
    spen_10hz, _ = nk.entropy_spectral(signal, sampling_rate=100, max_frequency=10)
    psd = nk.signal_psd(signal, sampling_rate=100, max_frequency=10)
    power = psd["Power"] / psd["Power"].sum()
    assert np.isclose(spen_10hz, nk.entropy_shannon(freq=power.values)[0] / np.log2(len(power)))


def test_complexity_dimension_fnn():
    # The Lorenz attractor has 3 variables: the proportion of false nearest neighbours drops
    # to ~0 at dimension 3, which must be returned (and not its index in the sequence, 2)
    signal = nk.complexity_simulate(duration=2, sampling_rate=250, method="lorenz")
    dimension, info = nk.complexity_dimension(signal, delay=5, dimension_max=10, method="fnn")
    assert dimension == 3
    f3 = info["f3"]
    assert dimension == info["Values"][np.argmax(f3 <= max(0.02, 1.85 * np.min(f3)))]

    # A custom sequence of dimensions (not starting at 1) must return one of its members
    dimension, _ = nk.complexity_dimension(signal, delay=5, dimension_max=[2, 3, 4, 5, 6], method="fnn")
    assert dimension == 3

    # complexity_optimize() uses the same selection
    _, _, dimension = _complexity_dimension(signal, delay=5, dimension_max=10, method="fnn")
    assert dimension == 3

    # complexity_optimize(show=True) plots the fnn fractions against the full sequence of
    # dimensions (the afnn ratios are one shorter and are plotted against the trimmed one)
    out = nk.complexity_optimize(signal, delay_max=10, dimension_max=10, dimension_method="fnn", show=True)
    assert out["Dimension"] == 3
    nk.complexity_optimize(signal, delay_max=10, dimension_max=10, dimension_method="afnn", show=True)
    plt.close("all")


def test_fractal_higuchi():
    def hfd_reference(x, k_max):
        # Higuchi (1988): average normalized length L(k) of the k series x(m), x(m + k), ...
        n = len(x)
        lengths = []
        for k in range(1, k_max + 1):
            lm = []
            for m in range(1, k + 1):
                n_m = (n - m) // k
                total = sum(abs(x[m - 1 + i * k] - x[m - 1 + (i - 1) * k]) for i in range(1, n_m + 1))
                lm.append(total * (n - 1) / (n_m * k) / k)
            lengths.append(np.mean(lm))
        # log L(k) = intercept + HFD * log(1 / k)
        return np.polyfit(np.log(1 / np.arange(1, k_max + 1)), np.log(lengths), 1)

    rng = np.random.default_rng(42)
    signal = rng.normal(size=301)
    for k_max in [2, 5, 10, 150]:
        hfd, info = nk.fractal_higuchi(signal, k_max=k_max)
        slope, intercept = hfd_reference(signal, k_max)
        assert np.allclose(hfd, slope)
        assert np.allclose(info["Intercept"], intercept)

    # complexity_k() scores are the HFD of each k_max
    _, info = nk.complexity_k(signal, k_max=[2, 5, 10, 150])
    assert np.allclose(info["Scores"], [hfd_reference(signal, k)[0] for k in [2, 5, 10, 150]])

    # Known values: 2 for white noise, 1.5 for a Brownian motion, 1 for a sine
    noise = rng.normal(size=5000)
    assert np.allclose(nk.fractal_higuchi(noise, k_max=10)[0], 2, atol=0.05)
    assert np.allclose(nk.fractal_higuchi(np.cumsum(noise), k_max=10)[0], 1.5, atol=0.1)
    sine = np.sin(2 * np.pi * 5 * np.arange(5000) / 1000)
    assert np.allclose(nk.fractal_higuchi(sine, k_max=10)[0], 1, atol=0.05)

    # Lists, Series and numpy integers are accepted
    hfd = nk.fractal_higuchi(signal, k_max=10)[0]
    assert np.allclose(nk.fractal_higuchi(list(signal), k_max=np.int64(10))[0], hfd)
    assert np.allclose(nk.fractal_higuchi(pd.Series(signal, index=np.arange(100, 401)), k_max=10)[0], hfd)
    assert nk.complexity_k(list(signal), k_max=np.int64(30))[0] == nk.complexity_k(signal, k_max=30)[0]

    # k_max must be between 2 and half the length of the signal
    with pytest.warns(nk.misc.NeuroKitWarning):
        assert np.isnan(nk.fractal_higuchi(signal, k_max=151)[0])
    with pytest.raises(ValueError):
        nk.complexity_k(signal, k_max=[5, 151])
    with pytest.raises(ValueError):
        nk.complexity_k(signal, k_max=20.0)


def test_complexity_k_plateau():
    def is_plateau(hfd, i):
        # The HFD stays within 0.01 of its value for the next 10 tested k_max
        return len(hfd) > i + 10 and np.all(np.abs(hfd[i + 1 : i + 11] - hfd[i]) < 0.01)

    # The first k_max of the plateau is returned, and it does not drift with the length of the signal
    for n in [500, 2000]:
        for seed in range(3):
            signal = nk.signal_simulate(duration=n / 100, sampling_rate=100, frequency=[5, 6], noise=0.5, random_state=seed)
            k_max, info = nk.complexity_k(signal)
            i = list(info["Values"]).index(k_max)
            assert is_plateau(info["Scores"], i)
            assert not any(is_plateau(info["Scores"], j) for j in range(i))
            assert 15 <= k_max <= 25

    # A list of k_max is tested in increasing order
    k_values = np.arange(2, 60)
    assert nk.complexity_k(signal, k_max=k_values[::-1])[0] == nk.complexity_k(signal, k_max=k_values)[0]

    # Without 10 values after it, there is no plateau: the largest k_max is returned
    with pytest.warns(nk.misc.NeuroKitWarning, match="No plateau"):
        assert nk.complexity_k(signal, k_max=10)[0] == 10


def test_entropy_bubble():
    def bubble_reference(signal, dimension, alpha=2):
        # Manis et al. (2017): Renyi entropy of the number of swaps of bubble sort, at m and m + 1
        def renyi(m):
            swaps = []
            for embedded in nk.complexity_embedding(signal, dimension=m):
                vector, n_swaps = list(embedded), 0
                for i in range(m - 1):
                    for j in range(m - i - 1):
                        if vector[j] > vector[j + 1]:
                            vector[j], vector[j + 1] = vector[j + 1], vector[j]
                            n_swaps += 1
                swaps.append(n_swaps)
            p = np.unique(swaps, return_counts=True)[1] / len(swaps)
            return np.log(np.sum(p**alpha)) / (1 - alpha)

        return (renyi(dimension + 1) - renyi(dimension)) / np.log((dimension + 1) / (dimension - 1))

    rng = np.random.default_rng(42)
    signal = rng.normal(size=500)
    for x in [signal, np.round(signal)]:  # with and without ties
        for dimension in [2, 3, 6]:
            assert np.allclose(nk.entropy_bubble(x, dimension=dimension)[0], bubble_reference(x, dimension))
    assert np.allclose(nk.entropy_bubble(signal, alpha=3)[0], bubble_reference(signal, 3, alpha=3))

    # The embedded vectors are not sorted in place
    _, info = nk.complexity_ordinalpatterns(signal, algorithm="bubblesort")
    assert np.array_equal(info["Embedded"], nk.complexity_embedding(signal, dimension=3))

    with pytest.raises(ValueError):
        nk.entropy_bubble(signal, dimension=1)


def test_entropy_increment():
    # Worked example of Liu et al. (2016): 8 unique words with m = 2 and R = 4, i.e., 3 bits
    assert np.allclose(nk.entropy_increment([3, 3, 2, -8, -5, 4, 20, 10, 11, 8], dimension=2, q=4)[0], 3)

    def incren_reference(signal, dimension, q):
        # Each increment is mapped to two letters: its sign, and its size relative to the SD of its vector
        words = []
        for vector in nk.complexity_embedding(np.diff(signal), dimension=dimension):
            sd = np.std(vector, ddof=1)
            sizes = [0 if sd == 0 else min(q, np.floor(abs(v) * q / sd)) for v in vector]
            words.append(list(np.sign(vector)) + sizes)
        p = np.unique(words, axis=0, return_counts=True)[1] / len(words)
        return -np.sum(p * np.log2(p)) / (dimension - 1)

    rng = np.random.default_rng(42)
    signal = rng.normal(size=500)
    for dimension, q in [(2, 4), (3, 4), (3, 2), (4, 1)]:
        assert np.allclose(nk.entropy_increment(signal, dimension=dimension, q=q)[0], incren_reference(signal, dimension, q))

    # With q = 0, the words are the signs of the increments
    signs = nk.complexity_embedding(np.sign(np.diff(signal)), dimension=3)
    p = np.unique(signs, axis=0, return_counts=True)[1] / len(signs)
    assert np.allclose(nk.entropy_increment(signal, dimension=3, q=0)[0], -np.sum(p * np.log2(p)) / 2)

    # Constant increments give a single word
    assert nk.entropy_increment(np.arange(20), dimension=3)[0] == 0

    with pytest.raises(ValueError):
        nk.entropy_increment(signal, dimension=1)


def test_entropy_phase():
    def phasen_reference(signal, k):
        # Rohila & Sharma (2019): sum of the angles (in [0, 2pi)) of the SODP points in each sector
        sums = np.zeros(k)
        for i in range(len(signal) - 2):
            x, y = signal[i + 1] - signal[i], signal[i + 2] - signal[i + 1]
            angle = np.arctan2(y, x)
            if angle < 0:
                angle += 2 * np.pi
            sums[min(int(angle // (2 * np.pi / k)), k - 1)] += angle
        p = sums[sums > 0] / np.sum(sums)
        return -np.sum(p * np.log(p)) / np.log(k)

    rng = np.random.default_rng(42)
    # The rounded signal has points on the axes of the SODP (e.g., RR intervals in ms)
    for signal in [rng.normal(size=1000), np.round(rng.normal(800, 20, size=1000))]:
        for k in [4, 7, 8]:
            phasen = nk.entropy_phase(signal, k=k)[0]
            assert np.allclose(phasen, phasen_reference(signal, k))
            assert 0 < phasen <= 1
            # The normalized value does not depend on the logarithmic base
            assert np.allclose(nk.entropy_phase(signal, k=k, base=np.e)[0], phasen)

    # Lists and Series (with any index) are accepted
    assert np.allclose(nk.entropy_phase(list(signal))[0], nk.entropy_phase(signal)[0])
    assert np.allclose(nk.entropy_phase(pd.Series(signal, index=np.arange(5, 1005)))[0], nk.entropy_phase(signal)[0])


def test_entropy_attention():
    import scipy.signal

    def peaks_reference(x):
        # PkFind of EntropyHub: strict local maxima, and the middle (rounded down) of flat ones
        peaks = []
        for n in range(1, len(x) - 1):
            if x[n - 1] < x[n] > x[n + 1]:
                peaks.append(n)
            elif x[n - 1] < x[n] == x[n + 1]:
                k = 1
                while (n + k) < len(x) - 1 and x[n] == x[n + k]:
                    k += 1
                if x[n] > x[n + k]:
                    peaks.append(n + (k - 1) // 2)
        return np.array(peaks)

    def atten_reference(xmax, xmin, base=2):
        def shannon(intervals):
            p = np.unique(intervals, return_counts=True)[1] / len(intervals)
            return -np.sum(p * np.log(p)) / np.log(base)

        # Interval from each maximum (minimum) to the first minimum (maximum) after it
        maxmin = [next(j for j in xmin if j > i) - i for i in xmax if i < xmin[-1]]
        minmax = [next(j for j in xmax if j > i) - i for i in xmin if i < xmax[-1]]
        return np.mean([shannon(np.diff(xmax)), shannon(np.diff(xmin)), shannon(maxmin), shannon(minmax)])

    rng = np.random.default_rng(42)
    signal = nk.signal_simulate(duration=2, frequency=[5, 12], noise=0.3, random_state=42)
    for x in [signal, np.round(signal * 4) / 4]:  # with and without plateaus
        xmax, xmin = peaks_reference(x), peaks_reference(-x)
        atten = nk.entropy_attention(x)[0]
        assert np.allclose(atten, atten_reference(xmax, xmin))
        assert np.allclose(nk.entropy_attention(x, base=np.e)[0], atten * np.log(2))

    # With a tuned peak detection, the maxima and minima no longer alternate
    for kwargs in [{"distance": 5}, {"height": 0.2}]:
        xmax = scipy.signal.find_peaks(signal, **kwargs)[0]
        xmin = scipy.signal.find_peaks(-signal, **kwargs)[0]
        assert np.allclose(nk.entropy_attention(signal, **kwargs)[0], atten_reference(xmax, xmin))

    # Lists and Series (with any index) are accepted
    x = rng.normal(size=200)
    atten = nk.entropy_attention(x)[0]
    assert np.allclose(nk.entropy_attention(list(x))[0], atten)
    assert np.allclose(nk.entropy_attention(pd.Series(x, index=np.arange(5, 205)), show=True)[0], atten)
    plt.close("all")

    # NaN (not 0) when a type of interval is missing
    with pytest.warns(nk.misc.NeuroKitWarning, match="Not enough"):
        atten, info = nk.entropy_attention([0, 1, 0, 1, 0])
    assert np.isnan(atten) and np.isnan(info["AttEn_MinMin"])
    assert nk.entropy_attention([0, 1, 0, 1, 0, 1, 0], silent=True)[0] == 0


def test_entropy_differential():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=10000)

    # Standard normal distribution: 0.5 * log2(2 * pi * e) bits
    assert np.allclose(nk.entropy_differential(signal)[0], 0.5 * np.log2(2 * np.pi * np.e), atol=0.05)
    # Scaling the signal by a adds log(a)
    assert np.allclose(nk.entropy_differential(10 * signal)[0], nk.entropy_differential(signal)[0] + np.log2(10))

    # The default estimator depends on the length of the signal, as in scipy
    assert nk.entropy_differential(signal)[0] == nk.entropy_differential(signal, method="vasicek")[0]
    short = signal[:500]
    assert nk.entropy_differential(short)[0] == nk.entropy_differential(short, method="ebrahimi")[0]
    assert nk.entropy_differential(pd.Series(short, index=np.arange(3, 503)))[0] == nk.entropy_differential(short)[0]

    with pytest.raises(ValueError):
        nk.entropy_differential("ABBA")


# =============================================================================
# Multiscale entropy
# =============================================================================
def _multiscale_counts(x, m, r, delay=1):
    """Richman & Moorman (2000): the number of template pairs (i < j) within r at m + 1 (A) and at
    m (B), on the same N - m * delay templates, with the Chebyshev distance."""
    x = np.asarray(x, dtype=float)
    n = len(x) - m * delay
    emb = np.array([x[i : i + (m + 1) * delay : delay] for i in range(n)])
    d_m = np.max(np.abs(emb[:, None, :m] - emb[None, :, :m]), axis=2)
    d_m1 = np.maximum(d_m, np.abs(emb[:, None, m] - emb[None, :, m]))
    iu = np.triu_indices(n, 1)
    return np.sum(d_m1[iu] <= r), np.sum(d_m[iu] <= r)


def _multiscale_sampen(x, m, r, delay=1):
    A, B = _multiscale_counts(x, m, r, delay)
    return -np.log(A / B)


def _multiscale_coarse(x, tau):
    # Costa et al. (2002): means of non-overlapping windows
    n = len(x) // tau
    return np.asarray(x[: n * tau]).reshape(n, tau).mean(axis=1)


def _multiscale_mse(x, scales, m, r):
    # Costa et al. (2002)
    return [_multiscale_sampen(_multiscale_coarse(x, t), m, r) for t in scales]


def _multiscale_cmse(x, scales, m, r):
    # Wu et al. (2013), Entropy: the mean of the SampEn of the tau time-shifted series
    return [np.mean([_multiscale_sampen(_multiscale_coarse(x[k:], t), m, r) for k in range(t)]) for t in scales]


def _multiscale_rcmse(x, scales, m, r):
    # Wu et al. (2014): the numbers of matches are summed over the tau series before the ratio
    out = []
    for t in scales:
        A, B = np.sum([_multiscale_counts(_multiscale_coarse(x[k:], t), m, r) for k in range(t)], axis=0)
        out.append(-np.log(A / B))
    return out


def _multiscale_mmse(x, scales, m, r):
    # Wu et al. (2013), Physica A: SampEn of the moving average with a delay of tau
    return [_multiscale_sampen(np.convolve(x, np.ones(t) / t, mode="valid"), m, r, delay=t) for t in scales]


def test_entropy_sample_short():
    # Short series are counted with a dense distance matrix instead of a KDTree: same values
    rng = np.random.default_rng(42)
    for n in [20, 30, 50, 51, 120]:
        signal = rng.normal(size=n)
        for m in [2, 3]:
            reference = _multiscale_sampen(signal, m, 0.5)
            if np.isfinite(reference):  # Undefined (no match) cases follow NeuroKit's own convention
                assert np.allclose(nk.entropy_sample(signal, dimension=m, tolerance=0.5)[0], reference)
    # Distances exactly at the tolerance count as matches (quantized signal, integer tolerance)
    signal = rng.integers(0, 5, size=40).astype(float)
    assert np.allclose(nk.entropy_sample(signal, dimension=2, tolerance=1)[0], _multiscale_sampen(signal, 2, 1))
    assert np.allclose(
        nk.entropy_sample(signal, dimension=2, tolerance=1)[0], nk.entropy_sample(signal, dimension=2, tolerance=1.5)[0]
    )


def test_complexity_coarsegraining():
    signal = np.arange(12.0)
    assert np.allclose(nk.complexity_coarsegraining(signal, scale=3), [1, 4, 7, 10])
    rolling = nk.complexity_coarsegraining(signal, scale=3, method="rolling")
    assert np.allclose(rolling, np.arange(1, 11))

    # Time-shift (Wu et al., 2013): the k-th series averages the windows of 3 starting at sample k,
    # i.e., every 3rd value of the moving average from offset k (lengths differ by one window)
    coarse = nk.complexity_coarsegraining(signal, scale=3, method="timeshift")
    assert len(coarse) == 3
    assert np.allclose(coarse[0], [1, 4, 7, 10])
    assert np.allclose(coarse[1], [2, 5, 8])
    assert np.allclose(coarse[2], [3, 6, 9])
    for k in range(3):
        assert np.allclose(coarse[k], rolling[k::3])

    # Lists, Series and numpy integers are accepted; scale 1 returns the signal as an array
    assert isinstance(nk.complexity_coarsegraining([0, 2, 4], scale=1), np.ndarray)
    assert np.allclose(nk.complexity_coarsegraining([0, 2, 4, 6, 8, 10], scale=2), [1, 5, 9])
    assert np.allclose(nk.complexity_coarsegraining(pd.Series(signal), scale=np.int64(4)), [1.5, 5.5, 9.5])


def test_entropy_multiscale():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=400)
    m, r = 2, 0.2 * np.std(signal, ddof=1)  # The default tolerance="sd"
    scales = [1, 2, 3, 5, 8]  # 400 is not a multiple of 3 or 8: the time-shifted series differ in length

    # Values at each scale against brute-force implementations of the papers
    for method, reference in [
        ("MSEn", _multiscale_mse),
        ("CMSEn", _multiscale_cmse),
        ("RCMSEn", _multiscale_rcmse),
        ("MMSEn", _multiscale_mmse),
    ]:
        index, info = nk.entropy_multiscale(signal, scale=scales, dimension=m, method=method)
        assert np.allclose(info["Tolerance"], r)
        assert np.allclose(info["Value"], reference(signal, scales, m, r))
        # Coarse-graining low-pass filters white noise, so the entropy decreases with the scale
        assert info["Value"][-1] < info["Value"][0] - 0.5
        # The index is the average height of the curve
        assert np.allclose(index, np.trapezoid(info["Value"], x=scales) / (scales[-1] - scales[0]))

    # Index: a single scale gives SampEn, a flat curve its value, uneven scales are weighted by
    # their spacing, and undefined scales are left out
    assert np.allclose(nk.entropy_multiscale(signal, scale=[1], dimension=m)[0], nk.entropy_sample(signal, dimension=m)[0])
    from neurokit2.complexity.entropy_multiscale import _entropy_multiscale_index

    assert _entropy_multiscale_index(np.array([2.0, 2.0, 2.0, 2.0]), np.array([1, 2, 3, 4])) == 2
    assert _entropy_multiscale_index(np.array([1.0, 3.0]), np.array([1, 5])) == 2
    assert _entropy_multiscale_index(np.array([1.0, np.inf, 3.0]), np.array([1, 2, 5])) == 2
    assert np.isnan(_entropy_multiscale_index(np.array([np.inf]), np.array([1])))

    # Fuzzy versions apply the fuzzy membership at every scale (it used to apply at scale 1 only)
    rcmsen = nk.entropy_multiscale(signal, scale=[1, 2, 3], dimension=m, method="RCMSEn")[1]["Value"]
    fuzzy = nk.entropy_multiscale(signal, scale=[1, 2, 3], dimension=m, method="RCMSEn", fuzzy=True)[1]["Value"]
    assert np.allclose(fuzzy[0], nk.entropy_fuzzy(signal, dimension=m)[0])
    assert np.all(fuzzy < rcmsen - 0.3)
    assert np.allclose(nk.complexity_fuzzyrcmse(signal, scale=[1, 2, 3], dimension=m)[1]["Value"], fuzzy)

    # Scale specifications, list and Series inputs
    index, info = nk.entropy_multiscale(signal, scale=[1, 2, 3, 4, 5], dimension=m)
    for scale in [5, np.int64(5), np.arange(1, 6), (1, 2, 3, 4, 5)]:
        assert np.array_equal(nk.entropy_multiscale(signal, scale=scale, dimension=m)[1]["Scale"], np.arange(1, 6))
        assert np.allclose(nk.entropy_multiscale(signal, scale=scale, dimension=m)[0], index)
    assert np.allclose(nk.entropy_multiscale(list(signal), scale=[1, 2, 3, 4, 5], dimension=m)[0], index)
    assert np.allclose(nk.entropy_multiscale(pd.Series(signal, index=np.arange(10, 410)), scale=5, dimension=m)[0], index)
    # Default scales: 1 to 20, as long as the coarse-grained series keep at least 100 samples
    from neurokit2.complexity.utils_complexity_coarsegraining import _get_scales

    assert np.array_equal(nk.entropy_multiscale(signal, dimension=m)[1]["Scale"], np.arange(1, 5))
    assert np.array_equal(_get_scales(np.zeros(4000)), np.arange(1, 21))
    assert np.array_equal(_get_scales(np.zeros(199)), np.array([1]))
    assert np.array_equal(_get_scales(np.zeros(50)), np.array([1]))
    assert np.isfinite(nk.entropy_multiscale(signal, scale=[1, 2, 3], dimension=m, show=True)[0])
    plt.close("all")

    # complexity_mspe() is entropy_multiscale(method="MSPEn")
    assert np.allclose(nk.complexity_mspe(signal, scale=5)[0], nk.entropy_multiscale(signal, scale=5, method="MSPEn")[0])

    # Scales at which the coarse-grained series is too short to embed are NaN (instead of an error)
    for method in ["MSEn", "CMSEn", "RCMSEn", "MMSEn", "IMSEn"]:
        values = nk.entropy_multiscale(signal[:60], scale=[1, 10, 20, 29], dimension=m, method=method)[1]["Value"]
        assert np.isfinite(values[0]) and np.isnan(values[-1])
    assert np.isfinite(nk.entropy_multiscale(signal[:100], scale="max", dimension=m)[0])

    with pytest.raises(ValueError, match="'foo'"):
        nk.entropy_multiscale(signal, method="foo")

    # The hierarchical entropy uses the same index (its decomposition keeps 2 ** k samples)
    assert np.allclose(
        nk.entropy_hierarchical(signal[:256], scale=1, dimension=m)[0], nk.entropy_sample(signal[:256], dimension=m)[0]
    )


# =============================================================================
# Shannon-based entropies
# =============================================================================
def _shannon_reference(sequence, base=2):
    # Shannon (1948): -sum p log p over the states that occur
    _, counts = np.unique(np.asarray(sequence), return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p)) / np.log(base)


def test_entropy_shannon():
    rng = np.random.default_rng(42)
    symbols = rng.integers(0, 7, size=500)
    for base in [2, np.e, 10]:
        assert np.allclose(nk.entropy_shannon(symbols, base=base)[0], _shannon_reference(symbols, base))
    assert np.allclose(nk.entropy_shannon(np.arange(8))[0], 3)
    assert np.allclose(nk.entropy_shannon("ABBACDA")[0], _shannon_reference(list("ABBACDA")))
    series = pd.Series(list("ABBACDA"), index=np.arange(10, 17))
    assert np.allclose(nk.entropy_shannon(series)[0], _shannon_reference(list("ABBACDA")))

    # The arguments of complexity_symbolize (e.g., the number of symbols c) are passed on
    signal = rng.normal(size=1000)
    symbolic = nk.complexity_symbolize(signal, method="MEP", c=5)
    assert np.allclose(nk.entropy_shannon(signal, symbolize="MEP", c=5)[0], _shannon_reference(symbolic))
    assert np.allclose(nk.entropy_shannon(signal, symbolize="MEP", c=5)[0], np.log2(5), atol=1e-3)
    # A numpy integer is a number of bins, like a Python integer
    binned = pd.cut(signal, bins=5, labels=False)
    assert np.allclose(nk.entropy_shannon(signal, symbolize=np.int64(5))[0], _shannon_reference(binned))
    assert nk.entropy_shannon(signal, symbolize=np.int64(5))[0] == nk.entropy_shannon(signal, symbolize=5)[0]


def test_entropy_shannon_joint():
    def joint_reference(x, y, base=2):
        # Cover & Thomas (2006): Shannon entropy of the pairs (x_i, y_i)
        return _shannon_reference([f"{a}|{b}" for a, b in zip(x, y)], base)

    x = ["A", "A", "A", "B", "A", "B"]
    y = ["A", "B", "A", "A", "A", "A"]
    # Pairs AA (3), AB (1) and BA (2); it used to be the KL divergence of the marginals (0.119)
    assert np.allclose(nk.entropy_shannon_joint(x, y)[0], -np.sum([p * np.log2(p) for p in [1 / 2, 1 / 6, 1 / 3]]))

    rng = np.random.default_rng(42)
    a = rng.integers(0, 4, size=2000)
    b = (a + rng.integers(0, 2, size=2000)) % 5
    for base in [2, np.e]:
        assert np.allclose(nk.entropy_shannon_joint(a, b, base=base)[0], joint_reference(a, b, base))
    # H(X, X) = H(X), and max(H(X), H(Y)) <= H(X, Y) <= H(X) + H(Y)
    assert np.allclose(nk.entropy_shannon_joint(a, a)[0], nk.entropy_shannon(a)[0])
    hx, hy, hxy = nk.entropy_shannon(a)[0], nk.entropy_shannon(b)[0], nk.entropy_shannon_joint(a, b)[0]
    assert max(hx, hy) <= hxy <= hx + hy

    # Different alphabets (and types), strings and Series
    c = np.array(["x", "y", "z"])[rng.integers(0, 3, size=2000)]
    assert np.allclose(nk.entropy_shannon_joint(a, c)[0], joint_reference(a, c))
    assert np.allclose(nk.entropy_shannon_joint("ABBA", "ABAB")[0], 2)
    assert np.allclose(nk.entropy_shannon_joint(pd.Series(x, index=np.arange(3, 9)), y)[0], joint_reference(x, y))

    with pytest.raises(ValueError):
        nk.entropy_shannon_joint([0, 1, 0, 1], [0, 1, 0])


def test_entropy_renyi_tsallis():
    def renyi_reference(p, alpha):
        p = np.asarray(p, dtype=float)
        p = p[p > 0] / p.sum()
        if alpha == 1:
            return -np.sum(p * np.log(p))
        if alpha == np.inf:
            return -np.log(np.max(p))
        return np.log(np.sum(p**alpha)) / (1 - alpha)

    def tsallis_reference(p, q):
        p = np.asarray(p, dtype=float)
        p = p[p > 0] / p.sum()
        if q == 1:
            return -np.sum(p * np.log(p))
        return (1 - np.sum(p**q)) / (q - 1)

    # Counts of 6 states, 2 of which do not occur (they used to give NaN, or be counted by alpha=0)
    freq = np.array([5, 3, 0, 1, 1, 0])
    for alpha in [0, 0.5, 1, 2, 3.5, np.inf]:
        assert np.allclose(nk.entropy_renyi(freq=freq, alpha=alpha)[0], renyi_reference(freq, alpha))
    for q in [0, 0.5, 1, 2, 3.5]:
        assert np.allclose(nk.entropy_tsallis(freq=freq, q=q)[0], tsallis_reference(freq, q))

    # Known values: Hartley entropy (log of the number of states that occur), min-entropy, q=0
    assert np.allclose(nk.entropy_renyi(freq=freq, alpha=0)[0], np.log(4))
    assert np.allclose(nk.entropy_renyi(freq=freq, alpha=np.inf)[0], -np.log(0.5))
    assert np.allclose(nk.entropy_tsallis(freq=freq, q=0)[0], 4 - 1)
    # Both tend to the Shannon entropy (in nats) when alpha or q tend to 1
    shanen = nk.entropy_shannon(freq=freq, base=np.e)[0]
    assert np.allclose(nk.entropy_renyi(freq=freq, alpha=1 + 1e-7)[0], shanen)
    assert np.allclose(nk.entropy_tsallis(freq=freq, q=1 + 1e-7)[0], shanen)
    # The Rényi entropy does not increase with alpha
    values = [nk.entropy_renyi(freq=freq, alpha=alpha)[0] for alpha in [0, 0.5, 1, 2, 5, np.inf]]
    assert np.all(np.diff(values) <= 1e-12)
    # The contributions of the states (0 for those that do not occur) sum to TSEn
    tsen, info = nk.entropy_tsallis(freq=freq, q=2)
    assert len(info["Values"]) == 6 and info["Values"][2] == 0
    assert np.allclose(np.sum(info["Values"]), tsen)

    # Signals (lists, strings)
    signal = [1, 3, 3, 2, 6, 6, 6, 1, 0]
    assert np.allclose(nk.entropy_renyi(signal, alpha=2)[0], renyi_reference(np.bincount(signal), 2))
    assert np.allclose(nk.entropy_tsallis(signal, q=2)[0], tsallis_reference(np.bincount(signal), 2))
    assert np.allclose(nk.entropy_renyi("ABBA", alpha=2)[0], np.log(2))


def test_entropy_cumulativeresidual():
    def cren_reference(values):
        # Rao et al. (2004): -int_0^inf P(|X| > x) log2 P(|X| > x) dx for the empirical distribution
        values = np.abs(np.asarray(values, dtype=float))
        grid = np.unique(np.r_[0, values])
        cren = 0
        for lo, hi in zip(grid[:-1], grid[1:]):
            survival = np.mean(values > lo)  # constant on [lo, hi)
            cren -= (hi - lo) * survival * np.log2(survival)
        return cren

    rng = np.random.default_rng(42)
    # It used to compute the CDF instead of the survival function (0.311 instead of 0.5 here)
    assert np.allclose(nk.entropy_cumulativeresidual([0, 0, 0, 1])[0], 0.5)
    for signal in [[1, 1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 3], rng.integers(0, 6, size=200), rng.exponential(size=300)]:
        assert np.allclose(nk.entropy_cumulativeresidual(signal)[0], cren_reference(signal))

    # Exponential distribution of mean 2: 2 nats; uniform distribution between 0 and 4: 1 nat
    assert np.allclose(nk.entropy_cumulativeresidual(rng.exponential(scale=2, size=100000))[0], 2 / np.log(2), rtol=0.02)
    assert np.allclose(nk.entropy_cumulativeresidual(rng.uniform(0, 4, size=100000))[0], 1 / np.log(2), rtol=0.02)

    # Signals with negative values: survival function of the signal itself (independent of the origin)
    signal = rng.normal(size=300)
    cren = nk.entropy_cumulativeresidual(signal)[0]
    assert np.allclose(cren, cren_reference(signal - np.min(signal)))
    assert np.allclose(nk.entropy_cumulativeresidual(signal + 10)[0], cren)
    assert np.allclose(nk.entropy_cumulativeresidual(3 * signal)[0], 3 * cren)
    assert np.allclose(nk.entropy_cumulativeresidual(pd.Series(signal, index=np.arange(7, 307)))[0], cren)
    assert np.allclose(
        nk.entropy_cumulativeresidual(signal, symbolize=4)[0], cren_reference(pd.cut(signal, bins=4, labels=False))
    )

    # Probabilities of the states 0, 1, 2, ... (the freq argument used to raise an error)
    assert np.allclose(nk.entropy_cumulativeresidual(freq=[3, 0, 1, 2])[0], cren_reference([0, 0, 0, 2, 3, 3]))

    assert np.isnan(nk.entropy_cumulativeresidual([1, 2, np.nan, 3])[0])
    with pytest.raises(ValueError):
        nk.entropy_cumulativeresidual("ABBA")


def test_entropy_kl():
    import scipy.special

    def kl_reference(signal, delay, dimension, norm):
        # Kraskov et al. (2004, eq. 20), with k = 1 and brute-force distances
        n = len(signal) - (dimension - 1) * delay
        embedded = np.array([signal[i : i + (dimension - 1) * delay + 1 : delay] for i in range(n)])
        diff = np.abs(embedded[:, None, :] - embedded[None, :, :])
        dist = diff.max(axis=2) if norm == "max" else np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        eps = 2 * dist.min(axis=1)
        eps = eps[eps > 0]
        # Log of the volume of the ball of diameter 1
        log_c = (
            0 if norm == "max" else np.log(np.pi ** (dimension / 2) / scipy.special.gamma(1 + dimension / 2) / 2**dimension)
        )
        return scipy.special.digamma(n) + np.euler_gamma + log_c + dimension * np.mean(np.log(eps))

    rng = np.random.default_rng(42)
    signal = rng.normal(size=400)
    quantized = np.round(signal * 2)  # With identical delay vectors (but not all of them)
    for norm in ["euclidean", "max"]:
        for dimension, delay in [(1, 1), (2, 1), (3, 2)]:
            assert np.allclose(
                nk.entropy_kl(signal, delay=delay, dimension=dimension, norm=norm)[0],
                kl_reference(signal, delay, dimension, norm),
            )
        for dimension, delay in [(2, 1), (3, 2)]:
            assert np.allclose(
                nk.entropy_kl(quantized, delay=delay, dimension=dimension, norm=norm)[0],
                kl_reference(quantized, delay, dimension, norm),
            )

    # White noise: d-dimensional standard normal distribution, d / 2 * log(2 * pi * e). The Euclidean
    # norm used to be d * log(2) too high
    noise = rng.normal(size=20000)
    for dimension in [1, 2, 4]:
        for norm in ["euclidean", "max"]:
            klen = nk.entropy_kl(noise, dimension=dimension, norm=norm)[0]
            assert np.allclose(klen, dimension / 2 * np.log(2 * np.pi * np.e), atol=0.05)

    # Multiplying the signal by a adds d * log(a), also with identical delay vectors
    for x in [signal, quantized]:
        assert np.allclose(nk.entropy_kl(10 * x, dimension=3)[0] - nk.entropy_kl(x, dimension=3)[0], 3 * np.log(10))

    assert np.allclose(nk.entropy_kl(list(signal))[0], nk.entropy_kl(pd.Series(signal, index=np.arange(5, 405)))[0])
    with pytest.warns(nk.misc.NeuroKitWarning):
        assert np.isnan(nk.entropy_kl(np.ones(100))[0])
    with pytest.raises(ValueError):
        nk.entropy_kl(signal, norm="manhattan")


def test_entropy_maximum():
    # Strings are sequences of characters, as in entropy_shannon() (it used to be 0 for "ABBA")
    assert nk.entropy_maximum("ABBA")[0] == 1
    rng = np.random.default_rng(42)
    for signal in ["ABBACDA", [1, 1, 5, 5, 2, 8, 1], rng.integers(0, 9, size=100)]:
        assert nk.entropy_maximum(signal)[0] >= nk.entropy_shannon(signal)[0]
    assert np.allclose(nk.entropy_maximum(np.arange(6))[0], nk.entropy_shannon(np.arange(6))[0])
    assert np.allclose(nk.entropy_maximum(pd.Series([1, 2, 2, 3], index=[4, 5, 6, 7]))[0], np.log2(3))


# =============================================================================
# Symbolic and ordinal entropies
# =============================================================================
def test_entropy_permutation():
    import math

    def pen_reference(x, m, delay=1, weighted=False):
        # Bandt & Pompe (2002), weighted by the variance of each vector (Fadlallah et al., 2013)
        x = np.asarray(x, dtype=float)
        weights = {}
        for i in range(len(x) - (m - 1) * delay):
            vector = x[i : i + (m - 1) * delay + 1 : delay]
            pattern = tuple(np.argsort(vector, kind="stable"))
            weights[pattern] = weights.get(pattern, 0) + (np.var(vector) if weighted else 1)
        p = np.array(list(weights.values()))
        p = p[p > 0] / np.sum(p)
        return -np.sum(p * np.log2(p))

    def cpen_reference(x, m, delay=1, weighted=False):
        # Sorting entropy of Bandt & Pompe (2002): the rank of the last value of each vector of m + 1
        # values among the first m, given the pattern of the first m (weighted by the variance of the
        # m + 1 values). Returns the (weighted) counts of the pairs (pattern, rank) and of the patterns
        x = np.asarray(x, dtype=float)
        joint, first = {}, {}
        for i in range(len(x) - m * delay):
            vector = x[i : i + m * delay + 1 : delay]
            pattern = tuple(np.argsort(vector[:-1], kind="stable"))
            # Ties are ranked in their order of appearance: the last value is above the equal ones
            rank = np.sum(vector[:-1] <= vector[-1])
            w = np.var(vector) if weighted else 1
            joint[pattern, rank] = joint.get((pattern, rank), 0) + w
            first[pattern] = first.get(pattern, 0) + w
        return joint, first

    def conditional_entropy(joint, first):
        # -sum p(pattern, rank) log2 p(rank | pattern)
        total = sum(first.values())
        return -sum(w / total * np.log2(w / first[pattern]) for (pattern, _), w in joint.items() if w > 0)

    # Example of Bandt & Pompe (2002): 2 x 012, 2 x 201 and 1 x 102 out of 5 vectors
    x = [4, 7, 9, 10, 6, 11, 3]
    h = -(2 * 0.4 * np.log2(0.4) + 0.2 * np.log2(0.2))
    assert np.allclose(nk.entropy_permutation(x, corrected=False)[0], h)
    assert np.allclose(nk.entropy_permutation(x)[0], h / np.log2(6))
    # The normalized value does not depend on the logarithmic base
    assert np.allclose(nk.entropy_permutation(x, base=np.e)[0], h / np.log2(6))
    # CPEn with m = 2: 4 of the 5 vectors of 3 values start with a rise, after which the third value
    # is the highest (4, 7, 9 and 7, 9, 10) or the lowest (9, 10, 6 and 6, 11, 3), i.e., 1 bit, and
    # the third value after the fall (10, 6, 11) is the highest
    assert np.allclose(nk.entropy_permutation(x, dimension=2, conditional=True, corrected=False)[0], 0.8)
    assert np.allclose(nk.entropy_permutation(x, dimension=2, conditional=True, base=np.e)[0], 0.8 / np.log2(3))

    rng = np.random.default_rng(42)
    signal = rng.normal(size=1000)
    for m in [3, 4, 5]:
        for delay in [1, 2]:
            for weighted in [False, True]:
                h = pen_reference(signal, m, delay, weighted)
                pen = nk.entropy_permutation(signal, dimension=m, delay=delay, weighted=weighted)[0]
                assert np.allclose(pen, h / np.log2(math.factorial(m)))
                # CPEn: the conditional entropy, corrected by its maximum log(m + 1)
                h = conditional_entropy(*cpen_reference(signal, m, delay, weighted))
                cpen = nk.entropy_permutation(
                    signal, dimension=m, delay=delay, weighted=weighted, conditional=True, corrected=False
                )[0]
                assert np.allclose(cpen, h)
                cpen = nk.entropy_permutation(signal, dimension=m, delay=delay, weighted=weighted, conditional=True)[0]
                assert np.allclose(cpen, h / np.log2(m + 1))

    # White noise reaches the maximum of CPEn, and CPEn with m = 1 is the PEn with m = 2
    for weighted in [False, True]:
        for m in [1, 2, 3]:
            assert 0.98 < nk.entropy_permutation(signal, dimension=m, weighted=weighted, conditional=True)[0] <= 1
        cpen = nk.entropy_permutation(signal, dimension=1, weighted=weighted, conditional=True)[0]
        assert np.allclose(cpen, nk.entropy_permutation(signal, dimension=2, weighted=weighted)[0])
    # A monotonic signal has a single pattern, whose next value is always the highest
    assert nk.entropy_permutation(np.arange(100), conditional=True)[0] == 0

    # The correction uses the maximum of the entropy function (Rényi in nats, Tsallis)
    _, info = nk.complexity_ordinalpatterns(signal, dimension=3)
    p = info["Frequencies"]
    renyi = nk.entropy_permutation(signal, algorithm=nk.entropy_renyi, alpha=2)[0]
    assert np.allclose(renyi, -np.log(np.sum(p**2)) / np.log(6))
    tsallis = nk.entropy_permutation(signal, algorithm=nk.entropy_tsallis, q=2)[0]
    assert np.allclose(tsallis, (1 - np.sum(p**2)) / (1 - 1 / 6))
    assert 0.99 < renyi <= 1 and 0.99 < tsallis <= 1
    # For CPEn, the maximum of the difference: log(m + 1) for Rényi, and ln_q(m + 1) * max(1, (m!)^(1 - q))
    # for Tsallis, which white noise does not reach if q > 1
    joint, first = cpen_reference(signal, 3)
    pj = np.array(list(joint.values())) / sum(joint.values())
    pf = np.array(list(first.values())) / sum(first.values())
    renyi = nk.entropy_permutation(signal, conditional=True, algorithm=nk.entropy_renyi, alpha=2)[0]
    assert np.allclose(renyi, (np.log(np.sum(pf**2)) - np.log(np.sum(pj**2))) / np.log(4))
    tsallis = nk.entropy_permutation(signal, conditional=True, algorithm=nk.entropy_tsallis, q=2)[0]
    assert np.allclose(tsallis, (np.sum(pf**2) - np.sum(pj**2)) / (1 - 1 / 4))
    assert np.isclose(tsallis, (1 / 6 - 1 / 24) / (1 - 1 / 4), atol=0.01)
    tsallis = nk.entropy_permutation(signal, conditional=True, algorithm=nk.entropy_tsallis, q=0.5)[0]
    assert np.allclose(tsallis, 2 * (np.sum(pj**0.5) - np.sum(pf**0.5)) / (2 * (np.sqrt(4) - 1) * np.sqrt(6)))
    assert 0.98 < renyi <= 1 and 0.98 < tsallis <= 1

    # The corrected CPEn lies between 0 and 1, even for short signals with ties
    algorithms = [
        {"base": np.e},
        {"algorithm": nk.entropy_renyi, "alpha": 0},
        {"algorithm": nk.entropy_renyi, "alpha": 0.5},
        {"algorithm": nk.entropy_renyi, "alpha": 2},
        {"algorithm": nk.entropy_renyi, "alpha": np.inf},
        {"algorithm": nk.entropy_tsallis, "q": 0.5},
        {"algorithm": nk.entropy_tsallis, "q": 2},
    ]
    for i in range(100):
        x = rng.integers(0, 3, size=rng.integers(8, 41)) if i % 2 else rng.normal(size=rng.integers(8, 41))
        for m in [2, 3]:
            for weighted in [False, True]:
                for kwargs in algorithms:
                    cpen = nk.entropy_permutation(x, dimension=m, weighted=weighted, conditional=True, **kwargs)[0]
                    assert -1e-10 <= cpen <= 1 + 1e-10

    # Lists and Series (with any index) are accepted
    pen = nk.entropy_permutation(signal)[0]
    assert np.allclose(nk.entropy_permutation(list(signal))[0], pen)
    assert np.allclose(nk.entropy_permutation(pd.Series(signal, index=np.arange(5, 1005)))[0], pen)

    # Undefined cases
    with pytest.warns(nk.misc.NeuroKitWarning, match="missing values"):
        assert np.isnan(nk.entropy_permutation(np.r_[signal[:50], np.nan, signal[51:100]])[0])
    with pytest.warns(nk.misc.NeuroKitWarning, match="constant"):
        assert np.isnan(nk.entropy_permutation(np.ones(100), weighted=True)[0])
    assert nk.entropy_permutation(np.ones(100))[0] == 0
    with pytest.raises(ValueError):
        nk.entropy_permutation(signal, dimension=1)
    with pytest.raises(ValueError):
        nk.entropy_permutation(signal, dimension=0, conditional=True)


def test_entropy_dispersion():
    import scipy.special
    import scipy.stats

    def dispen_reference(x, m, c, delay=1, fluctuation=False):
        # Rostaghi & Azami (2016), Azami & Escudero (2018), Li et al. (2019)
        x = np.asarray(x, dtype=float)
        y = scipy.stats.norm.cdf(x, loc=np.mean(x), scale=np.std(x, ddof=1))
        z = np.floor(c * y + 0.5 + 0.5).astype(int)  # round(c * y + 0.5), with halves rounded up
        patterns = np.array([z[i : i + (m - 1) * delay + 1 : delay] for i in range(len(x) - (m - 1) * delay)])
        if fluctuation:
            patterns = np.diff(patterns, axis=1)
        _, counts = np.unique(patterns, axis=0, return_counts=True)
        p = counts / np.sum(counts)
        n = (2 * c - 1) ** (m - 1) if fluctuation else c**m
        return -np.sum(p * np.log(p)) / np.log(n), (np.sum(p**2) - 1 / n) / (1 - 1 / n)

    # Worked examples of Azami & Escudero (2018): DispEn = 1.7351 and FDispEn = 1.5596 nats
    x = [3.6, 4.2, 1.2, 3.1, 4.2, 2.1, 3.3, 4.6, 6.8, 8.4]
    assert np.allclose(nk.entropy_dispersion(x, dimension=2, c=3)[0], 1.7351 / np.log(9), atol=1e-4)
    x = [3, 4.5, 6.2, 5.1, 3.2, 1.2, 3.5, 5.6, 4.9, 8.4]
    assert np.allclose(nk.entropy_dispersion(x, dimension=3, c=2, fluctuation=True)[0], 1.5596 / np.log(9), atol=1e-4)

    rng = np.random.default_rng(42)
    signal = rng.normal(size=300)
    for m in [2, 3]:
        for c in [3, 6]:
            for delay in [1, 2]:
                for fluctuation in [False, True]:
                    dispen, info = nk.entropy_dispersion(signal, dimension=m, c=c, delay=delay, fluctuation=fluctuation)
                    reference, rden = dispen_reference(signal, m, c, delay, fluctuation)
                    assert np.allclose(dispen, reference)
                    assert 0 < dispen <= 1
                    # RDEn includes the patterns that do not occur
                    assert np.allclose(info["RDEn"], rden)
                    # The normalized value does not depend on the logarithmic base
                    assert np.allclose(
                        nk.entropy_dispersion(signal, dimension=m, c=c, delay=delay, base=np.e)[0],
                        nk.entropy_dispersion(signal, dimension=m, c=c, delay=delay)[0],
                    )

    # Fine-sorted DispEn (Fu et al., 2019): the factor is computed from the NCDF values
    y = scipy.special.ndtr(nk.standardize(signal))
    embedded = nk.complexity_embedding(y, dimension=3)
    f = np.floor(np.max(np.abs(np.diff(embedded, axis=1)), axis=1) / np.std(np.abs(np.diff(signal))))
    patterns = np.hstack(
        (nk.complexity_embedding(nk.complexity_symbolize(signal, method="NCDF", c=3), dimension=3), f[:, None])
    )
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / np.sum(counts)
    assert np.allclose(nk.entropy_dispersion(signal, c=3, symbolize="finesort")[0], -np.sum(p * np.log(p)) / np.log(27))
    assert nk.entropy_dispersion(signal, c=3, symbolize="finesort")[0] > nk.entropy_dispersion(signal, c=3)[0]
    # With a large rho, the factor is always 0
    assert np.allclose(
        nk.entropy_dispersion(signal, c=3, symbolize="finesort", rho=100)[0], nk.entropy_dispersion(signal, c=3)[0]
    )
    assert nk.entropy_dispersion(np.ones(100), symbolize="finesort")[0] == 0

    # Lists and Series (with any index) are accepted
    dispen = nk.entropy_dispersion(signal, symbolize="finesort")[0]
    assert np.allclose(nk.entropy_dispersion(list(signal), symbolize="finesort")[0], dispen)
    assert np.allclose(nk.entropy_dispersion(pd.Series(signal, index=np.arange(5, 305)), symbolize="finesort")[0], dispen)

    with pytest.warns(nk.misc.NeuroKitWarning, match="missing values"):
        assert np.isnan(nk.entropy_dispersion(np.r_[signal[:50], np.nan, signal[51:]])[0])
    with pytest.raises(ValueError):
        nk.entropy_dispersion(signal, dimension=1, fluctuation=True)
    with pytest.raises(ValueError):
        nk.entropy_dispersion(signal, c=1)


def test_entropy_slope():
    def slopen_reference(x, m, delta=0.001, gamma=1, delay=1):
        # Cuesta-Frau (2019), on the differences between the consecutive samples of each vector
        x = np.asarray(x, dtype=float)
        patterns = []
        for i in range(len(x) - (m - 1) * delay):
            d = np.diff(x[i : i + (m - 1) * delay + 1 : delay])
            symbols = np.select([d > gamma, d > delta, d >= -delta, d >= -gamma], [2, 1, 0, -1], default=-2)
            patterns.append(tuple(symbols))
        _, counts = np.unique(np.array(patterns), axis=0, return_counts=True)
        p = counts / np.sum(counts)
        return -np.sum(p * np.log2(p))

    # Worked example of Cuesta-Frau (2019, Appendix A): 14 different patterns among 31 vectors
    x = [8.2, 8.1, 4.4, 3.6, 5.3, 5.4, 8.3, 1.9, 3.7, 8.6, 9.6, 9, 6, 8.7, 6.7, 3.3, 2, 2.5, 2.7]
    x += [4.6, 9.1, 1, 3.1, 1.7, 4.1, 3.8, 6.4, 1.3, 5.7, 3.4, 2.4, 2.1, 4.2]
    counts = np.array([2, 2, 3, 2, 2, 6, 5, 2, 1, 2, 1, 1, 1, 1])
    p = counts / np.sum(counts)
    delta = np.degrees(np.arctan(0.001))
    assert np.allclose(nk.entropy_slope(x, dimension=3, thresholds=[delta, 45])[0], -np.sum(p * np.log2(p)))
    assert np.allclose(nk.entropy_slope(x, dimension=3)[0], -np.sum(p * np.log2(p)))

    rng = np.random.default_rng(42)
    signal = rng.normal(size=500)
    for m in [2, 3, 4]:
        for delay in [1, 2]:
            slopen = nk.entropy_slope(signal, dimension=m, thresholds=[delta, 45], delay=delay)[0]
            assert np.allclose(slopen, slopen_reference(signal, m, delay=delay))
            # Thresholds given as angles
            slopen = nk.entropy_slope(signal, dimension=m, thresholds=[5, 30, 60], delay=delay)[0]
            gammas = np.tan(np.radians([5, 30, 60]))
            angles = nk.complexity_embedding(signal, dimension=m, delay=delay)
            symbols = np.sign(np.diff(angles, axis=1)) * np.sum(np.abs(np.diff(angles, axis=1))[..., None] > gammas, axis=2)
            _, counts = np.unique(symbols, axis=0, return_counts=True)
            assert np.allclose(slopen, -np.sum(counts / counts.sum() * np.log2(counts / counts.sum())))

    # Lists and Series (with any index) are accepted
    slopen = nk.entropy_slope(signal)[0]
    assert np.allclose(nk.entropy_slope(list(signal))[0], slopen)
    assert np.allclose(nk.entropy_slope(pd.Series(signal, index=np.arange(5, 505)))[0], slopen)
    # A single threshold separates the ties from the rises and declines
    signs = np.sign(np.diff(nk.complexity_embedding(np.round(signal), dimension=3), axis=1))
    _, counts = np.unique(signs, axis=0, return_counts=True)
    p = counts / np.sum(counts)
    assert np.allclose(nk.entropy_slope(np.round(signal), thresholds=[1])[0], -np.sum(p * np.log2(p)))

    with pytest.warns(nk.misc.NeuroKitWarning, match="missing values"):
        assert np.isnan(nk.entropy_slope(np.r_[signal[:50], np.nan, signal[51:]])[0])
    with pytest.raises(ValueError):
        nk.entropy_slope(signal, dimension=1)
    with pytest.raises(ValueError):
        nk.entropy_slope(signal, thresholds=[45, 10])


def test_entropy_symbolicdynamic():
    def sydyen_reference(symbols, m, c, delay=1):
        # Li et al. (2017), as corrected by Wei et al. (2023, eqs. 3-7): state entropy + state
        # transition entropy (conditional on the state), with the state transitions to the symbol at
        # j + m * delay for all j <= N - m * delay
        n = len(symbols)
        states = [tuple(symbols[j : j + (m - 1) * delay + 1 : delay]) for j in range(n - (m - 1) * delay)]
        p_states = {q: states.count(q) / len(states) for q in set(states)}
        transitions = {}
        for j in range(n - m * delay):
            transitions.setdefault(states[j], []).append(symbols[j + m * delay])
        h = -sum(p * np.log(p) for p in p_states.values())
        for q, following in transitions.items():
            for s in set(following):
                p = following.count(s) / len(following)
                h -= p_states[q] * p * np.log(p)
        return h / np.log(c ** (m + 1))

    rng = np.random.default_rng(42)
    signal = rng.normal(size=300)
    for symbolize in ["MEP", "binning"]:  # "binning" gives symbols from 0 to c - 1
        for c in [3, 6]:
            symbols = list(nk.complexity_symbolize(signal, method=symbolize, c=c))
            for m in [1, 2, 3]:
                for delay in [1, 2]:
                    sydyen = nk.entropy_symbolicdynamic(signal, dimension=m, c=c, symbolize=symbolize, delay=delay)[0]
                    assert np.allclose(sydyen, sydyen_reference(symbols, m, c, delay))

    # Fluctuation-based (FSDE, Shen et al., 2025): the same on the differences between the symbols
    # spaced by the delay, with states of m - 1 differences, of which there are 2c - 1 possible ones
    for c in [3, 6]:
        symbols = nk.complexity_symbolize(signal, method="MEP", c=c)
        for m in [2, 3]:
            for delay in [1, 2]:
                fsde = nk.entropy_symbolicdynamic(signal, dimension=m, c=c, delay=delay, fluctuation=True)[0]
                differences = list(symbols[delay:] - symbols[:-delay])
                assert np.allclose(fsde, sydyen_reference(differences, m - 1, 2 * c - 1, delay))
    assert 0 <= nk.entropy_symbolicdynamic(np.cumsum(signal), symbolize="A", fluctuation=True)[0] <= 1
    with pytest.raises(ValueError):
        nk.entropy_symbolicdynamic(signal, dimension=1, fluctuation=True)
    # MFSDE: its multiscale version, with the usual coarse-graining
    _, info = nk.entropy_multiscale(signal, method="MSSyDyEn", scale=[1, 2], fluctuation=True)
    coarse = nk.complexity_coarsegraining(signal, scale=2)
    assert np.allclose(
        info["Value"],
        [nk.entropy_symbolicdynamic(s, fluctuation=True)[0] for s in [signal, coarse]],
    )

    # Normalized by the maximum: between 0 and 1, and 1 for white noise (with 2 symbols for the binary
    # symbolizations, and as many as the bins given as `symbolize`, equally likely for uniform noise)
    long = rng.normal(size=100000)
    assert np.allclose(nk.entropy_symbolicdynamic(long, dimension=2, c=3)[0], 1, atol=1e-3)
    assert np.allclose(nk.entropy_symbolicdynamic(long, symbolize="A")[0], 1, atol=1e-3)
    assert np.allclose(nk.entropy_symbolicdynamic(rng.uniform(size=100000), dimension=2, symbolize=4)[0], 1, atol=1e-3)
    for sym in ["MEP", "linear", "A", 9]:
        for walk in [signal, np.cumsum(signal), np.sin(np.arange(300) / 5)]:
            assert 0 <= nk.entropy_symbolicdynamic(walk, dimension=2, symbolize=sym)[0] <= 1

    # Lists and Series (with any index) are accepted
    sydyen = nk.entropy_symbolicdynamic(signal)[0]
    assert np.allclose(nk.entropy_symbolicdynamic(list(signal))[0], sydyen)
    assert np.allclose(nk.entropy_symbolicdynamic(pd.Series(signal, index=np.arange(5, 305)))[0], sydyen)

    with pytest.warns(nk.misc.NeuroKitWarning, match="missing values"):
        assert np.isnan(nk.entropy_symbolicdynamic(np.r_[signal[:50], np.nan, signal[51:]])[0])
    with pytest.raises(ValueError):
        nk.entropy_symbolicdynamic(signal[:3], dimension=3)
    assert np.allclose(
        nk.entropy_symbolicdynamic(signal[:4], dimension=3)[0],
        sydyen_reference(list(nk.complexity_symbolize(signal[:4], method="MEP", c=6)), 3, 6),
    )


def test_entropy_coalition():
    import scipy.signal

    def coalition_reference(signal, method):
        # Schartner et al. (2015), with the normalization drawn from np.random.RandomState(30)
        rng = np.random.RandomState(30)
        x = np.array([nk.signal_detrend(i - np.mean(i)) for i in signal.values.T])

        def entropy_of_columns(binary):
            states = [sum(int(binary[j, t]) * 2**j for j in range(len(binary))) for t in range(binary.shape[1])]
            _, counts = np.unique(states, return_counts=True)
            p = counts / np.sum(counts)
            return -np.sum(p * np.log2(p))

        if method == "ACE":
            env = np.abs(scipy.signal.hilbert(x))
            binary = (env > np.mean(env, axis=1, keepdims=True)).astype(float)
            raw = entropy_of_columns(binary)
            for row in binary:
                rng.shuffle(row)
            return raw / entropy_of_columns(binary)
        phases = np.angle(scipy.signal.hilbert(x))
        norm = entropy_of_columns((rng.rand(len(x) - 1, x.shape[1]) > 0.5).astype(float))
        values = []
        for i in range(len(x)):
            d = np.abs(phases[i] - np.delete(phases, i, axis=0))
            d[d > np.pi] = 2 * np.pi - d[d > np.pi]
            values.append(entropy_of_columns((d < 0.8).astype(float)))
        return np.mean(values) / norm

    signal = pd.DataFrame(
        {
            f"EEG{i}": nk.signal_simulate(duration=4, sampling_rate=250, frequency=[5 + i, 10], noise=0.8, random_state=i)
            for i in range(4)
        }
    )
    state = np.random.get_state()[1].copy()
    for method in ["ACE", "SCE"]:
        ce, info = nk.entropy_coalition(signal, method=method)
        assert info["Method"] == method
        assert np.allclose(ce, coalition_reference(signal, method))
        # Reproducible, and the same for the method's name or abbreviation
        name = {"ACE": "amplitude", "SCE": "synchrony"}[method]
        assert nk.entropy_coalition(signal, method=name)[0] == ce
        # An array of shape (n_channels, n_samples) is accepted
        assert nk.entropy_coalition(signal.values.T, method=method)[0] == ce
    # The global random number generator is not modified
    assert np.array_equal(np.random.get_state()[1], state)

    with pytest.raises(ValueError):
        nk.entropy_coalition(signal, method="foo")
    with pytest.raises(ValueError):
        nk.entropy_coalition(signal[["EEG0"]])
    with pytest.raises(ValueError):
        nk.entropy_coalition(signal["EEG0"])


# =============================================================================
# Ordinal patterns and Fisher-Shannon information
# =============================================================================
def test_complexity_ordinalpatterns_ties():
    # Ties are ranked in their order of appearance (Bandt & Pompe, 2002)
    rng = np.random.default_rng(0)
    signal = rng.integers(0, 3, size=300)
    for m in [3, 4, 6]:
        _, info = nk.complexity_ordinalpatterns(signal, dimension=m)
        embedded = nk.complexity_embedding(signal, dimension=m)
        # Sorted by value, then by position
        reference = np.array([np.lexsort((np.arange(m), vector)) for vector in embedded])
        assert np.array_equal(info["Permutations"], reference)


def test_fishershannon_information():
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 1, size=2000)
    fsc, info = nk.fishershannon_information(signal)
    # The Fisher information of a normal distribution is 1 / variance, and its Fisher-Shannon
    # complexity is 1 (the minimum)
    assert np.isclose(info["FI"], 1, atol=0.1)
    assert np.isclose(fsc, 1, atol=0.05)
    # It does not depend on the units of the signal
    assert np.isclose(nk.fishershannon_information(10 * signal + 3)[0], fsc)
    assert nk.fishershannon_information(rng.laplace(size=2000))[0] > 1.2


# =============================================================================
# Spectral, SVD, distribution and density-based entropies
# =============================================================================
def test_entropy_spectral():
    rng = np.random.default_rng(42)
    x = rng.normal(size=1000)

    # Inouye et al. (1991): Shannon entropy of the normalized periodogram over the band, divided by
    # the log of the number of frequency bins
    spen, info = nk.entropy_spectral(x, method="fft", sampling_rate=100, min_frequency=1)
    power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    power = power[np.fft.rfftfreq(len(x), 1 / 100) >= 1]
    p = power / np.sum(power)
    assert np.isclose(spen, -np.sum(p * np.log2(p)) / np.log2(len(p)))
    assert len(info["PSD"]) == len(p)

    # Known values: close to 1 for white noise, 0 for a sinusoid at a frequency of the periodogram
    assert nk.entropy_spectral(x)[0] > 0.95
    sine = np.sin(2 * np.pi * 5 * np.arange(400) / 100)
    assert nk.entropy_spectral(sine, method="fft", sampling_rate=100)[0] < 1e-6

    # Binned spectrum: normalized by log2(bins); numpy integers are accepted (they were ignored)
    spen10, info = nk.entropy_spectral(x, bins=10)
    assert len(info["PSD"]) == 10
    p = info["PSD"]["Power"].values
    assert np.isclose(spen10, -np.sum(p[p > 0] * np.log2(p[p > 0])) / np.log2(10))
    assert nk.entropy_spectral(x, bins=np.int64(10))[0] == spen10
    assert spen10 != nk.entropy_spectral(x)[0]
    for bins in [1, 2.5, "10"]:
        with pytest.raises(ValueError):
            nk.entropy_spectral(x, bins=bins)

    # Lists and Series (with any index) are accepted
    assert nk.entropy_spectral(list(x))[0] == nk.entropy_spectral(x)[0]
    assert nk.entropy_spectral(pd.Series(x, index=np.arange(5, 1005)))[0] == nk.entropy_spectral(x)[0]

    # Undefined cases: NaN with a warning (a too short signal used to give -0.0)
    for signal in [np.ones(100), np.r_[x[:100], np.nan], x[:5], x[:8]]:
        with pytest.warns(nk.misc.NeuroKitWarning):
            assert np.isnan(nk.entropy_spectral(signal)[0])


def test_entropy_power():
    import scipy.integrate

    rng = np.random.default_rng(42)
    x = rng.normal(0, 3, size=5000)

    # The entropy power of a Gaussian variable is its variance (it used to be about 1 / (2 pi e)
    # whatever the signal, as 2H was divided by the number of samples)
    powen, info = nk.entropy_power(x)
    assert np.isclose(powen, 9, rtol=0.1)
    # It is exp(2H) / (2 pi e), H being the differential entropy (in nats) of the density, which is
    # normalized to integrate to 1
    fx = info["Density"]
    assert np.isclose(scipy.integrate.simpson(fx, x=info["Values"]), 1)
    H = -scipy.integrate.simpson(fx * np.log(fx), x=info["Values"])
    assert np.isclose(powen, np.exp(2 * H) / (2 * np.pi * np.e))
    # It is in the squared units of the signal (also for short signals, whose density estimate
    # misses more of the tails), and does not depend on the length of the signal
    assert np.isclose(nk.entropy_power(10 * x)[0], 100 * powen)
    assert np.isclose(nk.entropy_power(10 * x[:100])[0], 100 * nk.entropy_power(x[:100])[0])
    assert np.isclose(nk.entropy_power(x[:1000])[0], 9, rtol=0.1)
    # A uniform distribution has an entropy power of 6 / (pi e) times its variance
    u = rng.uniform(size=5000)
    assert np.isclose(nk.entropy_power(u)[0] / np.var(u), 6 / (np.pi * np.e), rtol=0.02)

    # Lists and Series (with any index) are accepted
    assert nk.entropy_power(list(x))[0] == powen
    assert nk.entropy_power(pd.Series(x, index=np.arange(5, 5005)))[0] == powen

    # Clear errors instead of a LinAlgError (constant signal) or a scipy error (missing values)
    for signal in [np.ones(100), np.r_[x[:100], np.nan]]:
        with pytest.raises(ValueError, match="entropy_power"):
            nk.entropy_power(signal)


def test_entropy_svd():
    def svden_reference(signal, delay, dimension):
        # Roberts et al. (1999): entropy of the normalized singular values of the embedding matrix
        n = len(signal) - (dimension - 1) * delay
        embedded = np.array([[signal[i + j * delay] for j in range(dimension)] for i in range(n)])
        s = np.linalg.svd(embedded, compute_uv=False)
        s = s[s > 0] / np.sum(s)
        return -np.sum(s * np.log2(s))

    # Values of the docstring of antropy.svd_entropy()
    x = [4, 7, 9, 10, 6, 11, 3]
    assert np.isclose(nk.entropy_svd(x, dimension=2)[0], 0.7618909465130066)
    assert np.isclose(nk.entropy_svd(x, dimension=3)[0] / np.log2(3), 0.6870083043946692)

    rng = np.random.default_rng(42)
    signal = rng.normal(size=500)
    for delay, dimension in [(1, 2), (3, 3), (2, 5)]:
        svden = nk.entropy_svd(signal, delay=delay, dimension=dimension)[0]
        assert np.isclose(svden, svden_reference(signal, delay, dimension))
        assert 0 < svden <= np.log2(dimension)

    # A rank-deficient embedding (null singular value) gives 0 instead of NaN
    assert nk.entropy_svd([1, 0, 0, 0, 0], dimension=2)[0] == 0

    # Lists and Series (with any index) are accepted
    assert nk.entropy_svd(list(signal))[0] == nk.entropy_svd(signal)[0]
    assert nk.entropy_svd(pd.Series(signal, index=np.arange(5, 505)))[0] == nk.entropy_svd(signal)[0]

    # Missing values and null signals: NaN with a warning (instead of a LinAlgError)
    for x in [np.r_[signal[:100], np.nan], np.zeros(100)]:
        with pytest.warns(nk.misc.NeuroKitWarning):
            assert np.isnan(nk.entropy_svd(x)[0])


def test_entropy_distribution():
    def distren_reference(signal, delay, dimension, bins):
        # Li et al. (2015): Chebyshev distances between all the pairs of distinct vectors,
        # histogram of M bins, Shannon entropy normalized by log2(M)
        n = len(signal) - (dimension - 1) * delay
        embedded = np.array([[signal[i + j * delay] for j in range(dimension)] for i in range(n)])
        d = [np.max(np.abs(embedded[i] - embedded[j])) for i in range(n) for j in range(n) if i != j]
        p, _ = np.histogram(d, bins)
        p = p[p > 0] / np.sum(p)
        return -np.sum(p * np.log2(p)) / np.log2(bins)

    rng = np.random.default_rng(42)
    signal = rng.normal(size=200)
    for delay, dimension, bins in [(1, 2, 10), (2, 3, 64), (1, 3, 512)]:
        distren = nk.entropy_distribution(signal, delay=delay, dimension=dimension, bins=bins)[0]
        assert np.isclose(distren, distren_reference(signal, delay, dimension, bins))
        assert 0 < distren <= 1

    # The documented "Doane" method works (only "doanes" did)
    assert nk.entropy_distribution(signal, bins="Doane")[0] == nk.entropy_distribution(signal, bins="doanes")[0]
    with pytest.raises(ValueError):
        nk.entropy_distribution(signal, bins="unknown")
    with pytest.raises(ValueError):
        nk.entropy_distribution(signal, bins=1)

    # Lists and Series (with any index) are accepted; constant signals give 0
    assert nk.entropy_distribution(list(signal))[0] == nk.entropy_distribution(signal)[0]
    assert nk.entropy_distribution(pd.Series(signal, index=np.arange(5, 205)))[0] == nk.entropy_distribution(signal)[0]
    assert nk.entropy_distribution(np.ones(50))[0] == 0

    # Missing values and too short signals: NaN with a warning
    for x in [np.r_[signal[:100], np.nan], signal[:3], signal[:4]]:
        with pytest.warns(nk.misc.NeuroKitWarning):
            assert np.isnan(nk.entropy_distribution(x)[0])


def test_entropy_quadratic():
    def sampen_reference(signal, delay, dimension, r):
        # Richman & Moorman (2000), with the same N - m * delay templates at m and m + 1
        n = len(signal) - dimension * delay
        a = b = 0
        for i in range(n):
            for j in range(i + 1, n):
                if np.max(np.abs([signal[i + k * delay] - signal[j + k * delay] for k in range(dimension)])) <= r:
                    b += 1
                    a += np.abs(signal[i + dimension * delay] - signal[j + dimension * delay]) <= r
        return -np.log(a / b)

    rng = np.random.default_rng(42)
    signal = rng.normal(size=300)
    r = 0.2 * np.std(signal, ddof=1)
    for delay, dimension in [(1, 2), (2, 3)]:
        qse, info = nk.entropy_quadratic(signal, delay=delay, dimension=dimension)
        assert np.isclose(info["Tolerance"], r)
        # Lake (2006): QSE = SampEn + ln(2r)
        assert np.isclose(qse, sampen_reference(signal, delay, dimension, r) + np.log(2 * r))

    # With a tolerance relative to the SD, multiplying the signal by a adds ln(a)
    assert np.isclose(nk.entropy_quadratic(10 * signal)[0] - nk.entropy_quadratic(signal)[0], np.log(10))


def test_entropy_ofentropy():
    def enofen_reference(signal, scale, bins, base=2):
        # Hsu et al. (2017): Shannon entropy of each window (states = equal slices of the range of
        # the signal), then Shannon entropy of the distribution of these entropies
        edges = np.linspace(np.min(signal), np.max(signal), bins + 1)
        entropies = []
        for j in range(len(signal) // scale):
            counts, _ = np.histogram(signal[j * scale : (j + 1) * scale], edges)
            p = counts[counts > 0] / scale
            entropies.append(-np.sum(p * np.log(p)) / np.log(base))
        _, counts = np.unique(np.round(entropies, 10), return_counts=True)
        p = counts / np.sum(counts)
        return -np.sum(p * np.log(p)) / np.log(base), np.mean(entropies)

    rng = np.random.default_rng(42)
    signal = rng.normal(size=1003)
    for scale, bins in [(10, 10), (5, 55), (7, 3)]:
        enofen, info = nk.entropy_ofentropy(signal, scale=scale, bins=bins)
        reference = enofen_reference(signal, scale, bins)
        assert np.isclose(enofen, reference[0])
        assert np.isclose(info["AvEn"], reference[1])
    enofen, info = nk.entropy_ofentropy(signal, base=np.e)
    assert np.allclose([enofen, info["AvEn"]], enofen_reference(signal, 10, 10, base=np.e))

    # Lists and Series (with any index) are accepted; constant signals give 0
    assert nk.entropy_ofentropy(list(signal))[0] == nk.entropy_ofentropy(signal)[0]
    assert nk.entropy_ofentropy(pd.Series(signal, index=np.arange(5, 1008)))[0] == nk.entropy_ofentropy(signal)[0]
    assert nk.entropy_ofentropy(np.ones(100))[0] == 0

    # Missing values and signals shorter than the scale: NaN with a warning (instead of 0)
    for x in [np.r_[signal[:100], np.nan], signal[:5]]:
        with pytest.warns(nk.misc.NeuroKitWarning):
            assert np.isnan(nk.entropy_ofentropy(x, scale=10)[0])
    with pytest.raises(ValueError):
        nk.entropy_ofentropy(signal, scale=0)


# =============================================================================
# Angular and grid entropies
# =============================================================================
def test_entropy_angular():
    import scipy.optimize

    def kde_reference(data, n):
        # kde.m of Botev et al. (2010), as distributed with ComEDA (with its default interval of
        # [min - range / 2, max + range / 2])
        minimum, maximum = np.min(data), np.max(data)
        xmesh = np.linspace(minimum - (maximum - minimum) / 2, maximum + (maximum - minimum) / 2, n)
        R = xmesh[-1] - xmesh[0]
        # Distinct values (rounded, so that the cosines equal but for floating-point errors are
        # counted once)
        N = len(np.unique(np.round(data, 12)))
        # histc(data, xmesh)
        initial = np.array([np.sum((data >= xmesh[i]) & (data < xmesh[i + 1])) for i in range(n - 1)] + [0])
        initial = initial / np.sum(initial)
        # dct1d
        weight = np.r_[1, 2 * np.exp(-1j * np.arange(1, n) * np.pi / (2 * n))]
        a = np.real(weight * np.fft.fft(np.r_[initial[0::2], initial[-1::-2]]))
        squares, a2 = np.arange(1, n, dtype=float) ** 2, (a[1:] / 2) ** 2

        def fixed_point(t):
            f = 2 * np.pi**14 * np.sum(squares**7 * a2 * np.exp(-squares * np.pi**2 * t))
            for s in range(6, 1, -1):
                K0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
                const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
                time = (2 * const * K0 / N / f) ** (2 / (3 + 2 * s))
                f = 2 * np.pi ** (2 * s) * np.sum(squares**s * a2 * np.exp(-squares * np.pi**2 * time))
            return t - (2 * N * np.sqrt(np.pi) * f) ** (-2 / 5)

        tol = 10**-12 + 0.01 * (min(max(N, 50), 1050) - 50) / 1000
        while True:
            if fixed_point(0) * fixed_point(tol) < 0:
                t_star = scipy.optimize.brentq(fixed_point, 0, tol, xtol=1e-15)
                break
            tol = min(tol * 2, 0.1)
            if tol == 0.1:
                t_star = scipy.optimize.fminbound(lambda x: abs(fixed_point(x)), 0, 0.1)
                break
        # idct1d
        a_t = a * np.exp(-(np.arange(n) ** 2) * np.pi**2 * t_star / 2)
        data = np.real(np.fft.ifft(n * np.exp(1j * np.arange(n) * np.pi / (2 * n)) * a_t))
        density = np.zeros(n)
        density[0::2], density[1::2] = data[: n // 2], data[: n // 2 - 1 : -1]
        density[density < 0] = np.finfo(float).eps
        return density / R

    def angen_reference(signal, dimension, delay):
        # comEDA.m of Nardelli et al. (2022)
        x = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        vectors = [x[i : i + (dimension - 1) * delay + 1 : delay] for i in range(len(x) - (dimension - 1) * delay)]
        vectors = [v for v in vectors if np.linalg.norm(v) > 0]  # the angle with 0 is undefined
        cosines = np.array(
            [np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)) for i, u in enumerate(vectors) for v in vectors[i + 1 :]]
        )
        B = int(2 ** np.ceil(np.log2(np.ceil(np.log2(len(cosines)) + 1))))  # Sturges, power of 2
        freq = kde_reference(cosines, B)
        freq = freq / np.sum(freq)
        return -np.log(np.sum(freq**2)) / np.log(B)

    rng = np.random.default_rng(42)
    signals = [
        rng.normal(size=150),
        nk.eda_simulate(duration=30, sampling_rate=5, scr_number=3, random_state=42),
        np.round(rng.normal(size=150)),  # quantized: repeated minimum (vectors at the origin) and ties
    ]
    for signal in signals:
        for dimension, delay in [(2, 1), (3, 2)]:
            angen, info = nk.entropy_angular(signal, delay=delay, dimension=dimension)
            assert np.allclose(angen, angen_reference(signal, dimension, delay))
            assert 0 < angen < 1
            assert np.allclose(np.sum(info["pdf"]), 1)
            assert len(info["bins"]) == 16  # ~11,000 pairs: 15 bins (Sturges), rounded up to 16

    # It does not depend on the offset or the units of the signal
    signal = signals[0]
    angen = nk.entropy_angular(signal)[0]
    assert np.allclose(nk.entropy_angular(10 * signal + 5)[0], angen)

    # Lists and Series (with any index) are accepted
    assert np.allclose(nk.entropy_angular(list(signal))[0], angen)
    assert np.allclose(nk.entropy_angular(pd.Series(signal, index=np.arange(5, 155)), show=True)[0], angen)
    plt.close("all")

    # NaN (with a warning) for constant signals and missing values
    with pytest.warns(nk.misc.NeuroKitWarning, match="missing values or is constant"):
        assert np.isnan(nk.entropy_angular(np.ones(50))[0])
    with pytest.warns(nk.misc.NeuroKitWarning, match="missing values or is constant"):
        assert np.isnan(nk.entropy_angular(np.r_[signal, np.nan])[0])


def test_entropy_grid():
    from fractions import Fraction

    def griden_reference(signal, k, delay=1):
        # Yan et al. (2019): min-max normalization, k x k grids of the unit square (the maximum
        # falls in the last ones), Shannon entropy (in bits) of the proportion of points in each
        # grid. Exact arithmetic for integer signals, so that the points on the grid lines are
        # assigned to the grid above them
        lo, hi = min(signal), max(signal)
        if np.issubdtype(np.asarray(signal).dtype, np.integer):
            x = [Fraction(int(v - lo), int(hi - lo)) for v in signal]
        else:
            x = [(v - lo) / (hi - lo) for v in signal]
        cells = [(min(int(x[i] * k), k - 1), min(int(x[i + delay] * k), k - 1)) for i in range(len(x) - delay)]
        _, counts = np.unique(np.array(cells), axis=0, return_counts=True)
        p = counts / np.sum(counts)
        return -np.sum(p * np.log2(p)), len(counts) / k**2

    rng = np.random.default_rng(42)
    for signal in [rng.normal(size=500), np.round(rng.normal(800, 50, size=300)).astype(int)]:
        for k in [3, 5, 10]:
            for delay in [1, 2]:
                griden, info = nk.entropy_grid(signal, delay=delay, k=k)
                ref, gdr = griden_reference(signal, k, delay)
                assert np.allclose(griden, ref)
                assert np.allclose(info["GDR"], gdr)
                assert 0 < griden <= np.log2(k**2)
                assert np.allclose(nk.entropy_grid(signal, delay=delay, k=k, base=np.e)[0], griden * np.log(2))

    # RR intervals (in ms) on the grid lines: 720 is at 0.3 of the range of [600, 1000]
    rr = np.array([800, 600, 700, 720, 700, 720, 1000, 800])
    assert np.allclose(nk.entropy_grid(rr, k=10)[0], griden_reference(rr, 10)[0])  # 2.522 (was 2.128)

    # The grid is the same along both axes when the maximum is the first sample
    x = rng.normal(size=100)
    x[0] = 5
    griden, info = nk.entropy_grid(x, k=3)
    assert np.allclose([griden, info["GDR"]], griden_reference(x, 3))  # 2.183 and 0.778 (was 2.322 and 0.889)

    # Lists and Series (with any index) are accepted
    griden = nk.entropy_grid(x)[0]
    assert np.allclose(nk.entropy_grid(list(x))[0], griden)
    assert np.allclose(nk.entropy_grid(pd.Series(x, index=np.arange(5, 105)), show=True)[0], griden)
    plt.close("all")

    # NaN (with a warning) for constant signals and missing values, errors for invalid arguments
    with pytest.warns(nk.misc.NeuroKitWarning, match="missing values or is constant"):
        assert np.isnan(nk.entropy_grid(np.ones(50))[0])
    with pytest.warns(nk.misc.NeuroKitWarning, match="missing values or is constant"):
        assert np.isnan(nk.entropy_grid(np.r_[x, np.nan])[1]["GDR"])
    for kwargs in [{"k": 0}, {"k": 2.5}, {"delay": 0}, {"delay": 100}]:
        with pytest.raises(ValueError):
            nk.entropy_grid(x, **kwargs)


# =============================================================================
# Lyapunov exponents
# =============================================================================
def _lyapunov_logistic(n=1000, r=4.0):
    # Logistic map at r = 4: LLE = ln(2) per iteration
    x = np.zeros(n + 100)
    x[0] = 0.1
    for i in range(1, len(x)):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[100:]


def _lyapunov_henon(n=1000, a=1.4, b=0.3):
    # x-component of the Hénon map: LEs = 0.419 and -1.62 per iteration
    x, y = np.zeros(n + 100), np.zeros(n + 100)
    x[0], y[0] = 0.1, 0.1
    for i in range(1, len(x)):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
    return x[100:]


def _lyapunov_lorenz(n=5000, dt=0.01):
    # x-component of the Lorenz system (sigma=10, rho=28, beta=8/3): LEs = 0.906, 0, -14.57 per time unit
    import scipy.integrate

    def lorenz(t, s):
        return [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]]

    t = np.arange(n + 1000) * dt
    sol = scipy.integrate.solve_ivp(lorenz, (0, t[-1]), [1.0, 1.0, 1.0], t_eval=t, rtol=1e-10, atol=1e-10, method="DOP853")
    return sol.y[0, 1000:]


def test_complexity_lyapunov_rosenstein():
    def lle_reference(signal, delay, dimension, separation, len_trajectory):
        # Rosenstein et al. (1993) with a full distance matrix (nolds' lyap_r with fit="poly")
        embedded = nk.complexity_embedding(signal, delay=delay, dimension=dimension)
        n = len(embedded) - len_trajectory + 1
        dists = np.sqrt(np.sum((embedded[:n, np.newaxis] - embedded[np.newaxis, :n]) ** 2, axis=2))
        i = np.arange(n)
        dists[np.abs(i[:, np.newaxis] - i) <= separation] = np.inf
        neighbours = np.argmin(dists, axis=1)
        curve = []
        for k in range(len_trajectory):
            d = np.linalg.norm(embedded[i + k] - embedded[neighbours + k], axis=1)
            curve.append(np.mean(np.log(d[d > 0])))
        return np.polyfit(np.arange(len_trajectory), curve, 1)[0]

    def rosenstein(signal, **kwargs):
        return nk.complexity_lyapunov(signal, method="rosenstein1993", **kwargs)

    logistic = _lyapunov_logistic()
    henon = _lyapunov_henon()

    # Known exponents, when the fit is restricted to the linear part of the divergence curve
    assert np.isclose(rosenstein(logistic, len_trajectory=8)[0], np.log(2), atol=0.02)
    assert np.isclose(rosenstein(henon, len_trajectory=10)[0], 0.419, atol=0.02)
    # The default of 20 samples goes beyond it for chaotic maps
    lle, info = rosenstein(logistic)
    assert 0.3 < lle < 0.4
    assert len(info["Divergence_Rate"]) == info["Trajectory_Length"] == 20

    # Same nearest neighbours (KDTree) as with the full distance matrix
    rng = np.random.default_rng(3)
    for signal, delay, dimension, separation in [(henon, 1, 2, 5), (logistic, 2, 3, 0), (rng.normal(size=500), 1, 2, 10)]:
        lle = rosenstein(signal, delay=delay, dimension=dimension, separation=separation)[0]
        assert np.isclose(lle, lle_reference(signal, delay, dimension, separation, 20))

    # The automatic separation is the mean period, from the whole power spectrum (Rosenstein et al., 1993)
    walk = np.cumsum(rng.normal(size=2000))
    power = np.abs(np.fft.rfft(walk - np.mean(walk))) ** 2
    frequency = np.fft.rfftfreq(len(walk))
    mean_period = int(np.ceil(np.sum(power[1:]) / np.sum(frequency[1:] * power[1:])))
    assert rosenstein(walk)[1]["Separation"] == mean_period
    sine = np.sin(2 * np.pi * np.arange(2000) / 50)
    assert rosenstein(sine, delay=12)[1]["Separation"] == 50

    # Per second rather than per sample
    lle = rosenstein(henon)[0]
    assert np.isclose(rosenstein(henon, sampling_rate=100)[0], 100 * lle)

    # Lists, Series (with any index) and numpy integers
    assert rosenstein(list(henon))[0] == lle
    assert rosenstein(pd.Series(henon, index=np.arange(7, 1007)))[0] == lle
    lle_int = rosenstein(henon, delay=np.int64(1), dimension=np.int64(2), separation=np.int64(4))[0]
    assert np.isclose(lle_int, rosenstein(henon, separation=4)[0])

    # Constant signals and missing values give NaN
    with pytest.warns(nk.misc.NeuroKitWarning):
        assert np.isnan(rosenstein(np.ones(500))[0])
    henon_nan = henon.copy()
    henon_nan[10] = np.nan
    with pytest.warns(nk.misc.NeuroKitWarning):
        assert np.isnan(rosenstein(henon_nan)[0])

    # Signals too short to find neighbours outside of the separation window, and unknown arguments
    with pytest.raises(ValueError):
        rosenstein(henon[:25], separation=5)
    with pytest.raises(TypeError):
        rosenstein(henon, len_trajectori=10)

    rosenstein(henon, show=True)
    plt.close("all")


def test_complexity_lyapunov_makowski():
    # Default method. Known exponents of chaotic maps: the line is fitted to the linear part of the
    # divergence curve, from step 1 to before it levels off
    lle, info = nk.complexity_lyapunov(_lyapunov_logistic())
    assert info["Method"] == "makowski"
    assert np.isclose(lle, np.log(2), atol=0.01)
    start, end = info["Fit_Range"]
    assert 1 <= start < end < len(info["Divergence_Rate"]) == len(info["Pairs"])
    assert info["R2"] > 0.999
    assert np.isclose(nk.complexity_lyapunov(_lyapunov_henon())[0], 0.419, atol=0.02)

    # Lorenz system (a finely sampled flow): the initial transient of the curve is left out, so
    # that the exponent per second does not depend on the sampling rate (the same record every
    # 0.01 and 0.02 time unit). Rosenstein's method (over the first 20 samples) gives about 2.5
    # times the LLE
    lorenz = _lyapunov_lorenz(5000, dt=0.01)
    lle, info = nk.complexity_lyapunov(lorenz, delay=10, dimension=3, sampling_rate=100)
    assert 0.75 * 0.906 < lle < 1.25 * 0.906
    assert info["Fit_Range"][0] > 20
    assert np.isclose(nk.complexity_lyapunov(lorenz[::2], delay=5, dimension=3, sampling_rate=50)[0], lle, rtol=0.15)
    assert nk.complexity_lyapunov(lorenz, delay=10, dimension=3, method="rosenstein1993", sampling_rate=100)[0] > 2 * 0.906

    # No linear part: NaN for white noise (the neighbours become unrelated within a step or two),
    # and close to 0 for periodic signals, with or without noise (the trajectories do not diverge)
    rng = np.random.default_rng(3)
    with pytest.warns(nk.misc.NeuroKitWarning, match="reach their plateau"):
        assert np.isnan(nk.complexity_lyapunov(rng.normal(size=1000))[0])
    sine = np.sin(2 * np.pi * np.arange(2000) / (10 * np.pi))
    for signal in [sine, sine + 0.05 * rng.normal(size=2000)]:
        with pytest.warns(nk.misc.NeuroKitWarning, match="do not grow"):
            assert np.abs(nk.complexity_lyapunov(signal, delay=8)[0]) < 0.005

    # The divergence curve starts at step 0, with all the pairs that can still be followed, and
    # the nearest neighbours outside of the separation window (as in Rosenstein's method)
    henon = _lyapunov_henon(300)
    lle, info = nk.complexity_lyapunov(henon, separation=3)
    embedded = nk.complexity_embedding(henon, dimension=2)
    n = len(embedded)
    i = np.arange(n)
    dists = np.sqrt(np.sum((embedded[:, np.newaxis] - embedded[np.newaxis]) ** 2, axis=2))
    dists[np.abs(i[:, np.newaxis] - i) <= 3] = np.inf
    neighbours = np.argmin(dists, axis=1)
    for k in [0, 1, 5, 10]:
        valid = i[(i + k < n) & (neighbours + k < n)]
        d = np.linalg.norm(embedded[valid + k] - embedded[neighbours[valid] + k], axis=1)
        assert np.isclose(info["Divergence_Rate"][k], np.mean(np.log(d[d > 0])))
        assert info["Pairs"][k] == np.sum(d > 0)

    # max_length caps the number of steps
    assert len(nk.complexity_lyapunov(henon, max_length=3)[1]["Divergence_Rate"]) == 4
    with pytest.raises(ValueError):
        nk.complexity_lyapunov(henon, max_length=1)

    # Ignores the arguments of the other methods, and works with show=True and a sampling rate
    assert nk.complexity_lyapunov(henon, len_trajectory=5)[0] == nk.complexity_lyapunov(henon)[0]
    nk.complexity_lyapunov(henon, sampling_rate=100, show=True)
    plt.close("all")


def test_complexity_lyapunov_eckmann():
    henon = _lyapunov_henon(2000)

    # Hénon map: LEs = 0.419 and -1.62 (the default matrix_dim is min(dimension, 4))
    les, info = nk.complexity_lyapunov(henon, method="eckmann1986")
    assert len(les) == info["Matrix_Dimension"] == 2
    assert np.isclose(les[0], 0.419, atol=0.03)
    assert -1.8 < les[1] < -1.3

    # With m = (dimension - 1) / (matrix_dim - 1) > 1, the tangent maps are multiplied every m
    # steps: the same exponents are found (along with spurious ones, e.g., about 2 * 0.419)
    les_m1 = nk.complexity_lyapunov(henon, dimension=3, method="eckmann1986")[0]
    les_m2 = nk.complexity_lyapunov(henon, dimension=5, matrix_dim=3, method="eckmann1986")[0]
    assert np.isclose(les_m1[1], 0.419, atol=0.03)
    assert np.isclose(les_m2[1], 0.419, atol=0.03)

    # Lorenz system (sampled every 0.01 time unit): the delay is used, and the exponents are per
    # time unit with sampling_rate=100
    lorenz = _lyapunov_lorenz()
    les = nk.complexity_lyapunov(lorenz, delay=10, dimension=3, method="eckmann1986", sampling_rate=100)[0]
    assert np.isclose(les[0], 0.906, atol=0.15)
    assert np.isclose(les[1], 0, atol=0.2)
    assert les[2] < -5

    # Lists, Series (with any index) and numpy integers
    les = nk.complexity_lyapunov(henon, method="eckmann1986")[0]
    assert np.allclose(nk.complexity_lyapunov(list(henon), method="eckmann1986")[0], les)
    assert np.allclose(nk.complexity_lyapunov(pd.Series(henon, index=np.arange(5, 2005)), method="eckmann1986")[0], les)
    les_int = nk.complexity_lyapunov(henon, dimension=np.int64(2), matrix_dim=np.int64(2), method="eckmann1986")[0]
    assert np.allclose(les_int, les)

    # dimension - 1 must be a multiple of matrix_dim - 1
    with pytest.raises(ValueError):
        nk.complexity_lyapunov(henon, dimension=5, matrix_dim=4, method="eckmann1986")
    with pytest.raises(ValueError):
        nk.complexity_lyapunov(henon, dimension=1, method="eckmann1986")
    with pytest.warns(nk.misc.NeuroKitWarning):
        assert np.all(np.isnan(nk.complexity_lyapunov(np.ones(500), dimension=3, method="eckmann1986")[0]))


# =============================================================================
# Fuzzy, range, cosine similarity, Kolmogorov and rate entropies
# =============================================================================
def test_entropy_fuzzy_definition():
    # Chen et al. (2007): baseline-removed vectors, Chebyshev distance, membership exp(-d / r)
    # (n = 1), the same N - m * delay templates at m and m + 1, and self-matches excluded
    rng = np.random.default_rng(42)
    signal = rng.normal(size=300)
    r = 0.2 * np.std(signal, ddof=1)

    def vectors(x, m, delay, n):
        emb = nk.complexity_embedding(x, delay=delay, dimension=m)[:n]
        return emb - emb.mean(axis=1, keepdims=True)

    def similarity(v):
        return np.exp(-np.max(np.abs(v[:, None, :] - v[None, :, :]), axis=2) / r)

    for delay in [1, 3]:
        n = len(signal) - 2 * delay
        phi = []
        for m in [2, 3]:
            sim = similarity(vectors(signal, m, delay, n))
            phi.append((np.sum(sim) - n) / (n * (n - 1)))
        assert np.isclose(nk.entropy_fuzzy(signal, delay=delay, dimension=2)[0], np.log(phi[0] / phi[1]))

        # FuzzyApEn: as ApEn, all the vectors and the self-matches included
        phi = [np.mean(np.log(np.mean(similarity(vectors(signal, m, delay, None)), axis=1))) for m in [2, 3]]
        assert np.isclose(nk.entropy_fuzzy(signal, delay=delay, dimension=2, approximate=True)[0], phi[0] - phi[1])

    # The corrected version only exists for the approximate entropy (it used to be ignored)
    with pytest.raises(ValueError, match="approximate=True"):
        nk.entropy_fuzzy(signal, corrected=True)


def test_entropy_range_definition():
    # Omidvarnia et al. (2018): range distance, matches if d <= r (identical vectors, including the
    # vector itself, are undefined and never match), r between 0 and 1 (not scaled by the SD)
    def vectors(x, m, delay):
        return nk.complexity_embedding(x, delay=delay, dimension=m)

    def matches(v, r):
        a = np.abs(v[:, None, :] - v[None, :, :])
        with np.errstate(invalid="ignore"):
            d = (a.max(axis=2) - a.min(axis=2)) / (a.max(axis=2) + a.min(axis=2))
        return d <= r

    def rangeen_b(x, m, delay, r):
        n = len(x) - m * delay
        p = [np.sum(matches(vectors(x, mm, delay)[:n], r)) / (n * (n - 1)) for mm in [m, m + 1]]
        return -np.log(p[1] / p[0])

    def rangeen_a(x, m, delay, r):
        phi = [np.mean(np.log(np.mean(matches(vectors(x, mm, delay), r), axis=1))) for mm in [m, m + 1]]
        return phi[0] - phi[1]

    rng = np.random.default_rng(1)
    signal = rng.normal(size=300)
    for delay in [1, 3]:
        # RangEn (B): the self-match was removed twice (e.g., 1.82 instead of 1.55 for m = 3)
        for m in [2, 3]:
            rangen_b = nk.entropy_range(signal, dimension=m, delay=delay, tolerance=0.2)[0]
            assert np.isclose(rangen_b, rangeen_b(signal, m, delay, 0.2))
        # RangEn (A) (defined here, i.e., every vector has a match)
        rangen_a = nk.entropy_range(signal, dimension=2, delay=delay, tolerance=0.3, approximate=True)[0]
        assert np.isfinite(rangen_a)
        assert np.isclose(rangen_a, rangeen_a(signal, 2, delay, 0.3))
    # Ties (frequent with integer values) are matches
    quantized = np.round(signal * 3)
    assert np.isclose(nk.entropy_range(quantized, tolerance=0.2)[0], rangeen_b(quantized, 3, 1, 0.2))

    # The default tolerance is 0.2 whatever the amplitude of the signal
    rangen, info = nk.entropy_range(signal)
    assert info["Tolerance"] == 0.2
    assert np.isclose(rangen, rangeen_b(signal, 3, 1, 0.2))
    assert nk.entropy_range(800 + 50 * signal)[0] == rangen
    assert np.isclose(nk.entropy_range(pd.Series(list(signal * 1e-5), index=np.arange(5, 305)))[0], rangen)

    # Undefined cases
    with pytest.warns(nk.misc.NeuroKitWarning, match="NaN"):
        assert np.isnan(nk.entropy_range(np.ones(100))[0])
    with pytest.warns(nk.misc.NeuroKitWarning, match="NaN"):
        assert np.isnan(nk.entropy_range(np.r_[signal[:50], np.nan, signal[50:]])[0])
    with pytest.raises(ValueError, match="approximate=True"):
        nk.entropy_range(signal, corrected=True)


def test_entropy_cosinesimilarity_definition():
    # Chanwimalueang & Mandic (2017): zero-median signal, angular distance arccos(cos) / pi,
    # similar pairs (i < j) if <= r, binary Shannon entropy in bits
    def cosien(x, m, delay, r):
        x = x - np.median(x)
        emb = nk.complexity_embedding(x, delay=delay, dimension=m)
        norm = np.linalg.norm(emb, axis=1)
        cos = (emb @ emb.T) / np.outer(norm, norm)
        angle = np.arccos(np.clip(cos, -1, 1)) / np.pi
        i, j = np.triu_indices(len(emb), 1)
        b = np.mean(angle[i, j] <= r)
        return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))

    rng = np.random.default_rng(0)
    signal = rng.normal(size=500)
    for m, delay in [(2, 1), (3, 2)]:
        cosi = nk.entropy_cosinesimilarity(signal, dimension=m, delay=delay, tolerance=0.2)[0]
        assert np.isclose(cosi, cosien(signal, m, delay, 0.2))
    # Short signal (the cosines used to be stretched to [-1, 1])
    assert np.isclose(nk.entropy_cosinesimilarity(signal[:30], tolerance=0.2)[0], cosien(signal[:30], 3, 1, 0.2))

    # Values of the paper (m = 2, r = 0.07): 0.37 for white noise and 0.61 for an AR(1) process
    # (they were ln(2) = 0.69 times smaller, the entropy being in nats)
    assert np.isclose(nk.entropy_cosinesimilarity(rng.normal(size=2000), dimension=2, tolerance=0.07)[0], 0.37, atol=0.01)
    ar1 = np.zeros(3000)
    noise = rng.normal(size=3000)
    for t in range(1, 3000):
        ar1[t] = 0.9 * ar1[t - 1] + noise[t]
    assert np.isclose(nk.entropy_cosinesimilarity(ar1[1000:], dimension=2, tolerance=0.07)[0], 0.61, atol=0.04)

    # The default tolerance is 0.2 whatever the amplitude of the signal (e.g., RR intervals in ms
    # gave a tolerance of about 10, all the pairs were similar and CoSiEn was 0)
    cosi, info = nk.entropy_cosinesimilarity(signal)
    assert info["Tolerance"] == 0.2
    assert np.isclose(cosi, cosien(signal, 3, 1, 0.2))
    assert np.isclose(nk.entropy_cosinesimilarity(800 + 50 * signal)[0], cosi)
    assert np.isclose(nk.entropy_cosinesimilarity(pd.Series(list(signal), index=np.arange(3, 503)))[0], cosi)

    with pytest.warns(nk.misc.NeuroKitWarning, match="NaN"):
        assert np.isnan(nk.entropy_cosinesimilarity(np.r_[signal[:50], np.nan, signal[50:]])[0])
    with pytest.raises(ValueError, match="at least 2"):
        nk.entropy_cosinesimilarity(signal, dimension=1)


def test_entropy_kolmogorov_definition():
    # Grassberger & Procaccia (1983): K2 = ln(C_m(r) / C_m+1(r)) / delay, with the proportions of
    # the pairs of vectors closer than r (Euclidean distance)
    def correlation_integral(x, m, delay, r):
        emb = nk.complexity_embedding(x, delay=delay, dimension=m)
        d = np.sqrt(np.sum((emb[:, None, :] - emb[None, :, :]) ** 2, axis=2))
        i, j = np.triu_indices(len(emb), 1)
        return np.mean(d[i, j] < r)

    def k2(x, m, delay, r):
        return np.log(correlation_integral(x, m, delay, r) / correlation_integral(x, m + 1, delay, r)) / delay

    rng = np.random.default_rng(3)
    signal = rng.normal(size=400)
    for delay in [1, 2]:
        tolerance = 0.2 * np.std(signal, ddof=1)
        assert np.isclose(nk.entropy_kolmogorov(signal, delay=delay, dimension=2)[0], k2(signal, 2, delay, tolerance))
    # Identical vectors (distance 0) are neighbours (they used to be left out: NaN here)
    quantized = np.round(signal)
    assert np.isclose(nk.entropy_kolmogorov(quantized, dimension=2, tolerance=0.5)[0], k2(quantized, 2, 1, 0.5))

    # The logistic map at r = 4 has K2 = ln(2)
    logistic = np.zeros(3000)
    logistic[0] = 0.4
    for t in range(1, 3000):
        logistic[t] = 4 * logistic[t - 1] * (1 - logistic[t - 1])
    assert np.isclose(nk.entropy_kolmogorov(logistic, dimension=4, tolerance=0.02)[0], np.log(2), atol=0.01)

    with pytest.warns(nk.misc.NeuroKitWarning, match="NaN"):
        assert np.isnan(nk.entropy_kolmogorov(np.r_[signal[:50], np.nan, signal[50:]], tolerance=0.5)[0])


def test_entropy_rate_excess():
    # Binary Markov chain: H(k) = H(1) + (k - 1) * h exactly, so that the entropy rate is h and the
    # excess entropy E = H(1) - h (Crutchfield & Feldman, 2003); E used to be H(1)
    rng = np.random.default_rng(0)
    a, b = 0.1, 0.3  # P(1 | 0) and P(0 | 1)
    chain = np.zeros(50000, dtype=int)
    u = rng.random(len(chain))
    for t in range(1, len(chain)):
        chain[t] = u[t] < a if chain[t - 1] == 0 else u[t] >= b

    def h2(p):
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    p1 = a / (a + b)
    h = (1 - p1) * h2(a) + p1 * h2(b)
    raten, info = nk.entropy_rate(chain, kmax=6, symbolize=None)
    assert np.isclose(raten, h, atol=0.01)
    assert np.isclose(info["Excess_Entropy"], h2(p1) - h, atol=0.01)

    # Block entropies
    for k, entropy in zip(info["k"], info["Entropy"]):
        _, counts = np.unique(nk.complexity_embedding(chain, dimension=k), axis=0, return_counts=True)
        assert np.isclose(entropy, nk.entropy_shannon(freq=counts / counts.sum(), base=2)[0])

    # A random sequence has no excess entropy
    coin = rng.integers(0, 2, 20000)
    assert np.isclose(nk.entropy_rate(coin, kmax=6, symbolize=None)[1]["Excess_Entropy"], 0, atol=0.01)

    # A list of history lengths (it raised a TypeError with show=True), symbols of any type
    raten_list, _ = nk.entropy_rate(chain[:2000], kmax=[1, 2, 3, 4], symbolize=None, show=True)
    plt.close()
    assert raten_list == nk.entropy_rate(chain[:2000], kmax=4, symbolize=None)[0]
    assert raten_list == nk.entropy_rate(np.where(chain[:2000] == 1, "B", "A"), kmax=4, symbolize=None)[0]
    with pytest.raises(ValueError, match="two history lengths"):
        nk.entropy_rate(chain, kmax=1)
    with pytest.warns(nk.misc.NeuroKitWarning, match="NaN"):
        assert np.isnan(nk.entropy_rate(np.r_[rng.normal(size=50), np.nan], kmax=3)[0])


def test_entropy_utils_fuzzy_range():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=300)
    r = 0.2 * np.std(signal, ddof=1)

    # FuzzyEn with the membership function exp(-d^n / r) (Chen et al., 2007), n being passed on
    n_templates = len(signal) - 2
    for n in [1, 2]:
        phi = []
        for m in [2, 3]:
            embedded = np.array([signal[i : i + m] for i in range(n_templates)])
            embedded = embedded - embedded.mean(axis=1, keepdims=True)
            similarity = np.exp(-(np.max(np.abs(embedded[:, None] - embedded[None]), axis=2) ** n) / r)
            phi.append((similarity.sum() - n_templates) / (n_templates * (n_templates - 1)))
        assert np.isclose(nk.entropy_fuzzy(signal, tolerance=r, n=n)[0], np.log(phi[0]) - np.log(phi[1]))

    # SampEn with the range distance is RangEn (B)
    for m in [2, 3]:
        assert np.isclose(
            nk.entropy_sample(signal, dimension=m, tolerance=0.2, distance="range")[0],
            nk.entropy_range(signal, dimension=m, tolerance=0.2)[0],
        )

    # MSCoSiEn uses the tolerance of CoSiEn, which does not depend on the amplitude
    for amplitude in [1, 100]:
        _, info = nk.entropy_multiscale(amplitude * signal, scale=[1, 2], method="MSCoSiEn")
        assert np.isclose(info["Tolerance"], 0.2)
        assert np.isclose(info["Value"][0], nk.entropy_cosinesimilarity(signal)[0])


def test_fractal_hurst():
    import scipy.special

    from neurokit2.complexity.fractal_hurst import _fractal_hurst_rs, expected_rs

    # Anis & Lloyd (1976) expected R/S (exact for the population SD), continuous for n > 340
    for n in [5, 10, 50, 340, 341, 1000]:
        i = np.arange(1, n)
        expected = np.sum(np.sqrt((n - i) / i)) / np.sqrt(np.pi)
        expected *= np.exp(scipy.special.gammaln((n - 1) / 2) - scipy.special.gammaln(n / 2))
        assert np.isclose(expected_rs(n), expected)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000, 10))
    y = np.cumsum(x - x.mean(axis=1, keepdims=True), axis=1)
    rs = np.mean((y.max(axis=1) - y.min(axis=1)) / x.std(axis=1))
    assert np.isclose(rs, expected_rs(10), rtol=0.01)
    # ... which is what _fractal_hurst_rs() computes (for windows of 10)
    assert np.isclose(_fractal_hurst_rs(x.ravel(), 10), rs)

    # Corrected H of white noise is 0.5 (the R/S small-sample bias is removed)
    noise = rng.normal(size=4000)
    h, info = nk.fractal_hurst(noise)
    assert np.isclose(h, 0.5, atol=0.03)
    assert info["Corrected"]
    assert nk.fractal_hurst(noise, corrected=False)[0] > h
    assert np.isclose(nk.fractal_hurst(list(noise))[0], h)
    assert np.isclose(nk.fractal_hurst(pd.Series(noise))[0], h)


def test_fractal_sda():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=1000)

    # Reference: SD of the means of adjacent bins of `scale` points of the standardized signal
    def sda_reference(x, scales):
        x = nk.signal_detrend(x)
        x = (x - np.mean(x)) / np.std(x)
        sds = [np.std(x[: len(x) // s * s].reshape(-1, s).mean(axis=1), ddof=1) for s in scales]
        return 1 - np.polyfit(np.log10(scales), np.log10(sds), 1)[0]

    sda, info = nk.fractal_sda(signal)
    assert np.all(info["Scale"] == [1, 2, 4, 8, 16, 32, 64, 128, 256])
    assert np.isclose(sda, sda_reference(signal, info["Scale"]))
    assert np.isclose(sda, 1.5, atol=0.1)  # White noise: SD of the means ~ 1/sqrt(size)
    for scales in [[1, 2, 4, 8, 16], np.array([1, 3, 9, 27])]:
        assert np.isclose(nk.fractal_sda(list(signal), scales=scales)[0], sda_reference(signal, np.asarray(scales)))

    # Smoother signals (positively correlated increments) have a lower FD
    smooth = np.convolve(signal, np.ones(10) / 10, mode="valid")
    assert nk.fractal_sda(smooth)[0] < sda - 0.1
    with pytest.raises(ValueError, match="2 valid scales"):
        nk.fractal_sda(signal, scales=[1, 1000])


def test_fractal_psdslope():
    rng = np.random.default_rng(5)
    noise = rng.normal(size=4000)
    brown = np.cumsum(noise)

    # Voss (1988): FD = (5 - beta) / 2 for fBm, where the PSD decays as 1/f^beta (beta = -slope)
    fd, info = nk.fractal_psdslope(brown)
    assert np.isclose(info["Slope"], -2, atol=0.2)
    assert np.isclose(fd, (5 + info["Slope"]) / 2)
    assert np.isclose(fd, 1.5, atol=0.1)

    # Hasselman (2013): 1.5 for white noise, lower for persistent signals
    fd, info = nk.fractal_psdslope(noise, method="hasselman2013")
    assert np.isclose(info["Slope"], 0, atol=0.15)
    assert np.isclose(fd, 1.5, atol=0.05)
    assert nk.fractal_psdslope(brown, method="hasselman")[0] < 1.2

    # The method is validated
    with pytest.raises(ValueError, match="method"):
        nk.fractal_psdslope(noise, method="higuchi")


def test_fractal_nld():
    rng = np.random.default_rng(7)
    signal = rng.normal(size=30)

    # Kalauzi et al. (2009, eqs. 8-10): normalized by the mean absolute deviation, divided by N
    centered = signal - np.mean(signal)
    normalized = centered / np.mean(np.abs(centered))
    nld = nk.fractal_nld(signal)[0]
    assert np.isclose(nld, np.sum(np.abs(np.diff(normalized))) / len(signal))
    assert np.isclose(nk.fractal_nld(5 * signal + 3)[0], nld)

    # The calibration curve (fitted on Weierstrass functions) gives FD ~ 2 for white noise
    windows = rng.normal(size=(1000, 30))
    assert np.isclose(np.mean([nk.fractal_nld(w, corrected=True)[0] for w in windows]), 2, atol=0.03)

    # Missing values are ignored
    with_nan = np.r_[signal[:10], np.nan, signal[10:]]
    expected = np.sum(np.abs(np.diff(normalized[:10]))) + np.sum(np.abs(np.diff(normalized[10:])))
    assert np.isclose(nk.fractal_nld(with_nan)[0], expected / len(signal))


def test_fractal_linelength_sevcik():
    signal = np.array([0.0, 1.0, 3.0, 2.0, np.nan, 5.0, 4.0])
    # Differences involving missing values are ignored (not bridged over)
    assert np.isclose(nk.fractal_linelength(signal)[0], np.mean([1, 2, 1, 1]))

    # Sevcik (2010, eq. 6): a straight line in the unit square has length sqrt(2)
    n = 101
    sfd, _ = nk.fractal_sevcik(np.arange(n))
    assert np.isclose(sfd, 1 + np.log(np.sqrt(2)) / np.log(2 * (n - 1)))


def test_fractal_density():
    signal = nk.signal_simulate(duration=2, frequency=[5, 9], noise=0.01, random_state=1)
    edges = np.percentile(signal, [1, 99])
    for tolerance in [0.1, 0.25]:
        _, info = nk.fractal_density(signal, delay=20, tolerance=tolerance)
        assert info["Average"].shape == (int((edges[1] - edges[0]) / tolerance),) * 2
    assert np.isclose(nk.fractal_density(pd.Series(signal), delay=20)[0], nk.fractal_density(signal, delay=20)[0])


def test_fractal_tmf():
    signal = nk.signal_simulate(duration=1, sampling_rate=300, frequency=[5, 12], noise=0.5, random_state=2)

    tmf, info = nk.fractal_tmf(signal, n=10, random_state=3)
    assert tmf == nk.fractal_tmf(signal, n=10, random_state=3)[0]
    # tMF = (W_original - mean(W_surrogates)) / SE(W_surrogates) (Bell et al., 2019)
    w = info["Width_Surrogates"]
    assert np.isclose(tmf, (info["Width"] - np.mean(w)) / (np.std(w, ddof=1) / np.sqrt(len(w))))
    assert np.isclose(info["Width"], nk.fractal_dfa(signal, multifractal=True)[0]["Width"].iloc[0])

    # Arguments are passed on to fractal_dfa()
    _, info = nk.fractal_tmf(signal, n=3, random_state=3, q=[-3, -1, 1, 3])
    assert np.isclose(info["Width"], nk.fractal_dfa(signal, multifractal=True, q=[-3, -1, 1, 3])[0]["Width"].iloc[0])
