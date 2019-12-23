import numpy as np
import pandas as pd
import neurokit2 as nk

import nolds

from pyentrp import entropy as pyentrp

# Local packages (copied in the test folder)
#from .tests_utils_pyeeg import ap_entropy as pyeeg_ap_entropy
#from packages import pyrem
#from .packages import pyeeg
#from packages import entropy



# =============================================================================
# Complexity
# =============================================================================


def test_complexity():

    signal = np.cos(np.linspace(start=0, stop=30, num=100))

    # Shannon
    assert np.allclose(nk.entropy_shannon(signal), 6.6438561897747395, atol=0.0000001)
    assert nk.entropy_shannon(signal) == pyentrp.shannon_entropy(signal)

    # Approximate
    assert np.allclose(nk.entropy_approximate(signal), 0.17364897858477146, atol=0.000001)
    assert np.allclose(nk.entropy_approximate(np.array([85, 80, 89] * 17)), 1.0996541105257052e-05, atol=0.000001)
#    assert nk.entropy_approximate(signal, 2, 0.2*np.std(signal, ddof=1)) == entropy.app_entropy(signal, 2)
    assert nk.entropy_approximate(signal, 2, 0.2*np.std(signal, ddof=1)) != pyeeg_ap_entropy(signal, 2, 0.2*np.std(signal, ddof=1))


    # Sample
    assert np.allclose(nk.entropy_sample(signal, order=2, r=0.2*np.std(signal)),
                       nolds.sampen(signal, emb_dim=2, tolerance=0.2*np.std(signal)), atol=0.000001)
#    assert nk.entropy_sample(signal, 2, 0.2) == pyeeg.samp_entropy(signal, 2, 0.2)
#    assert nk.entropy_sample(signal, 2, 0.2) == pyrem.samp_entropy(signal, 2, 0.2, relative_r=False)
#    pyentrp.sample_entropy(signal, 2, 0.2)  # Gives something different

    # Fuzzy
    assert np.allclose(nk.entropy_fuzzy(signal), 0.5216395432372958, atol=0.000001)




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