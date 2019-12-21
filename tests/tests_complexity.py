import numpy as np
import pandas as pd
import neurokit2 as nk

from pyentrp import entropy as pyentrp


# =============================================================================
# Complexity
# =============================================================================


def test_complexity():

    signal = np.cos(np.linspace(start=0, stop=30, num=100))

    # Shannon
    assert np.allclose(nk.entropy_shannon(signal), 6.6438561897747395, atol=0.0000001)
    assert pyentrp.shannon_entropy(signal) == nk.entropy_shannon(signal)

    # Approximate
    assert np.allclose(nk.entropy_approximate(signal), 0.17364897858477146, atol=0.000001)
    assert np.allclose(nk.entropy_approximate(np.array([85, 80, 89] * 17)), 1.0996541105257052e-05, atol=0.000001)

#    nolds.sampen(signal, 2, 0.2)
#    nk.entropy_sample(signal, 2, 0.2)

# import entropy
# import nolds
# entropy.sample_entropy(signal)
#    pyentrp.sample_entropy(signal, 3, 0.2*np.std(signal))
#
