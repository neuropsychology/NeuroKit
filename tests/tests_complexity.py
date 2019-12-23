import numpy as np
import pandas as pd
import neurokit2 as nk

import nolds

from pyentrp import entropy as pyentrp

# Local packages (copied in the test folder)
from utils_pyeeg import ap_entropy as pyeeg_ap_entropy
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
#    assert nk.entropy_approximate(signal, 2, 0.2*np.std(signal, ddof=1)) == pyeeg.ap_entropy(signal, 0.2*np.std(signal, ddof=1))


    # Sample
    assert np.allclose(nk.entropy_sample(signal, order=2, r=0.2*np.std(signal)),
                       nolds.sampen(signal, emb_dim=2, tolerance=0.2*np.std(signal)), atol=0.000001)
#    assert nk.entropy_sample(signal, 2, 0.2) == pyeeg.samp_entropy(signal, 2, 0.2)
#    assert nk.entropy_sample(signal, 2, 0.2) == pyrem.samp_entropy(signal, 2, 0.2, relative_r=False)
#    pyentrp.sample_entropy(signal, 2, 0.2)  # Gives something different

    # Fuzzy
    assert np.allclose(nk.entropy_fuzzy(signal), 0.5216395432372958, atol=0.000001)


