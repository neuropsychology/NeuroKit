import numpy as np
import pandas as pd
import neurokit2 as nk

# from pyentrp import entropy as pyentrp


# =============================================================================
# Complexity
# =============================================================================


#def test_complexity():
#
#    eeg = pd.read_csv('https://raw.github.com/neuropsychology/NeuroKit/master/data/example_eeg.txt', header=None)[0].values
#    signal = np.cos(np.linspace(start=0, stop=30, num=len(eeg)))*70
#
#    # Shannon
#    assert np.allclose(nk.entropy_shannon(eeg), 7.566810239706894, atol=0.0000001)
#    assert pyentrp.shannon_entropy(eeg) == nk.entropy_shannon(eeg)
#
#    # Approximate
#    assert np.allclose(nk.entropy_approximate(eeg), 1.0006433431773685, atol=0.000001)
#    assert np.allclose(nk.entropy_approximate(np.array([85, 80, 89] * 17)), 1.0996541105257052e-05, atol=0.000001)
