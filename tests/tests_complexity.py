import numpy as np
import pandas as pd
import neurokit2 as nk


# =============================================================================
# Stats
# =============================================================================


def test_complexity():

    eeg = pd.read_csv('https://raw.github.com/neuropsychology/NeuroKit/master/data/example_eeg.txt', header=None)[0].values
    signal = np.cos(np.linspace(start=0, stop=30, num=len(eeg)))*70

    assert np.allclose(nk.complexity_shannon(eeg), 7.566810239706894, atol=0.0001)


