import numpy as np
import pandas as pd

import neurokit2 as nk

# =============================================================================
# Stats
# =============================================================================


def test_standardize():

    rez = np.sum(nk.standardize([1, 1, 5, 2, 1]))
    assert np.allclose(rez, 0, atol=0.0001)

    rez = np.sum(nk.standardize(np.array([1, 1, 5, 2, 1])))
    assert np.allclose(rez, 0, atol=0.0001)

    rez = np.sum(nk.standardize(pd.Series([1, 1, 5, 2, 1])))
    assert np.allclose(rez, 0, atol=0.0001)

    rez = np.sum(nk.standardize([1, 1, 5, 2, 1, 5, 1, 7], robust=True))
    assert np.allclose(rez, 14.8387, atol=0.001)


def test_fit_loess():

    signal = np.cos(np.linspace(start=0, stop=10, num=1000))
    fit = nk.fit_loess(signal, alpha=0.75)
    assert np.allclose(np.mean(signal - fit), -0.0201905899, atol=0.0001)
