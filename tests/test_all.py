import pytest
import doctest

import numpy as np
import pandas as pd
import neurokit2 as nk


if __name__ == '__main__':
    doctest.testmod()
    pytest.main()

def test_foo():
    assert 3 == 3

# =============================================================================
# Signal
# =============================================================================


    
def test_signal_binarize():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    assert len(signal) == 1000
#     binary = nk.signal_binarize(signal)
#     assert len(binary) == 1000
