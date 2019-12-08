import numpy as np
import pandas as pd
import neurokit2 as nk


# =============================================================================
# Data
# =============================================================================

#
def test_read_acqknowledge():

    df, sampling_rate = nk.read_acqknowledge("example1")
    assert sampling_rate == 4000
