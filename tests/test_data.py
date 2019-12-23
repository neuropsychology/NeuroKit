import numpy as np
import pandas as pd
import neurokit2 as nk

import os

path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# =============================================================================
# Data
# =============================================================================


def test_read_acqknowledge():

    df, sampling_rate = nk.read_acqknowledge(os.path.join(path_data, "example_acqnowledge.acq"), sampling_rate=2000)
    assert sampling_rate == 2000
