import numpy as np
import pandas as pd
import neurokit2 as nk

import mne


# =============================================================================
# EEG
# =============================================================================


def test_eeg_add_channel():


    raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_raw.fif', preload=True)
    ecg = nk.ecg_simulate(length=170000)
    raw = nk.eeg_add_channel(raw, ecg, channel_type="ecg")
    df = raw.to_data_frame()
    assert len(df.columns) == 377
