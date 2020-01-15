import numpy as np
import pandas as pd
import neurokit2 as nk
import biosppy

# =============================================================================
# EDA
# =============================================================================

def test_eda_simulate():

    eda1 = nk.eda_simulate(duration=10, length=None, n_scr=1)
    assert len(nk.signal_findpeaks(eda1, height_min=0.6)[0]) == 1

    eda2 = nk.eda_simulate(duration=10, length=None, n_scr=5)
    assert len(nk.signal_findpeaks(eda2, height_min=0.6)[0]) == 5
#   pd.DataFrame({"EDA1": eda1, "EDA2": eda2}).plot()

    assert len(nk.signal_findpeaks(eda2, height_min=0.6)[0]) > len(nk.signal_findpeaks(eda1, height_min=0.6)[0])





def test_eda_clean():

    sampling_rate = 1000
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate,
                          n_scr=6, noise=0.01, drift=0.01, random_state=42)

    clean = nk.eda_clean(eda, sampling_rate=sampling_rate)
    assert len(clean) == len(eda)

    # Comparison to biosppy (https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/eda.py)

    eda_biosppy = nk.eda_clean(eda, sampling_rate=sampling_rate, method="biosppy")
    original, _, _ = biosppy.tools.filter_signal(signal=eda,
                                             ftype='butter',
                                             band='lowpass',
                                             order=4,
                                             frequency=5,
                                             sampling_rate=sampling_rate)

    original, _ = biosppy.tools.smoother(signal=original,
                              kernel='boxzen',
                              size=int(0.75 * sampling_rate),
                              mirror=True)

#    pd.DataFrame({"our":eda_biosppy, "biosppy":original}).plot()
    assert np.allclose((eda_biosppy - original).mean(), 0, atol=1e-5)


