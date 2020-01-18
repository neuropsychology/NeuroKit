import numpy as np
import pandas as pd
import neurokit2 as nk
import biosppy

# =============================================================================
# EDA
# =============================================================================

def test_eda_simulate():

    eda1 = nk.eda_simulate(duration=10, length=None, n_scr=1, random_state=333)
    assert len(nk.signal_findpeaks(eda1, height_min=0.6)["Peaks"]) == 2

    eda2 = nk.eda_simulate(duration=10, length=None, n_scr=5, random_state=333)
    assert len(nk.signal_findpeaks(eda2, height_min=0.6)["Peaks"]) == 5
#   pd.DataFrame({"EDA1": eda1, "EDA2": eda2}).plot()

    assert len(nk.signal_findpeaks(eda2, height_min=0.6)["Peaks"]) > len(nk.signal_findpeaks(eda1, height_min=0.6)["Peaks"])





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





def test_eda_phasic():

    sampling_rate = 1000
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate,
                          n_scr=6, noise=0.01, drift=0.01, random_state=42)


    cvxEDA = nk.eda_phasic(nk.standardize(eda), method='cvxeda')
    assert len(cvxEDA) == len(eda)


    smoothMedian = nk.eda_phasic(nk.standardize(eda), method='smoothmedian')
    assert len(smoothMedian) == len(eda)


    highpass = nk.eda_phasic(nk.standardize(eda), method='highpass')
    assert len(highpass) == len(eda)






def test_eda_findpeaks():

    sampling_rate = 1000
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate,
                          n_scr=6, noise=0, drift=0.01, random_state=42)
    eda_phasic = nk.eda_phasic(nk.standardize(eda), method='highpass')["EDA_Phasic"]


    signals, info = nk.eda_findpeaks(eda_phasic, method="gamboa2008")
    onsets, peaks, amplitudes = biosppy.eda.basic_scr(eda_phasic, sampling_rate=1000)
    assert np.allclose((info["SCR_Peaks"] - peaks).mean(), 0, atol=1e-5)

    signals, info = nk.eda_findpeaks(eda_phasic, method="kim2004")
    onsets, peaks, amplitudes = biosppy.eda.kbk_scr(eda_phasic, sampling_rate=1000)
    assert np.allclose((info["SCR_Peaks"] - peaks).mean(), 0, atol=1)
