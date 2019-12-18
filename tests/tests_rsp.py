import numpy as np
import pandas as pd
import neurokit2 as nk

import scipy.stats

# =============================================================================
# RSP
# =============================================================================

def test_rsp_simulate():
    rsp1 = nk.rsp_simulate(duration=20, length=3000)
    assert len(rsp1) == 3000

    rsp2 = nk.rsp_simulate(duration=20, length=3000, respiratory_rate=80)
#    pd.DataFrame({"RSP1":rsp1, "RSP2":rsp2}).plot()
#    pd.DataFrame({"RSP1":rsp1, "RSP2":rsp2}).hist()
    assert len(nk.signal_findpeaks(rsp1, height_min = 0.2)[0]) < len(nk.signal_findpeaks(rsp2, height_min = 0.2)[0])

    rsp3 = nk.rsp_simulate(duration=20, length=3000, method="sinusoidal")
    rsp4 = nk.rsp_simulate(duration=20, length=3000, method="breathmetrics")
#    pd.DataFrame({"RSP3":rsp3, "RSP4":rsp4}).plot()
    assert len(nk.signal_findpeaks(rsp3, height_min = 0.2)[0]) > len(nk.signal_findpeaks(rsp4, height_min = 0.2)[0])



