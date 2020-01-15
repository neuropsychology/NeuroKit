# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import cvxopt






# =============================================================================
# cvxEDA
# =============================================================================
def _eda_decompose_cvxEDA(eda_signal, sampling_rate=1000, sampling_rate=1000, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2, solver=None, verbose=False, options={'reltol':1e-9}):
    """Uses the same defaults as `BioSPPy
    <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/eda.py>`_.
    """
    print(3)

    return eda_signal
