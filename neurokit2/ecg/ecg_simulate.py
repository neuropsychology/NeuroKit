# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy

from ..signal import signal_resample
from ..signal import signal_distord
from .ecg_simulate_ecgsyn import _ecg_simulate_ecgsyn


def ecg_simulate(duration=10, length=None, sampling_rate=1000, noise=0.01,
                 heart_rate=70, method="ecgsyn", random_state=42):
    """Simulate an ECG/EKG signal

    Generate an artificial (synthetic) ECG signal of a given duration and sampling rate using either the ECGSYN dynamical model (McSharry et al., 2003) or a simpler model based on Daubechies wavelets to roughly approximate cardiac cycles.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate, length : int
        The desired sampling rate (in Hz, i.e., samples/second) or the desired length of the signal (in samples).
    noise : float
        Noise level (amplitude of the laplace noise).
    heart_rate : int
        Desired simulated heart rate (in beats per minute).
    method : str
        The model used to generate the signal. Can be 'simple' for a
        simulation based on Daubechies wavelets that roughly approximates
        a single cardiac cycle. If 'ecgsyn' (default), will use an
        advanced model desbribed `McSharry et al. (2003)
        <https://physionet.org/content/ecgsyn/>`_.
    random_state : int
        Seed for the random number generator.



    Returns
    ----------
    array
        Vector containing the ECG signal.

    Examples
    ----------
    >>> import pandas as pd
    >>> import neurokit as nk
    >>>
    >>> ecg1 = nk.ecg_simulate(duration=10, method="simple")
    >>> ecg2 = nk.ecg_simulate(duration=10, method="ecgsyn")
    >>> pd.DataFrame({"ECG_Simple": ecg1,
                      "ECG_Complex": ecg2}).plot(subplots=True)

    See Also
    --------
    rsp_simulate, eda_simulate, ppg_simulate, emg_simulate


    References
    -----------
    - McSharry, P. E., Clifford, G. D., Tarassenko, L., & Smith, L. A. (2003). A dynamical model for generating synthetic electrocardiogram signals. IEEE transactions on biomedical engineering, 50(3), 289-294.
    - https://github.com/diarmaidocualain/ecg_simulation
    """
    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate
    if duration is None:
        duration = length / sampling_rate

    # Run appropriate method
    if method.lower() in ["simple", "daubechies"]:
        ecg = _ecg_simulate_daubechies(duration=duration,
                                       length=length,
                                       sampling_rate=sampling_rate,
                                       noise=noise,
                                       heart_rate=heart_rate,
                                       random_state=random_state)
    else:
        approx_number_beats = int(np.round(duration * (heart_rate / 60)))
        ecg = _ecg_simulate_ecgsyn(sfecg=sampling_rate,
                                   N=approx_number_beats,
                                   Anoise=0,
                                   hrmean=heart_rate,
                                   hrstd=1,
                                   lfhfratio=0.5,
                                   sfint=sampling_rate,
                                   ti=(-70, -15, 0, 15, 100),
                                   ai=(1.2, -5, 30, -7.5, 0.75),
                                   bi=(0.25, 0.1, 0.1, 0.1, 0.4),
                                   random_state=random_state)
        # Cut to match expected length
        ecg = ecg[0:length]

    # Add random noise
    if noise > 0:
        ecg = signal_distord(ecg,
                             sampling_rate=sampling_rate,
                             noise_amplitude=noise,
                             noise_frequency=[5, 10, 100],
                             noise_shape="laplace")

    return(ecg)










def _ecg_simulate_daubechies(duration=10, length=None, sampling_rate=1000, noise=0.01,
                             heart_rate=70, random_state=42):
    """Generate an artificial (synthetic) ECG signal of a given duration and sampling rate.
    It uses a 'Daubechies' wavelet that roughly approximates a single cardiac cycle.
    This function is based on `this script <https://github.com/diarmaidocualain/ecg_simulation>`_.
    """

    # Seed the random generator for reproducible results
    np.random.seed(random_state)

    # The "Daubechies" wavelet is a rough approximation to a real, single, cardiac cycle
    cardiac = scipy.signal.wavelets.daub(10)

    # Add the gap after the pqrst when the heart is resting.
    cardiac = np.concatenate([cardiac, np.zeros(10)])

    # Caculate the number of beats in capture time period
    num_heart_beats = int(duration * heart_rate / 60)

    # Concatenate together the number of heart beats needed
    ecg = np.tile(cardiac , num_heart_beats)

    # Add random (gaussian distributed) noise
    ecg += np.random.normal(0, noise, len(ecg))

    # Resample
    ecg = signal_resample(ecg,
                          sampling_rate=int(len(ecg)/10),
                          desired_length=length,
                          desired_sampling_rate=sampling_rate)

    return(ecg)
