# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy

from ..signal import signal_resample


def ecg_simulate(duration=10, length=None, sampling_rate=1000, bpm=60, noise=0.01):
    """Simulate an ECG/EKG signal

    Generate an artificial (synthetic) ECG signal of a given duration and sampling rate. It uses a 'Daubechies' wavelet that roughly approximates a single cardiac cycle.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate, length : int
        The desired sampling rate (in Hz, i.e., samples/second) or the desired length of the signal (in samples).
    bpm : int
        Desired simulated heart rate.
    noise : float
       Noise level.


    Returns
    ----------
   array
        Array containing the ECG signal.

    Example
    ----------
    >>> import neurokit as nk
    >>> import pandas as pd
    >>>
    >>> ecg = nk.ecg_simulate(duration=10, bpm=60, sampling_rate=1000, noise=0.01)
    >>> pd.Series(ecg).plot()

    See Also
    --------
    signal_resample


    Credits
    -------
    This function is based on `this script <https://github.com/diarmaidocualain/ecg_simulation>`_.
    """
    # The "Daubechies" wavelet is a rough approximation to a real, single, cardiac cycle
    cardiac = scipy.signal.wavelets.daub(10)
    # Add the gap after the pqrst when the heart is resting.
    cardiac = np.concatenate([cardiac, np.zeros(10)])

    # Caculate the number of beats in capture time period
    num_heart_beats = int(duration * bpm / 60)

    # Concatenate together the number of heart beats needed
    ecg = np.tile(cardiac , num_heart_beats)

    # Add random (gaussian distributed) noise
    noise = np.random.normal(0, noise, len(ecg))
    ecg = noise + ecg

    # Resample
    ecg = signal_resample(ecg, sampling_rate=1000, desired_length=length, desired_sampling_rate=sampling_rate)

    return(ecg)




def _ecg_simulate2(sfecg=256, N=256, Anoise=0, hrmean=60, hrstd=1, lfhfratio=0.5, sfint=256, ti=[-70, -15, 0, 15, 100], ai=[1.2, -5, 30, -7.5, 0.75], bi=[0.25, 0.1, 0.1, 0.1, 0.4]):
    """
    Credits
    -------
    This function is a python translation of the matlab script by Patrick McSharry & Gari Clifford (2013). All credits go to them.
    """
    # Set parameter default values
    sfecg = 256
    N = 256
    Anoise = 0
    hrmean = 60
    hrstd = 1
    lfhfratio = 0.5
    sfint = 512
    ti = [-70, -15, 0, 15, 100]
    ai=[1.2, -5, 30, -7.5, 0.75]
    bi=[0.25, 0.1, 0.1, 0.1, 0.4]


    ti = np.array(ti)*np.pi/180

    # Adjust extrema parameters for mean heart rate
    hrfact =  np.sqrt(hrmean/60)
    hrfact2 = np.sqrt(hrfact)
    bi = hrfact*np.array(bi)
    ti = np.array([hrfact2, hrfact, 1, hrfact, hrfact2])*ti

    # Check that sfint is an integer multiple of sfecg
    q = np.round(sfint/sfecg)
    qd = sfint/sfecg
    if q != qd:
        raise ValueError('Internal sampling frequency (sfint) must be an integer multiple of the ECG sampling frequency (sfecg). Your current choices are: sfecg = ' + str(sfecg) + ' and sfint = ' + str(sfint) + '.')

    # Define frequency parameters for rr process
    # flo and fhi correspond to the Mayer waves and respiratory rate respectively
    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01
    fid = 1


    # Calculate time scales for rr and total output
    sfrr = 1
    trr = 1/sfrr
    tstep = 1/sfecg
    rrmean = 60/hrmean
    n = (np.ceil(np.log2(N*rrmean/trr)))**2

    rr0 = rrprocess(flo,fhi,flostd,fhistd,lfhfratio,hrmean,hrstd,sfrr,n)



def _ecg_simulate_rrprocess(flo=0.1, fhi=0.25, flostd=0.01, fhistd=0.01, lfhfratio=0.5, hrmean=60, hrstd=1, sfrr=1, n=64):

    # ----------
    # Here are the default values for the arguments (to be removed from the final function)
    flo=0.1
    fhi=0.25
    flostd=0.01
    fhistd=0.01
    lfhfratio=0.5
    hrmean=60
    hrstd=1
    sfrr=1
    n=64
    # ----------



    w1 = 2*np.pi*flo
    w2 = 2*np.pi*fhi
    c1 = 2*np.pi*flostd
    c2 = 2*np.pi*fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60/hrmean
    rrstd = 60*hrstd/(hrmean*hrmean)

    df = sfrr/n
    w = np.arange(n-1)*2*np.pi*df
    dw1 = w-w1
    dw2 = w-w2

    Hw1 = sig1*np.exp(-0.5*(dw1/c1)**2)/np.sqrt(2*np.pi*c1**2)
    Hw2 = sig2*np.exp(-0.5*(dw2/c2)**2)/np.sqrt(2*np.pi*c2**2)
    Hw = Hw1 + Hw2

    Hw0 = [Hw*np.arange(1, n/2), Hw*np.arange(n/2, 1 ,-1)]
#    Sw = (sfrr/2)*sqrt(Hw0)
#
#    ph0 = 2*np.pi*np.random.uniform(n/2-1, 1)
#    ph = [ 0; ph0; 0; -flipud(ph0) ];
#    SwC = Sw .* exp(j*ph)
#    x = (1/n)*real(ifft(SwC))
#
#    xstd = std(x)
#    ratio = rrstd/xstd
#    rr = rrmean + x*ratio
#    return(rr)

