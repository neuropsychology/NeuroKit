# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal


def signal_phase(signal):
    """Compute the phase of the signal.

    The real phase has the property to rotate uniformly, leading to a
    uniform distribution density. The prophase typically doesn't fulfill
    this property. The following functions applies a nonlinear transformation to
    the phase signal that makes its distribution exactly uniform.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.

    See Also
    --------
    signal_filter, signal_zerocrossings, signal_findpeaks

    Returns
    -------
    array
        A vector containing the phase of the signal, between 0 and 2*pi.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=10)
    >>> phase = nk.signal_phase(signal)
    >>> nk.signal_plot([signal, phase])
    >>>
    >>> rsp = nk.rsp_simulate(duration=30)
    >>> phase = nk.signal_phase(rsp)
    >>> nk.signal_plot([rsp, phase])
    """
    pi2 = 2.0*np.pi

    # Get pro-phase
    prophase = np.mod(np.angle(scipy.signal.hilbert(signal)), pi2)

    # Transform a pro-phase to a real phase
    sort_idx = np.argsort(prophase)  # Get a sorting index
    reverse_idx = np.argsort(sort_idx)  # Get index reversing sorting
    tht = pi2 * np.arange(prophase.size)/(prophase.size)  # Set up sorted real phase
    phase = tht[reverse_idx]  # Reverse the sorting of it

    return phase
