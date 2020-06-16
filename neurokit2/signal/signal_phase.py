# -*- coding: utf-8 -*-
import itertools

import numpy as np
import scipy.signal


def signal_phase(signal, method="radians"):
    """Compute the phase of the signal.

    The real phase has the property to rotate uniformly, leading to a uniform distribution density.
    The prophase typically doesn't fulfill this property. The following functions applies a nonlinear
    transformation to the phase signal that makes its distribution exactly uniform. If a binary vector
    is provided (containing 2 unique values), the function will compute the phase of completion of each
    phase as denoted by each value.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        The values in which the phase is expressed. Can be 'radians' (default), 'degrees'
        (for values between 0 and 360) or 'percents' (for values between 0 and 1).

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
    >>> phase = nk.signal_phase(rsp, method="degrees")
    >>> nk.signal_plot([rsp, phase])
    >>>
    >>> # Percentage of completion of two phases
    >>> signal = nk.signal_binarize(nk.signal_simulate(duration=10))
    >>> phase = nk.signal_phase(signal, method="percents")
    >>> nk.signal_plot([signal, phase])

    """
    # If binary signal
    if len(set(np.array(signal)[~np.isnan(np.array(signal))])) == 2:
        phase = _signal_phase_binary(signal)
    else:
        phase = _signal_phase_prophase(signal)

    if method.lower() in ["degree", "degrees"]:
        phase = np.rad2deg(phase)
    if method.lower() in ["perc", "percent", "percents", "percentage"]:
        phase = np.rad2deg(phase) / 360
    return phase


# =============================================================================
# Method
# =============================================================================
def _signal_phase_binary(signal):

    phase = itertools.chain.from_iterable(np.linspace(0, 1, sum([1 for i in v])) for _, v in itertools.groupby(signal))
    phase = np.array(list(phase))

    # Convert to radiant
    phase = np.deg2rad(phase * 360)
    return phase


def _signal_phase_prophase(signal):
    pi2 = 2.0 * np.pi

    # Get pro-phase
    prophase = np.mod(np.angle(scipy.signal.hilbert(signal)), pi2)

    # Transform a pro-phase to a real phase
    sort_idx = np.argsort(prophase)  # Get a sorting index
    reverse_idx = np.argsort(sort_idx)  # Get index reversing sorting
    tht = pi2 * np.arange(prophase.size) / (prophase.size)  # Set up sorted real phase
    phase = tht[reverse_idx]  # Reverse the sorting of it

    return phase
