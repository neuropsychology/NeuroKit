# -*- coding: utf-8 -*-
import numpy as np

from ..misc import replace


def microstates_classify(microstates, segmentation=None):
    """Reorder (sort) the microstates (experimental).

    Based on the pattern of values in the vector of channels (thus, depends on how channels
    are ordered).

    Parameters
    ----------
    microstates : array | dict
        Array of microstates maps or dict (output from ``microstates_segment``).
    segmentation : array | list
        Vector containing the segmentation.

    Examples
    ------------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw").filter(1, 35)
    >>> eeg = nk.eeg_rereference(eeg, 'average')
    >>>
    >>> # Original order
    >>> out = nk.microstates_segment(eeg)
    >>> nk.microstates_plot(out, gfp=out["GFP"][0:100])
    >>>
    >>> # Reorder
    >>> out = nk.microstates_classify(out)
    >>> nk.microstates_plot(out, gfp=out["GFP"][0:100])


    """
    # Prepare the output type
    if segmentation is None:
        return_segmentation = False
    else:
        return_segmentation = True

    # Try retrieving info
    if isinstance(microstates, dict):
        segmentation = microstates["Sequence"]
        states = microstates["Microstates"]
    else:
        segmentation = None
        states = microstates

    # Reorder
    new_order = _microstates_sort(states)
    states = states[new_order]
    if segmentation is not None:
        replacement = {i: j for i, j in enumerate(new_order)}
        segmentation = replace(segmentation, replacement)


    if isinstance(microstates, dict):
        microstates["Microstates"] = states
        microstates["Sequence"] = segmentation
        states = microstates

    if return_segmentation is True:
        return states, segmentation
    else:
        return states


# =============================================================================
# Methods
# =============================================================================
def _microstates_sort(microstates):

    n_states = len(microstates)
    order_original = np.arange(n_states)

    # For each state, get linear and quadratic coefficient
    coefs_quadratic = np.zeros(n_states)
    coefs_linear = np.zeros(n_states)
    for i in order_original:
        state = microstates[i, :]
        intercept, coefs_linear[i], coefs_quadratic[i] = np.polyfit(state, np.arange(len(state)), 2)

    # For each state, which is the biggest trend, linear or quadratic
    order_quad = order_original[np.abs(coefs_linear) <= np.abs(coefs_quadratic)]
    order_lin = order_original[np.abs(coefs_linear) > np.abs(coefs_quadratic)]

    # Reorder each
    order_quad = order_quad[np.argsort(coefs_quadratic[order_quad])]
    order_lin = order_lin[np.argsort(coefs_linear[order_lin])]


    new_order = np.concatenate([order_quad, order_lin])

    return new_order
