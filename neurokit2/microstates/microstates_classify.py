# -*- coding: utf-8 -*-
import numpy as np

from ..misc import replace


def microstates_classify(segmentation, microstates):
    """Reorder (sort) the microstates (experimental).

    Based on the pattern of values in the vector of channels (thus, depends on how channels
    are ordered).

    Parameters
    ----------
    segmentation : Union[np.array, dict]
        Vector containing the segmentation.
    microstates : Union[np.array, dict]
        Array of microstates maps . Defaults to None.

    Returns
    -------
    segmentation, microstates
        Tuple containing re-ordered input.

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
    # Reorder
    new_order = _microstates_sort(microstates)
    microstates = microstates[new_order]

    replacement = {i: j for i, j in enumerate(new_order)}
    segmentation = replace(segmentation, replacement)

    return segmentation, microstates



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
        _, coefs_linear[i], coefs_quadratic[i] = np.polyfit(state, np.arange(len(state)), 2)

    # For each state, which is the biggest trend, linear or quadratic
    order_quad = order_original[np.abs(coefs_linear) <= np.abs(coefs_quadratic)]
    order_lin = order_original[np.abs(coefs_linear) > np.abs(coefs_quadratic)]

    # Reorder each
    order_quad = order_quad[np.argsort(coefs_quadratic[order_quad])]
    order_lin = order_lin[np.argsort(coefs_linear[order_lin])]

    new_order = np.concatenate([order_quad, order_lin])

    return new_order
