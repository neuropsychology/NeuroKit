# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def microstates_plot(microstates, segmentation=None, gfp=None, info=None):
    """
    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw").filter(1, 35)
    >>> eeg = nk.eeg_rereference(eeg, 'average')
    >>>
    >>> microstates = nk.microstates_clustering(eeg, select="gfp")
    >>> nk.microstates_plot(microstates, eeg)
    """
    # Try retrieving info
    if isinstance(microstates, dict):
        if info is None and "Info" in microstates.keys():
            info = microstates["Info"]
        if gfp is None and "GFP" in microstates.keys():
            gfp = microstates["GFP"]
        segmentation = microstates["Sequence"]
        microstates = microstates["Microstates"]

    _microstates_plot_topos(microstates, info=info)
    _microstates_plot_segmentation(segmentation, gfp, info)

    pass



def _microstates_plot_topos(microstates, info):
    """Plot prototypical microstate maps.
    """
    # Sanity check
    if info is None:
        raise ValueError("NeuroKit error: microstate_plot(): An MNE-object must be passed to ",
                         " 'mne_object' in order to plot the topoplots.")

    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: eeg_add_channel(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        )

    plt.figure(figsize=(2 * len(microstates), 2))
    for i, map in enumerate(microstates):
        plt.subplot(1, len(microstates), i + 1)
        mne.viz.plot_topomap(map, info)
        plt.title('%d' % i)


def _microstates_plot_segmentation(segmentation, gfp, info=None):
    """Plot a microstate segmentation.
    """
    # Sanity checks
    if gfp is None:
        raise ValueError("NeuroKit error: microstate_plot(): GFP data must be passed to ",
                         " 'gfp' in order to plot the segmentation.")

    if info is not None and "sfreq" in info.keys():
        times = np.arange(len(gfp)) / info["sfreq"]
    else:
        times = np.arange(len(gfp))

    if len(segmentation) > len(gfp):
        segmentation = segmentation[0:len(gfp)]
    if len(segmentation) < len(gfp):
        gfp = gfp[0:len(segmentation)]

    n_states = len(np.unique(segmentation))
    plt.figure(figsize=(6 * np.ptp(times), 2))
    cmap = plt.cm.get_cmap('plasma', n_states)
    plt.plot(times, gfp, color='black', linewidth=1)
    for state, color in zip(range(n_states), cmap.colors):
        plt.fill_between(times, gfp, color=color,
                         where=(segmentation == state))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_states)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.yticks([])
    if info is not None and "sfreq" in info.keys():
        plt.xlabel('Time (s)')
    else:
        plt.xlabel('Sample')
    plt.title('Sequence of %d microstates' % n_states)
    plt.autoscale(tight=True)
    plt.tight_layout()