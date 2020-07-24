# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec


def microstates_plot(microstates, segmentation=None, gfp=None, info=None):
    """Plots the clustered microstates.

    Parameters
    ----------
    microstates : np.ndarray
        The topographic maps of the found unique microstates which has a shape of n_channels x n_states,
        generated from ``nk.microstates_segment()``.
    segmentation : array
        For each sample, the index of the microstate to which the sample has been assigned. Defaults to None.
    gfp : array
        The range of global field power (GFP) values to visualize. Defaults to None, which will plot
        the whole range of GFP values.
    info : dict
        The dictionary output of ``nk.microstates_segment()``. Defaults to None.

    Returns
    -------
    fig
        Plot of prototypical microstates maps and GFP across time.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw").filter(1, 35)
    >>> eeg = nk.eeg_rereference(eeg, 'average')
    >>>
    >>> microstates = nk.microstates_segment(eeg, method='kmod')
    >>> nk.microstates_plot(microstates, gfp=microstates["GFP"][0:500]) #doctest: +ELLIPSIS
    <Figure ...>

    """
    # Try retrieving info
    if isinstance(microstates, dict):
        if info is None and "Info" in microstates.keys():
            info = microstates["Info"]
        if gfp is None and "GFP" in microstates.keys():
            gfp = microstates["GFP"]
        segmentation = microstates["Sequence"]
        microstates = microstates["Microstates"]

    # Prepare figure layout
    fig = plt.figure(constrained_layout=False)
    spec = matplotlib.gridspec.GridSpec(ncols=len(microstates), nrows=2)

    ax_bottom = fig.add_subplot(spec[1, :])  # bottom row

    axes_list = []
    for i, _ in enumerate(microstates):
        ax = fig.add_subplot(spec[0, i])
        axes_list.append(ax)

    # Plot
    _microstates_plot_topos(microstates, info=info, ax=axes_list)
    _microstates_plot_segmentation(segmentation, gfp, info, ax=ax_bottom)

    return fig

    pass



def _microstates_plot_topos(microstates, info, ax=None):
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

    # Plot
    if ax is None:
        fig, ax = plt.subplots(ncols=len(microstates), figsize=(2 * len(microstates), 2))
    else:
        fig = None

    for i, map in enumerate(microstates):
        mne.viz.plot_topomap(map, info, axes=ax[i])
        ax[i].set_title('%d' % i)

    return fig


def _microstates_plot_segmentation(segmentation, gfp, info=None, ax=None):
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

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6 * np.ptp(times), 2))
    else:
        fig = None

    n_states = len(np.unique(segmentation))
    cmap = plt.cm.get_cmap('plasma', n_states)
    ax.plot(times, gfp, color='black', linewidth=1)
    for state, color in zip(range(n_states), cmap.colors):
        ax.fill_between(times, gfp, color=color,
                        where=(segmentation == state))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_states)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    ax.set_yticks([])
    if info is not None and "sfreq" in info.keys():
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Sample')
    ax.set_ylabel('Global Field Power (GFP)')
    ax.set_title('Sequence of the %d microstates' % n_states)
    ax.autoscale(tight=True)

    return fig
