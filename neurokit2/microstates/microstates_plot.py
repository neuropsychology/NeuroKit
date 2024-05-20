# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np


def microstates_plot(microstates, segmentation=None, gfp=None, info=None, epoch=None):
    """**Visualize Microstates**

    Plots the clustered microstates.

    Parameters
    ----------
    microstates : np.ndarray
        The topographic maps of the found unique microstates which has a shape of n_channels x
        n_states, generated from :func:`.microstates_segment`.
    segmentation : array
        For each sample, the index of the microstate to which the sample has been assigned.
        Defaults to ``None``.
    gfp : array
        The range of global field power (GFP) values to visualize. Defaults to ``None``, which will
        plot the whole range of GFP values.
    info : dict
        The dictionary output of :func:`.nk.microstates_segment`. Defaults to ``None``.
    epoch : tuple
        A sub-epoch of GFP to plot in the shape ``(beginning sample, end sample)``.

    Returns
    -------
    fig
        Plot of prototypical microstates maps and GFP across time.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      # Download data
      eeg = nk.mne_data("filt-0-40_raw")
      # Average rereference and band-pass filtering
      eeg = nk.eeg_rereference(eeg, 'average').filter(1, 30, verbose=False)

      # Cluster microstates
      microstates = nk.microstates_segment(eeg, method='kmeans', n_microstates=4)

      @savefig p_microstates_plot1.png scale=100%
      nk.microstates_plot(microstates, epoch=(500, 750))
      @suppress
      plt.close()


    """

    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "The 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        ) from e

    # Try retrieving info
    if isinstance(microstates, dict):
        if info is None and "Info" in microstates.keys():
            info = microstates["Info"]
        if gfp is None and "GFP" in microstates.keys():
            gfp = microstates["GFP"]
        segmentation = microstates["Sequence"]
        microstates = microstates["Microstates"]

    # Sanity checks
    if gfp is None:
        raise ValueError("GFP data must be passed to 'gfp' in order to plot the segmentation.")

    # Prepare figure layout
    n = len(microstates)
    fig, ax = plt.subplot_mosaic([np.arange(n), ["GFP"] * n])

    # Plot topomaps -----------------------------------------------------------
    for i, map in enumerate(microstates):
        _, _ = mne.viz.plot_topomap(map, info, axes=ax[i], ch_type="eeg", show=False)
        ax[i].set_title(f"{i}")

    # Plot GFP ---------------------------------------------------------------
    # Get x-axis
    if info is not None and "sfreq" in info.keys():
        times = np.arange(len(gfp)) / info["sfreq"]
    else:
        times = np.arange(len(gfp))

    # Correct lengths
    if len(segmentation) > len(gfp):
        segmentation = segmentation[0 : len(gfp)]
    if len(segmentation) < len(gfp):
        gfp = gfp[0 : len(segmentation)]

    if epoch is None:
        epoch = (0, len(gfp))

    cmap = plt.get_cmap("plasma").resampled(n)
    # Plot the GFP line above the area
    ax["GFP"].plot(
        times[epoch[0] : epoch[1]], gfp[epoch[0] : epoch[1]], color="black", linewidth=0.5
    )
    # Plot area
    for state, color in zip(range(n), cmap.colors):
        ax["GFP"].fill_between(
            times[epoch[0] : epoch[1]],
            gfp[epoch[0] : epoch[1]],
            color=color,
            where=(segmentation == state)[epoch[0] : epoch[1]],
        )

    # Create legend
    norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=n - 0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax["GFP"])
    ax["GFP"].set_yticks([])
    if info is not None and "sfreq" in info.keys():
        ax["GFP"].set_xlabel("Time (s)")
    else:
        ax["GFP"].set_xlabel("Sample")
    ax["GFP"].set_ylabel("Global Field Power (GFP)")
    ax["GFP"].set_title("Microstates Sequence")
