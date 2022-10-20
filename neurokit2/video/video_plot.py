# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def video_plot(video, sampling_rate=30, frames=3, signals=None):
    """**Visualize video**

    This function plots a few frames from a video as an image.

    Parameters
    ----------
    video : np.ndarray
        An video data numpy array of the shape (frame, channel, height, width)
    sampling_rate : int
        The number of frames per second (FPS), by default 30.
    signals : list
        A list of signals to plot under the videos.

    Examples
    --------
    .. ipython:: python

        import neurokit2 as nk

        # video, sampling_rate = nk.read_video("video.mp4")
        # nk.video_plot(video, sampling_rate=sampling_rate)

    """

    # Put into list if it's not already
    if isinstance(video, list) is False:
        video = [video]
    # How many subplots
    nrows = len(video)
    # Determine the height of each subplot (so that the width is the same)
    # height_ratios = [v.shape[3] / v.shape[2] for v in video]

    if signals is not None:
        if isinstance(signals, list) is False:
            signals = [signals]
        nrows += len(signals)
        # height_ratios += [1] * len(signals)

    # Get x-axis (of the first video)
    length = video[0].shape[0]

    # TODO: height_ratios doesn't work as expected
    fig, ax = plt.subplots(
        nrows=nrows,
        sharex=True,
        # gridspec_kw={"height_ratios": height_ratios},
        constrained_layout=True,
    )

    # Get frame locations
    if isinstance(frames, int):
        frames = np.linspace(0, length - 1, frames).astype(int)

    # For each videos in the list, plot them
    if nrows == 1:
        ax = [ax]  # Otherwise it will make ax[i] non subscritable
    for i, vid in enumerate(video):

        vid = _video_plot_format(vid, frames=frames, desired_length=length)
        ax[i].axis("off")
        ax[i].imshow(vid, aspect="auto")

    if signals is not None:
        for j, signal in enumerate(signals):

            # Make sure the size is correct
            assert (
                len(signal) == length
            ), f"Length if the {j+1} signals is not equal to the video length of the video (length = {length}). Use signal_resample() to get the right size."

            # Plot
            ax[i + j + 1].plot(signal, color="red")

            for frame in frames:
                ax[i + j + 1].axvline(x=frame, color="black", linestyle="--", alpha=0.5)

    # Ticks in seconds
    plt.xticks(
        np.linspace(0, length, 5),
        np.char.mod("%.1f", np.linspace(0, length / sampling_rate, 5)),
    )
    plt.xlabel("Time (s)")


def _video_plot_format(video, frames=[0], desired_length=1000):
    # Try loading cv2
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "The 'cv2' module is required for this function to run. ",
            "Please install it first (`pip install opencv-python`).",
        )

    # (frames, width, height, RGB channels) for matplotlib
    video = video.swapaxes(3, 1).swapaxes(2, 1)

    length = video.shape[0]

    # Concatenate
    vid = np.concatenate(video[frames], axis=1)

    # Rescale
    vid = cv2.resize(
        vid,
        dsize=(desired_length, int(video.shape[1] * desired_length / video.shape[2] / len(frames))),
        interpolation=cv2.INTER_CUBIC,
    )

    return vid
