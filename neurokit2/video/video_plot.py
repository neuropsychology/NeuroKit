# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def video_plot(video, sampling_rate=30, frames=3, signals=None):
    """**Visualize video**

    This function plots a few frames from a video.

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
        # nk.video_plot(video)

    """

    # Put into list if it's not already
    if isinstance(video, list) is False:
        video = [video]

    # Get x-axis (of the first video)
    length = video[0].shape[0]

    # For each videos in the list, plot them
    fig, ax = plt.subplots(nrows=len(video), sharex=True)
    if len(video) == 1:
        ax = [ax]  # Otherwise it will make ax[i] non subscritable
    for i, vid in enumerate(video):

        vid = _video_plot_format(vid, frames=frames, desired_length=length)

        ax[i].imshow(vid)

    # Ticks in seconds
    plt.xticks(
        np.linspace(0, length, 5),
        np.char.mod("%.1f", np.linspace(0, length / sampling_rate, 5)),
    )
    plt.xlabel("Time (s)")


def _video_plot_format(video, frames=3, desired_length=1000):
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
    idx_frames = np.linspace(0, length - 1, frames).astype(int)

    # Concatenate
    vid = np.concatenate(video[idx_frames], axis=1)

    # Rescale
    vid = cv2.resize(
        vid,
        dsize=(desired_length, int(video.shape[1] * desired_length / video.shape[2] / frames)),
        interpolation=cv2.INTER_CUBIC,
    )

    return vid
