# -*- coding: utf-8 -*-
import os

import numpy as np


def read_video(filename="video.mp4"):
    """**Reads a video file into an array**

    Reads a video file (e.g., .mp4) into a numpy array of shape. This function requires OpenCV to
    be installed via the ``opencv-python`` package.

    Parameters
    ----------
    filename : str
        The path of a video file.

    Returns
    -------
    array
        numpy array of shape (frame, RGB-channel, height, width).
    int
        Sampling rate in frames per second.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # video, sampling_rate = nk.read_video("video.mp4")

    """
    # Try loading cv2
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "The 'cv2' module is required for this function to run. ",
            "Please install it first (`pip install opencv-python`).",
        )

    # Check if file exists
    assert os.path.isfile(filename) is True, f"No file found with the specified name ({filename})."

    capture = cv2.VideoCapture(filename)
    sampling_rate = int(capture.get(cv2.CAP_PROP_FPS))

    frames = []
    while capture.isOpened():
        success, frame = capture.read()  # By default frame is BGR
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
    capture.release()
    # Swap axes to be (frame, RGB-channel, height, width)
    return np.array(frames).swapaxes(3, 1).swapaxes(3, 2), sampling_rate
