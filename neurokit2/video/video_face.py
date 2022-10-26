import numpy as np

from ..misc import progress_bar


def video_face(video, verbose=True):
    """**Extract face from video**

    This function extracts the faces from a video. This function requires the `cv2, `menpo` and
    `menpodetect` modules to be installed.

    .. note::

        This function is experimental. If you are interested in helping us improve that aspect of
        NeuroKit (e.g., by adding more detection algorithms), please get in touch!

    Parameters
    ----------
    video : np.ndarray
        An video data numpy array of the shape (frame, channel, height, width)
    verbose : bool
        Whether to print the progress bar.

    Returns
    -------
    list
        A list of cropped faces.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # video, sampling_rate = nk.read_video("video.mp4")
      # faces = nk.video_face(video)
      # nk.video_plot([video, faces])

    """

    faceboxes = np.full([len(video), 3, 500, 500], 0)
    for i, frame in progress_bar(video, verbose=verbose):
        faces = _video_face_landmarks(frame)
        if len(faces) > 0:
            faceboxes[i, :, :, :] = _video_face_crop(frame, faces[0])
    return faceboxes.astype("uint8")


# ==============================================================================
# Internals
# ==============================================================================
def _video_face_crop(frame, face):

    # Try loading cv2
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "The 'cv2' module is required for this function to run. ",
            "Please install it first (`pip install opencv-python`).",
        )

    facebox = face.as_vector().reshape(-1, 2).astype(int)

    # Crop
    img = frame[:, facebox[0, 0] : facebox[1, 0], facebox[0, 1] : facebox[2, 1]]

    # Resize
    img = cv2.resize(img.swapaxes(0, 1).swapaxes(1, 2), (500, 500))
    return img.swapaxes(0, 2).swapaxes(1, 2).astype(int)


def _video_face_landmarks(frame):

    # Try loading menpo
    try:
        import menpo.io
        import menpo.landmark
        import menpodetect
    except ImportError:
        raise ImportError(
            "The 'menpo' and 'menpodetect' modules are required for this function to run. ",
            "Please install them first (`pip install menpo` and `pip install menpodetect`).",
        )

    img = menpo.image.Image(frame, copy=True)

    img_bw = img.as_greyscale()

    # Face detection
    faces = menpodetect.load_opencv_frontal_face_detector()(img_bw)

    # Eyes detection
    # eyes = menpodetect.load_opencv_eye_detector()(img_bw)

    return faces
