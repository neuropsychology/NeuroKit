import matplotlib.pyplot as plt
import numpy as np


def video_face(video):

    """**Extract face from video**"""

    face_centers = np.full([len(video), 2], np.nan)
    for i, frame in enumerate(video):
        faces, eyes = _video_face_landmarks(frame)
        if len(faces) > 0:
            face_centers[i] = faces[0].centre()
        plt.imshow(frame.swapaxes(0, 1).swapaxes(1, 2))
        plt.text(face_centers[i][1], face_centers[i][0], "+")


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
    eyes = menpodetect.load_opencv_eye_detector()(img_bw)

    return faces, eyes
