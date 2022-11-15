import numpy as np

from ..misc import progress_bar
from .video_face import video_face
from .video_skin import video_skin


def video_ppg(video, sampling_rate=30, verbose=True):
    """**Remote Photoplethysmography (rPPG) from Video**

    Extracts the photoplethysmogram (PPG) from a webcam video using the Plane-Orthogonal-to-Skin
    (POS) algorithm.

    .. note::

        This function is experimental and does NOT seem to work at all
        (https://github.com/DominiqueMakowski/RemotePhysiology). If you
        are interested in helping us improve that aspect of NeuroKit
        (e.g., by adding more detection algorithms), please get in touch!

    Parameters
    ----------
    video : np.ndarray
        A video data numpy array of the shape (frame, channel, height, width).
    sampling_rate : int
        The sampling rate of the video, by default 30 fps (a common sampling rate for commercial
        webcams).
    verbose : bool
        Whether to print the progress bar.

    Returns
    -------
    np.ndarray
        A PPG signal.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # video, sampling_rate = nk.read_video("video.mp4")
      # ppg = nk.video_ppg(video)

    References
    ----------
    * Wang, W., Den Brinker, A. C., Stuijk, S., & De Haan, G. (2016). Algorithmic principles of
      remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.

    """
    # Initialize heart rate
    ppg = np.full((len(video)), np.nan)

    # Chunk into 8 second segments (5 * 1.6 which is the temporal smoothing window)
    chunk_size = int(sampling_rate * 8)
    for _, start in progress_bar(np.arange(0, len(video), chunk_size), verbose=verbose):
        end = start + chunk_size
        if end > len(video):
            end = len(video)
        ppg[start:end] = _video_ppg(video[start:end, :, :, :], sampling_rate, window=1.6)
    return ppg


# ==============================================================================
# Internals
# ==============================================================================
def _video_ppg(video, sampling_rate=30, window=1.6):
    # 1. Extract faces
    faces = video_face(video, verbose=False)

    rgb = np.full((len(faces), 3), np.nan)
    for i, face in enumerate(faces):
        # 2. Extract skin
        mask, masked_face = video_skin(face)

        # Extract color
        r = np.sum(masked_face[:, :, 0]) / np.sum(mask > 0)
        g = np.sum(masked_face[:, :, 1]) / np.sum(mask > 0)
        b = np.sum(masked_face[:, :, 2]) / np.sum(mask > 0)
        rgb[i, :] = [r, g, b]

    # Plane-Orthogonal-to-Skin (POS)
    # ==============================
    # Calculating window (l)
    window = int(sampling_rate * window)
    H = np.full(len(rgb), 0)

    for t in range(0, (rgb.shape[0] - window)):
        # 4. Spatial averaging
        C = rgb[t : t + window - 1, :].T

        # 5. Temporal normalization
        mean_color = np.mean(C, axis=1)
        try:
            Cn = np.matmul(np.linalg.inv(np.diag(mean_color)), C)
        except np.linalg.LinAlgError:  # Singular matrix
            continue

        # 6. Projection
        S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)

        # 7. Tuning (2D signal to 1D signal)
        std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
        P = np.matmul(std, S)

        # 8. Overlap-Adding
        H[t : t + window - 1] = H[t : t + window - 1] + (P - np.mean(P)) / np.std(P)

    return H
