import numpy as np

from .video_face import video_face
from .video_skin import video_skin


def video_ppg(video, sampling_rate=30, **kwargs):

    # 1. Extract faces
    faces = video_face(video, **kwargs)

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
    # Calculating l
    l = int(sampling_rate * 1.6)
    H = np.zeros(rgb.shape[0])

    for t in range(0, (rgb.shape[0] - l)):
        # 4. Spatial averaging
        C = rgb[t : t + l - 1, :].T

        # 5. Temporal normalization
        mean_color = np.mean(C, axis=1)
        Cn = np.matmul(np.linalg.inv(np.diag(mean_color)), C)

        # 6. Projection
        S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)

        # 7. Tuning (2D signal to 1D signal)
        std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
        P = np.matmul(std, S)

        # 8. Overlap-Adding
        H[t : t + l - 1] = H[t : t + l - 1] + (P - np.mean(P)) / np.std(P)

    return H
