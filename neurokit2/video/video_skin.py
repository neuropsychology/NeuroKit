import numpy as np

from ..misc import find_closest


def video_skin(face, show=False):
    """**Skin detection**

    This function detects the skin in a face.

    .. note::

        This function is experimental. If you are interested in helping us improve that aspect of
        NeuroKit (e.g., by adding more detection algorithms), please get in touch!

    Parameters
    ----------
    face : np.ndarray
        A face data numpy array of the shape (channel, height, width).
    show : bool
        Whether to show the skin detection mask.

    Returns
    -------
    np.ndarray
        A skin detection mask.

    See Also
    --------
    video_face, video_ppg

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # video, sampling_rate = nk.read_video("video.mp4")
      # faces = nk.video_face(video)
      # skin = nk.video_skin(faces[0], show=True)

    """
    # Try loading cv2
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "The 'cv2' module is required for this function to run. ",
            "Please install it first (`pip install opencv-python`).",
        )

    img = face.swapaxes(0, 1).swapaxes(1, 2)

    # Credits:
    # https://github.com/pavisj/rppg-pos/blob/master/SkinDetector/skin_detector/skin_detector.py

    # Get mask in HSV space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_thresh = np.array([0, 50, 0], dtype="uint8")
    upper_thresh = np.array([120, 150, 255], dtype="uint8")
    mask_hsv = cv2.inRange(img_hsv, lower_thresh, upper_thresh)
    mask_hsv[mask_hsv < 128] = 0
    mask_hsv[mask_hsv >= 128] = 1

    # Get mask in RGB space
    lower_thresh = np.array([45, 52, 108], dtype="uint8")
    upper_thresh = np.array([255, 255, 255], dtype="uint8")
    mask_a = cv2.inRange(img, lower_thresh, upper_thresh)
    mask_b = 255 * ((img[:, :, 2] - img[:, :, 1]) / 20)
    mask_c = 255 * ((np.max(img, axis=2) - np.min(img, axis=2)) / 20)
    mask_d = np.bitwise_and(np.uint64(mask_a), np.uint64(mask_b))
    mask_rgb = np.bitwise_and(np.uint64(mask_c), np.uint64(mask_d))
    mask_rgb[mask_rgb < 128] = 0
    mask_rgb[mask_rgb >= 128] = 1

    # Get mask in YCbCr space
    lower_thresh = np.array([90, 100, 130], dtype="uint8")
    upper_thresh = np.array([230, 120, 180], dtype="uint8")
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    mask_ycrcb = cv2.inRange(img_ycrcb, lower_thresh, upper_thresh)
    mask_ycrcb[mask_ycrcb < 128] = 0
    mask_ycrcb[mask_ycrcb >= 128] = 1

    mask = (mask_hsv + mask_rgb + mask_ycrcb) / 3

    # Get percentages of skin as a function of different thresholds
    threshold = np.arange(0, 1.2, 0.3)
    percent = [np.sum(mask >= t) / mask.size for t in threshold]
    threshold = threshold[find_closest(0.5, percent, return_index=True)]

    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255

    # Process mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Grab and cut
    kernel = np.ones((50, 50), np.float32) / (50 * 50)
    dst = cv2.filter2D(mask, -1, kernel)
    dst[dst != 0] = 255
    free = np.array(cv2.bitwise_not(dst), dtype="uint8")

    grab_mask = np.zeros(mask.shape, dtype="uint8")
    grab_mask[:, :] = 2
    grab_mask[mask == 255] = 1
    grab_mask[free == 255] = 0

    if np.unique(grab_mask).tolist() == [0, 1]:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        if img.size != 0:
            mask, bgdModel, fgdModel = cv2.grabCut(
                img, grab_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
            )
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    mask = mask.astype("uint8")
    masked_face = cv2.bitwise_and(img, img, mask=mask)

    if show is True:
        print(f"{int((100 / 255) * np.sum(mask) / mask.size)}% of the image is skin")
        cv2.imshow("img", cv2.cvtColor(mask.astype("uint8"), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    return mask, masked_face
