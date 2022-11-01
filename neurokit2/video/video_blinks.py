# !!!!!!!!!!!!!!!!!!!!!!!!
# ! NEED HELP WITH THAT  !
# !!!!!!!!!!!!!!!!!!!!!!!!

# import numpy as np

# from ..misc import progress_bar


# def video_blinks(video, verbose=True):
#     """**Extract blinks from video**"""

#     # Try loading menpo
#     try:
#         import cv2
#         import menpo.io
#         import menpo.landmark
#         import menpodetect
#     except ImportError:
#         raise ImportError(
#             "The 'menpo' and 'menpodetect' modules are required for this function to run. ",
#             "Please install them first (`pip install menpo` and `pip install menpodetect`).",
#         )

#     frame = video[0]
#     # 1. Extract faces
#     faces = nk.video_face(video, verbose=False)
#     face = faces[0]
#     for i, face in enumerate(faces):
#         img = menpo.image.Image(face, copy=True)
#         img_bw = img.as_greyscale()

#     # Eyes detection
#     eyes = menpodetect.load_opencv_eye_detector()(img_bw)

#     img_bw.view()
#     eye.view(line_width=1, render_markers=False, line_colour="r")
#     for eye in eyes:
#         eye.view(line_width=1, render_markers=False, line_colour="r")


# def detect_pupil(img_bw):
#     """
#     This method should use cv2.findContours and cv2.HoughCircles() function from cv2 library to find the pupil
#     and then set the coordinates for pupil circle coordinates
#     """
#     # as array
#     img = img_bw.as_vector().reshape(img_bw.shape).copy()
#     # First binarize the image so that findContours can work correctly.
#     menpo.image.Image(img, copy=True).view()
#     img[img >= 100] = 255
#     img[img < 100] = 0

#     # Now find the contours and then find the pupil in the contours.
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     # Make a copy image of the original and then use the drawContours function to actually apply
#     # the contours found in the previous step
#     img_with_contours = np.copy(img)
#     cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0))
#     c = cv2.HoughCircles(
#         img_with_contours, cv2.HOUGH_GRADIENT, 2, self._img.shape[0] / 2, maxRadius=150
#     )
#     # Then mask the pupil from the image and store it's coordinates.
#     for l in c:
#         # OpenCV returns the circles as a list of lists of circles
#         for circle in l:
#             center = (int(circle[0]), int(circle[1]))
#             radius = int(circle[2])
#             cv2.circle(self._img, center, radius, (0, 0, 0), thickness=-1)
#             pupil = (center[0], center[1], radius)
