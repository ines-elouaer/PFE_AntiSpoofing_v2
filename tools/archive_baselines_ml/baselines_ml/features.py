import cv2
import numpy as np
from utils.face_utils_contour import FaceContourSegmenter

_SEG = FaceContourSegmenter("models/face_landmarker.task")

def frame_features(img_bgr):
    face, mask = _SEG.segment(img_bgr, out_size=(224, 224), pad=6, feather=4)
    if face is None:
        return None

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean = float(gray.mean())
    std  = float(gray.std())

    b, g, r = cv2.split(face)
    rb_diff = float(np.mean(r) - np.mean(b))
    rg_diff = float(np.mean(r) - np.mean(g))
    gb_diff = float(np.mean(g) - np.mean(b))

    return np.array([lap_var, mean, std, rb_diff, rg_diff, gb_diff], dtype=np.float32)
