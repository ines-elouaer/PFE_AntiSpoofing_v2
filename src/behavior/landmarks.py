import os
import cv2
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1


def mp_image_from_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image(image_format=ImageFormat.SRGB, data=img_rgb)


def euclid(a, b):
    return float(np.linalg.norm(a - b))


def ear_from_landmarks(lm, eye_idx, w, h):
    pts = []
    for idx in eye_idx:
        x = lm[idx].x * w
        y = lm[idx].y * h
        pts.append(np.array([x, y], dtype=np.float32))

    p1, p2, p3, p4, p5, p6 = pts
    denom = 2.0 * euclid(p1, p4)
    if denom < 1e-6:
        return None

    ear = (euclid(p2, p6) + euclid(p3, p5)) / denom
    return float(ear)


def count_blinks(ears, thr_low, thr_high):
    if len(ears) < 3:
        return 0

    state = "OPEN"
    blinks = 0
    for e in ears:
        if state == "OPEN":
            if e < thr_low:
                state = "CLOSED"
        else:
            if e > thr_high:
                blinks += 1
                state = "OPEN"
    return blinks


class FaceLandmarkerHelper:

    def __init__(self, model_path="models/face_landmarker.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, img_bgr):
        mp_img = mp_image_from_bgr(img_bgr)
        res = self.detector.detect(mp_img)
        if not res.face_landmarks:
            return None
        return res.face_landmarks[0]


def extract_ear_and_motion_from_video(video_path, landmarker, every_n=1, max_frames=600):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    ears = []
    motions = []
    prev_xy = None
    skipped = 0
    used = 0
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % every_n == 0:
            lm = landmarker.detect_landmarks(frame)
            if lm is None:
                skipped += 1
            else:
                h, w = frame.shape[:2]

                ear_l = ear_from_landmarks(lm, LEFT_EYE, w, h)
                ear_r = ear_from_landmarks(lm, RIGHT_EYE, w, h)
                if ear_l is not None and ear_r is not None:
                    ears.append((ear_l + ear_r) / 2.0)

                x = lm[NOSE_TIP].x * w
                y = lm[NOSE_TIP].y * h
                xy = np.array([x, y], dtype=np.float32)

                if prev_xy is not None:
                    motions.append(float(np.linalg.norm(xy - prev_xy)))
                prev_xy = xy

            used += 1
            if used >= max_frames:
                break

        i += 1

    cap.release()
    return np.array(ears, dtype=np.float32), np.array(motions, dtype=np.float32), float(fps), int(skipped)


def video_features_from_signals(ears, motions):
   
    if len(ears) == 0:
        ear_mean    = 0.0
        ear_std     = 0.0
        ear_min     = 0.0
        ear_max     = 0.0
        blink_count = 0
    else:
        ear_mean = float(np.mean(ears))
        ear_std  = float(np.std(ears))
        ear_min  = float(np.min(ears))
        ear_max  = float(np.max(ears))
        ear_std_safe = max(ear_std, 0.01)
        thr_low  = ear_mean - 0.8 * ear_std_safe
        thr_high = ear_mean - 0.2 * ear_std_safe
        blink_count = int(count_blinks(ears, thr_low, thr_high))

    if len(motions) == 0:
        motion_mean = 0.0
        motion_std  = 0.0
        motion_max  = 0.0
    else:
        motion_mean = float(np.mean(motions))
        motion_std  = float(np.std(motions))
        motion_max  = float(np.max(motions))

    return {
        "ear_mean":    ear_mean,
        "ear_std":     ear_std,       
        "ear_min":     ear_min,
        "ear_max":     ear_max,
        "blink_count": blink_count,
        "motion_mean": motion_mean,
        "motion_std":  motion_std,
        "motion_max":  motion_max,
    }