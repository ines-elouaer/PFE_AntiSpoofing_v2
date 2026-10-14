import os
import cv2
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


def mp_image_from_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image(image_format=ImageFormat.SRGB, data=img_rgb)


class FaceContourSegmenter:
    
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109
    ]

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

    def segment(self, img_bgr, out_size=(224, 224), pad=8, feather=5):
        
        h, w = img_bgr.shape[:2]

        mp_img = mp_image_from_bgr(img_bgr)
        result = self.detector.detect(mp_img)

        if not result.face_landmarks:
            return None, None

        lm = result.face_landmarks[0] 

  
        pts = []
        for idx in self.FACE_OVAL:
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)

       
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

       
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None, None

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)

        img_crop = img_bgr[y1:y2 + 1, x1:x2 + 1]
        mask_crop = mask[y1:y2 + 1, x1:x2 + 1]

      
        if feather and feather > 0:
            k = feather * 2 + 1
            mask_crop = cv2.GaussianBlur(mask_crop, (k, k), 0)

       
        face_only = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)

        face_only = cv2.resize(face_only, out_size, interpolation=cv2.INTER_AREA)
        mask_crop = cv2.resize(mask_crop, out_size, interpolation=cv2.INTER_AREA)

        return face_only, mask_crop
