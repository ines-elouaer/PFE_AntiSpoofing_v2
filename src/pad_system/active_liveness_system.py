from pathlib import Path
from typing import Dict, Any, Optional, List
import math
import threading

import cv2
import numpy as np


class ActiveLivenessSystem:
    """
    Vérification active légère des challenges :
    - BLINK
    - TURN_LEFT
    - SMILE
    - EYEBROW_RAISE

    TURN_RIGHT a été retiré car il générait souvent des instabilités
    de pose en runtime webcam.
    """

    SUPPORTED_CHALLENGES = {
        "BLINK",
        "TURN_LEFT",
        "SMILE",
        "EYEBROW_RAISE",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_face_rate: float = 0.50,
        sample_every: int = 2,
    ):
        self.project_root = Path(__file__).resolve().parents[2]

        if model_path is None:
            model_path = self.project_root / "models" / "face_landmarker.task"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"FaceLandmarker introuvable: {model_path}")

        self.model_path = str(model_path)
        self.min_face_rate = float(min_face_rate)
        self.sample_every = max(1, int(sample_every))

        # Important pour MediaPipe en mode VIDEO :
        # les timestamps doivent être strictement croissants entre les appels.
        self._timestamp_lock = threading.Lock()
        self._last_timestamp_ms = 0

        import mediapipe as mp

        self.mp = mp
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

        print(f"[LIVENESS] FaceLandmarker chargé: {self.model_path}")

    # ==========================================================
    # UTILS
    # ==========================================================

    @staticmethod
    def _dist(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _next_timestamp_ms(self, step_ms: int = 33) -> int:
        """
        Génère un timestamp strictement croissant pour MediaPipe.

        Pourquoi ?
        L'API garde ActiveLivenessSystem en mémoire.
        Donc si une vidéo commence à 0 ms après une autre vidéo,
        MediaPipe peut lever :
        "Input timestamp must be monotonically increasing".
        """
        step_ms = max(1, int(step_ms))

        with self._timestamp_lock:
            self._last_timestamp_ms += step_ms
            return self._last_timestamp_ms

    def _eye_ear(self, landmarks, idxs: List[int]) -> float:
        """
        EAR approximatif avec indices MediaPipe FaceMesh.
        idxs = [p1, p2, p3, p4, p5, p6]
        """
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idxs]

        vertical_1 = self._dist(p2, p6)
        vertical_2 = self._dist(p3, p5)
        horizontal = self._dist(p1, p4)

        if horizontal <= 1e-6:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _extract_smile_score(self, result) -> float:
        """
        Utilise les blendshapes MediaPipe si disponibles.
        """
        try:
            if not result.face_blendshapes:
                return 0.0

            categories = result.face_blendshapes[0]
            scores = {}

            for c in categories:
                scores[c.category_name] = c.score

            left = scores.get("mouthSmileLeft", 0.0)
            right = scores.get("mouthSmileRight", 0.0)

            return float((left + right) / 2.0)

        except Exception:
            return 0.0

    def _extract_eyebrow_score(self, landmarks) -> float:
        """
        Score simple pour EYEBROW_RAISE.

        Principe :
        - On mesure la distance verticale entre sourcils et yeux.
        - Si l'utilisateur lève les sourcils, cette distance augmente.
        - On normalise par la largeur du visage.
        """
        try:
            left_brow = landmarks[70]
            right_brow = landmarks[300]

            left_eye = landmarks[159]
            right_eye = landmarks[386]

            left_face = landmarks[33]
            right_face = landmarks[263]

            left_dist = abs(left_eye.y - left_brow.y)
            right_dist = abs(right_eye.y - right_brow.y)

            face_width = abs(right_face.x - left_face.x) + 1e-6

            score = ((left_dist + right_dist) / 2.0) / face_width

            return float(score)

        except Exception:
            return 0.0

    # ==========================================================
    # MAIN ANALYZE
    # ==========================================================

    def analyze(
        self,
        video_path: str,
        challenge: str,
    ) -> Dict[str, Any]:
        challenge = str(challenge).upper().strip()

        if challenge not in self.SUPPORTED_CHALLENGES:
            return {
                "status": "FAIL",
                "passed": False,
                "challenge": challenge,
                "reason": "unsupported_challenge",
                "metrics": {},
            }

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return {
                "status": "FAIL",
                "passed": False,
                "challenge": challenge,
                "reason": "video_not_readable",
                "metrics": {"video_path": str(video_path)},
            }

        total_frames = 0
        processed_frames = 0
        detected_frames = 0

        ear_values = []
        nose_x_values = []
        smile_scores = []
        eyebrow_scores = []

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            total_frames += 1

            if frame_idx % self.sample_every != 0:
                frame_idx += 1
                continue

            processed_frames += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=rgb,
            )

            # Correction importante :
            # timestamp global strictement croissant, même entre plusieurs vidéos.
            timestamp_ms = self._next_timestamp_ms(step_ms=33)

            try:
                result = self.landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            except Exception as e:
                cap.release()
                return {
                    "status": "ERROR",
                    "passed": False,
                    "challenge": challenge,
                    "reason": f"liveness_exception: {str(e)}",
                    "metrics": {
                        "total_frames": total_frames,
                        "processed_frames": processed_frames,
                    },
                }

            if result.face_landmarks:
                detected_frames += 1
                landmarks = result.face_landmarks[0]

                try:
                    left_ear = self._eye_ear(landmarks, left_eye)
                    right_ear = self._eye_ear(landmarks, right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    ear_values.append(float(ear))
                except Exception:
                    pass

                try:
                    nose = landmarks[1]

                    xs = [p.x for p in landmarks]
                    face_min_x = min(xs)
                    face_max_x = max(xs)
                    face_center_x = (face_min_x + face_max_x) / 2.0
                    face_width = max(face_max_x - face_min_x, 1e-6)

                    normalized_nose_shift = (nose.x - face_center_x) / face_width
                    nose_x_values.append(float(normalized_nose_shift))
                except Exception:
                    pass

                try:
                    smile_scores.append(self._extract_smile_score(result))
                except Exception:
                    pass

                try:
                    eyebrow_scores.append(self._extract_eyebrow_score(landmarks))
                except Exception:
                    pass

            frame_idx += 1

        cap.release()

        face_rate = detected_frames / processed_frames if processed_frames > 0 else 0.0

        metrics = {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "detected_frames": detected_frames,
            "face_rate": round(face_rate, 4),
        }

        if ear_values:
            ear_arr = np.array(ear_values, dtype=np.float32)
            metrics.update({
                "ear_min": round(float(ear_arr.min()), 4),
                "ear_mean": round(float(ear_arr.mean()), 4),
                "blink_closed_frames": int((ear_arr < 0.18).sum()),
            })

        if nose_x_values:
            nose_arr = np.array(nose_x_values, dtype=np.float32)
            metrics.update({
                "nose_shift_min": round(float(nose_arr.min()), 4),
                "nose_shift_max": round(float(nose_arr.max()), 4),
                "nose_shift_range": round(float(nose_arr.max() - nose_arr.min()), 4),
            })

        if smile_scores:
            smile_arr = np.array(smile_scores, dtype=np.float32)
            metrics.update({
                "smile_score_max": round(float(smile_arr.max()), 4),
                "smile_score_mean": round(float(smile_arr.mean()), 4),
            })

        if eyebrow_scores:
            eyebrow_arr = np.array(eyebrow_scores, dtype=np.float32)

            eyebrow_baseline = float(np.percentile(eyebrow_arr, 25))
            eyebrow_peak = float(np.percentile(eyebrow_arr, 90))
            eyebrow_amplitude = eyebrow_peak - eyebrow_baseline

            metrics.update({
                "eyebrow_score_min": round(float(eyebrow_arr.min()), 4),
                "eyebrow_score_max": round(float(eyebrow_arr.max()), 4),
                "eyebrow_score_mean": round(float(eyebrow_arr.mean()), 4),
                "eyebrow_baseline": round(eyebrow_baseline, 4),
                "eyebrow_peak": round(eyebrow_peak, 4),
                "eyebrow_amplitude": round(float(eyebrow_amplitude), 4),
            })

        if processed_frames == 0:
            return {
                "status": "FAIL",
                "passed": False,
                "challenge": challenge,
                "reason": "no_processed_frame",
                "metrics": metrics,
            }

        if face_rate < self.min_face_rate:
            return {
                "status": "FAIL",
                "passed": False,
                "challenge": challenge,
                "reason": "face_detection_rate_too_low",
                "metrics": metrics,
            }

        # ======================================================
        # Challenge BLINK
        # ======================================================

        if challenge == "BLINK":
            blink_closed_frames = metrics.get("blink_closed_frames", 0)
            ear_min = metrics.get("ear_min", 1.0)

            passed = blink_closed_frames >= 2 and ear_min < 0.18

            return {
                "status": "PASS" if passed else "FAIL",
                "passed": bool(passed),
                "challenge": challenge,
                "reason": "blink_passed" if passed else "blink_not_detected",
                "metrics": metrics,
            }

        # ======================================================
        # Challenge TURN_LEFT
        # ======================================================

        if challenge == "TURN_LEFT":
            nose_range = metrics.get("nose_shift_range", 0.0)
            nose_min = metrics.get("nose_shift_min", 0.0)
            nose_max = metrics.get("nose_shift_max", 0.0)

            passed = (
                nose_range >= 0.06
                or abs(nose_min) >= 0.06
                or abs(nose_max) >= 0.06
            )

            return {
                "status": "PASS" if passed else "FAIL",
                "passed": bool(passed),
                "challenge": challenge,
                "reason": "head_turn_passed" if passed else "head_turn_not_detected",
                "metrics": metrics,
            }

        # ======================================================
        # Challenge SMILE
        # ======================================================

        if challenge == "SMILE":
            smile_max = metrics.get("smile_score_max", 0.0)
            passed = smile_max >= 0.20

            return {
                "status": "PASS" if passed else "FAIL",
                "passed": bool(passed),
                "challenge": challenge,
                "reason": "smile_passed" if passed else "smile_not_detected",
                "metrics": metrics,
            }

        # ======================================================
        # Challenge EYEBROW_RAISE
        # ======================================================

        if challenge == "EYEBROW_RAISE":
            eyebrow_amplitude = metrics.get("eyebrow_amplitude", 0.0)
            eyebrow_peak = metrics.get("eyebrow_peak", 0.0)

            passed = eyebrow_amplitude >= 0.007 and eyebrow_peak >= 0.055

            return {
                "status": "PASS" if passed else "FAIL",
                "passed": bool(passed),
                "challenge": challenge,
                "reason": (
                    "eyebrow_raise_passed"
                    if passed
                    else "eyebrow_raise_not_detected"
                ),
                "metrics": metrics,
            }

        return {
            "status": "FAIL",
            "passed": False,
            "challenge": challenge,
            "reason": "unknown_error",
            "metrics": metrics,
        }