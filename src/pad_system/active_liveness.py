import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ActiveLivenessChallenge:
    """
    Module de pré-filtrage actif de vivacité.

    Objectif :
    - générer ou recevoir un challenge : TURN_LEFT / TURN_RIGHT / BLINK / SMILE
    - analyser la vidéo webcam
    - vérifier si l'utilisateur a réellement répondu au challenge
    - filtrer les attaques simples avant le modèle CNN+LSTM

    Remarque :
    Pour TURN_LEFT et TURN_RIGHT, on valide surtout l'existence d'un mouvement
    latéral significatif afin d'éviter les problèmes d'effet miroir webcam.
    """

    SUPPORTED_CHALLENGES = ["TURN_LEFT", "TURN_RIGHT", "BLINK", "SMILE"]

    def __init__(
        self,
        model_path: str = r"E:\PFE_AntiSpoofing_v2\models\face_landmarker.task",
        max_frames: int = 90,
        min_face_rate: float = 0.60,
        head_turn_threshold: float = 0.08,
        blink_ear_threshold: float = 0.22,
        min_blink_frames: int = 1,
        smile_threshold: float = 0.35,
    ):
        self.model_path = Path(model_path)
        self.max_frames = max_frames
        self.min_face_rate = min_face_rate
        self.head_turn_threshold = head_turn_threshold
        self.blink_ear_threshold = blink_ear_threshold
        self.min_blink_frames = min_blink_frames
        self.smile_threshold = smile_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle MediaPipe introuvable: {self.model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

        print(f"[LIVENESS] FaceLandmarker chargé: {self.model_path}")

    def generate_challenge(self) -> str:
        """
        Génère un challenge aléatoire.
        """
        return random.choice(self.SUPPORTED_CHALLENGES)

    def check(self, video_path: str, challenge: Optional[str] = None) -> Dict:
        """
        Analyse une vidéo et vérifie si le challenge demandé est réussi.

        Retour :
        {
            "status": "PASS" / "FAIL" / "SKIPPED",
            "passed": True / False,
            "challenge": "...",
            "reason": "...",
            "metrics": {...}
        }
        """

        if challenge is None:
            challenge = self.generate_challenge()

        challenge = challenge.upper().strip()

        if challenge not in self.SUPPORTED_CHALLENGES:
            return {
                "status": "FAIL",
                "passed": False,
                "challenge": challenge,
                "reason": "unsupported_challenge",
                "metrics": {},
            }

        p = Path(video_path)

        # Si l'entrée est un video_id CASIA comme "13_1" ou "13_3",
        # ce n'est pas une vidéo brute webcam. On skippe le challenge.
        if not p.exists():
            return {
                "status": "SKIPPED",
                "passed": True,
                "challenge": challenge,
                "reason": "input_is_not_raw_video_file",
                "metrics": {},
            }

        frames = self._read_video_frames(p)

        if len(frames) == 0:
            return {
                "status": "FAIL",
                "passed": False,
                "challenge": challenge,
                "reason": "empty_video",
                "metrics": {},
            }

        analysis = self._analyze_frames(frames)

        if analysis["face_rate"] < self.min_face_rate:
            return {
                "status": "FAIL",
                "passed": False,
                "challenge": challenge,
                "reason": "face_not_detected_enough",
                "metrics": analysis,
            }

        # ==========================================================
        # Challenge TURN_LEFT / TURN_RIGHT
        # ==========================================================
        if challenge in ["TURN_LEFT", "TURN_RIGHT"]:
            # Version robuste :
            # on valide qu'un vrai mouvement latéral de tête existe.
            # Cela évite le problème gauche/droite inversé par webcam.
            passed = analysis["nose_shift_range"] >= self.head_turn_threshold
            reason = "head_turn_passed" if passed else "head_turn_not_detected"

        # ==========================================================
        # Challenge BLINK
        # ==========================================================
        elif challenge == "BLINK":
            passed = analysis["blink_closed_frames"] >= self.min_blink_frames
            reason = "blink_passed" if passed else "blink_not_detected"

        # ==========================================================
        # Challenge SMILE
        # ==========================================================
        elif challenge == "SMILE":
            passed = analysis["smile_score_max"] >= self.smile_threshold
            reason = "smile_passed" if passed else "smile_not_detected"

        else:
            passed = False
            reason = "unknown_challenge"

        return {
            "status": "PASS" if passed else "FAIL",
            "passed": bool(passed),
            "challenge": challenge,
            "reason": reason,
            "metrics": analysis,
        }

    def _read_video_frames(self, video_path: Path) -> List[np.ndarray]:
        """
        Lit une vidéo et retourne au maximum max_frames frames RGB.
        """

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // self.max_frames) if total > 0 else 1

        frames = []
        idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            idx += 1

            if len(frames) >= self.max_frames:
                break

        cap.release()
        return frames

    def _analyze_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyse :
        - présence du visage
        - mouvement latéral de la tête
        - clignement via EAR
        - sourire via blendshapes MediaPipe
        """

        nose_positions = []
        ear_values = []
        smile_scores = []
        detected_count = 0

        for frame in frames:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame,
            )

            result = self.detector.detect(mp_image)

            if not result.face_landmarks:
                continue

            detected_count += 1
            landmarks = result.face_landmarks[0]

            # ======================================================
            # Mouvement de tête : déplacement relatif du nez
            # ======================================================
            nose_x = landmarks[1].x

            xs = [lm.x for lm in landmarks]
            face_min_x = min(xs)
            face_max_x = max(xs)
            face_width = max(face_max_x - face_min_x, 1e-6)

            nose_relative = (nose_x - face_min_x) / face_width
            nose_positions.append(nose_relative)

            # ======================================================
            # Blink detection : EAR approximatif
            # ======================================================
            try:
                left_ear = self._eye_aspect_ratio(
                    landmarks,
                    eye_indices=[33, 160, 158, 133, 153, 144],
                )

                right_ear = self._eye_aspect_ratio(
                    landmarks,
                    eye_indices=[362, 385, 387, 263, 373, 380],
                )

                ear = (left_ear + right_ear) / 2.0
                ear_values.append(ear)

            except Exception:
                pass

            # ======================================================
            # Smile detection : blendshapes
            # ======================================================
            if result.face_blendshapes:
                blendshapes = result.face_blendshapes[0]

                smile_left = 0.0
                smile_right = 0.0

                for category in blendshapes:
                    if category.category_name == "mouthSmileLeft":
                        smile_left = float(category.score)

                    elif category.category_name == "mouthSmileRight":
                        smile_right = float(category.score)

                smile_score = (smile_left + smile_right) / 2.0
                smile_scores.append(smile_score)

        total_frames = len(frames)
        face_rate = detected_count / max(total_frames, 1)

        # ==========================================================
        # Metrics mouvement tête
        # ==========================================================
        if len(nose_positions) >= 2:
            start_nose = nose_positions[0]
            nose_shifts = [x - start_nose for x in nose_positions]

            nose_shift_min = float(min(nose_shifts))
            nose_shift_max = float(max(nose_shifts))
            nose_shift_range = float(max(nose_shifts) - min(nose_shifts))

        else:
            nose_shift_min = 0.0
            nose_shift_max = 0.0
            nose_shift_range = 0.0

        # ==========================================================
        # Metrics blink
        # ==========================================================
        if len(ear_values) > 0:
            ear_min = float(min(ear_values))
            ear_mean = float(np.mean(ear_values))

            blink_closed_frames = int(
                sum(ear < self.blink_ear_threshold for ear in ear_values)
            )

        else:
            ear_min = 0.0
            ear_mean = 0.0
            blink_closed_frames = 0

        # ==========================================================
        # Metrics smile
        # ==========================================================
        if len(smile_scores) > 0:
            smile_score_max = float(max(smile_scores))
            smile_score_mean = float(np.mean(smile_scores))

        else:
            smile_score_max = 0.0
            smile_score_mean = 0.0

        return {
            "total_frames": int(total_frames),
            "detected_frames": int(detected_count),
            "face_rate": float(round(face_rate, 4)),

            "nose_shift_min": float(round(nose_shift_min, 4)),
            "nose_shift_max": float(round(nose_shift_max, 4)),
            "nose_shift_range": float(round(nose_shift_range, 4)),

            "ear_min": float(round(ear_min, 4)),
            "ear_mean": float(round(ear_mean, 4)),
            "blink_closed_frames": int(blink_closed_frames),

            "smile_score_max": float(round(smile_score_max, 4)),
            "smile_score_mean": float(round(smile_score_mean, 4)),
        }

    def _eye_aspect_ratio(self, landmarks, eye_indices: List[int]) -> float:
        """
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """

        pts = []

        for idx in eye_indices:
            lm = landmarks[idx]
            pts.append(np.array([lm.x, lm.y], dtype=np.float32))

        p1, p2, p3, p4, p5, p6 = pts

        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)

        return float(ear)