from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class InputValidationResult:
    is_valid: bool
    input_type: Optional[str]
    reason: str
    details: Dict[str, Any]

    def to_dict(self):
        return asdict(self)


class PADInputValidator:
    """
    Validation des entrées du système PAD.

    Version adaptée au flux final société :
    - le flux principal accepte uniquement les vidéos challenge ;
    - les images sont détectées mais refusées si allow_images=False ;
    - la validation vérifie le format, la lisibilité, la durée, la résolution
      et le nombre minimal de frames.
    """

    def __init__(
        self,
        allow_images: bool = False,
        min_video_frames: int = 16,
        min_video_duration_sec: float = 1.0,
        min_width: int = 64,
        min_height: int = 64,
    ):
        self.allow_images = allow_images
        self.min_video_frames = min_video_frames
        self.min_video_duration_sec = min_video_duration_sec
        self.min_width = min_width
        self.min_height = min_height

    def detect_input_type(self, file_path: str) -> Optional[str]:
        suffix = Path(file_path).suffix.lower()

        if suffix in IMAGE_EXTS:
            return "image"

        if suffix in VIDEO_EXTS:
            return "video"

        return None

    def validate(self, file_path: str) -> InputValidationResult:
        path = Path(file_path)

        if not path.exists():
            return InputValidationResult(
                is_valid=False,
                input_type=None,
                reason="file_not_found",
                details={"path": str(path)},
            )

        if not path.is_file():
            return InputValidationResult(
                is_valid=False,
                input_type=None,
                reason="not_a_file",
                details={"path": str(path)},
            )

        input_type = self.detect_input_type(str(path))

        if input_type is None:
            return InputValidationResult(
                is_valid=False,
                input_type=None,
                reason="unsupported_format",
                details={
                    "path": str(path),
                    "suffix": path.suffix.lower(),
                    "supported_videos": sorted(list(VIDEO_EXTS)),
                    "image_input": "disabled",
                },
            )

        if input_type == "image" and not self.allow_images:
            return InputValidationResult(
                is_valid=False,
                input_type="image",
                reason="image_input_disabled",
                details={
                    "path": str(path),
                    "message": "Le flux image est désactivé. Utiliser une vidéo challenge.",
                    "supported_videos": sorted(list(VIDEO_EXTS)),
                },
            )

        if input_type == "image":
            return self.validate_image(path)

        if input_type == "video":
            return self.validate_video(path)

        return InputValidationResult(
            is_valid=False,
            input_type=None,
            reason="unknown_input_type",
            details={"path": str(path)},
        )

    def validate_image(self, path: Path) -> InputValidationResult:
        """
        Conservé pour compatibilité éventuelle, mais non utilisé dans le flux final
        lorsque allow_images=False.
        """
        img = cv2.imread(str(path))

        if img is None:
            return InputValidationResult(
                is_valid=False,
                input_type="image",
                reason="image_not_readable_or_corrupted",
                details={"path": str(path)},
            )

        h, w = img.shape[:2]

        if w < self.min_width or h < self.min_height:
            return InputValidationResult(
                is_valid=False,
                input_type="image",
                reason="image_too_small",
                details={
                    "path": str(path),
                    "width": w,
                    "height": h,
                    "min_width": self.min_width,
                    "min_height": self.min_height,
                },
            )

        return InputValidationResult(
            is_valid=True,
            input_type="image",
            reason="valid_image",
            details={
                "path": str(path),
                "width": w,
                "height": h,
                "channels": img.shape[2] if len(img.shape) == 3 else 1,
            },
        )

    def validate_video(self, path: Path) -> InputValidationResult:
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            return InputValidationResult(
                is_valid=False,
                input_type="video",
                reason="video_not_readable_or_corrupted",
                details={"path": str(path)},
            )

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            duration_sec = 0.0
        else:
            duration_sec = frame_count / fps

        cap.release()

        if frame_count <= 0:
            return InputValidationResult(
                is_valid=False,
                input_type="video",
                reason="empty_video",
                details={
                    "path": str(path),
                    "frame_count": frame_count,
                    "fps": fps,
                },
            )

        if frame_count < self.min_video_frames:
            return InputValidationResult(
                is_valid=False,
                input_type="video",
                reason="not_enough_frames",
                details={
                    "path": str(path),
                    "frame_count": frame_count,
                    "min_video_frames": self.min_video_frames,
                    "fps": fps,
                    "duration_sec": duration_sec,
                },
            )

        if duration_sec < self.min_video_duration_sec:
            return InputValidationResult(
                is_valid=False,
                input_type="video",
                reason="video_too_short",
                details={
                    "path": str(path),
                    "duration_sec": duration_sec,
                    "min_video_duration_sec": self.min_video_duration_sec,
                    "frame_count": frame_count,
                    "fps": fps,
                },
            )

        if width < self.min_width or height < self.min_height:
            return InputValidationResult(
                is_valid=False,
                input_type="video",
                reason="video_resolution_too_small",
                details={
                    "path": str(path),
                    "width": width,
                    "height": height,
                    "min_width": self.min_width,
                    "min_height": self.min_height,
                },
            )

        return InputValidationResult(
            is_valid=True,
            input_type="video",
            reason="valid_video",
            details={
                "path": str(path),
                "frame_count": frame_count,
                "fps": fps,
                "duration_sec": duration_sec,
                "width": width,
                "height": height,
            },
        )