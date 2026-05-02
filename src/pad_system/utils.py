import mimetypes
import re
from pathlib import Path


def detect_input_type(file_path):
    """
    Détecte le type d'entrée pour le système PAD.

    Cas supportés :
    - image réelle : .jpg, .jpeg, .png, .bmp, .webp
    - vidéo brute : .avi, .mp4, .mov, .mkv
    - video_id CASIA sans extension : 13_1, 10_3, 29_2, etc.
    """

    text = str(file_path).strip()
    lower_text = text.lower()
    suffix = Path(lower_text).suffix.lower()

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    video_exts = [".avi", ".mp4", ".mov", ".mkv"]

    # Cas 1 : extension image
    if suffix in image_exts:
        return "image"

    # Cas 2 : extension vidéo
    if suffix in video_exts:
        return "video"

    # Cas 3 : video_id CASIA sans extension
    # Exemple : 13_1, 10_3, 29_2
    if re.fullmatch(r"\d+_\d+", lower_text):
        return "video"

    # Cas 4 : fallback mimetype
    mime_type, _ = mimetypes.guess_type(text)

    if mime_type is None:
        return "unknown"

    if mime_type.startswith("image"):
        return "image"

    if mime_type.startswith("video"):
        return "video"

    return "unknown"