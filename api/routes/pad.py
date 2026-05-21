import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import time
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.pad_system.router import PADRouter
from api.config import settings, ensure_runtime_dirs
from api.dependencies import get_pad_router
from api.logging_config import logger


router = APIRouter(prefix="/pad", tags=["PAD Analysis"])


def _safe_video_filename(original_name: str) -> str:
    """
    Génère un nom de fichier sécurisé pour la vidéo uploadée.

    Le navigateur peut envoyer :
    - .mp4
    - .webm
    - .mov
    - .avi
    - .mkv

    Si le format n'est pas accepté, on refuse la requête.
    """

    if not original_name:
        original_name = "video.webm"

    suffix = Path(original_name).suffix.lower()

    if not suffix:
        suffix = ".webm"

    if suffix not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Format vidéo non supporté: {suffix}. "
            f"Formats acceptés: {sorted(settings.ALLOWED_VIDEO_EXTENSIONS)}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"upload_{timestamp}{suffix}"


def _save_upload_file(upload_file: UploadFile) -> Path:
    """
    Sauvegarde la vidéo reçue dans data/runtime_uploads.
    """

    ensure_runtime_dirs()

    safe_name = _safe_video_filename(upload_file.filename or "video.webm")
    destination = settings.RUNTIME_UPLOAD_DIR / safe_name

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

    finally:
        upload_file.file.close()

    if not destination.exists() or destination.stat().st_size == 0:
        raise ValueError("La vidéo uploadée est vide ou invalide.")

    return destination


def _save_result_json(result: dict, video_path: Path) -> Path:
    """
    Sauvegarde le résultat d'analyse dans data/runtime_results.
    """

    ensure_runtime_dirs()

    result_name = video_path.stem.replace("upload_", "result_") + ".json"
    result_path = settings.RUNTIME_RESULTS_DIR / result_name

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result_path


@router.post("/analyze-video")
def analyze_video(
    video: UploadFile = File(...),
    challenge_id: Optional[str] = Form(None),
    enable_liveness: bool = Form(True),
    require_challenge_validation: bool = Form(True),
    pad_router: PADRouter = Depends(get_pad_router),
):
    """
    Analyse une vidéo utilisateur avec le pipeline PAD final.

    Pipeline :
    1. sauvegarde de la vidéo uploadée ;
    2. validation du challenge si demandée ;
    3. liveness actif si activé ;
    4. modèle PAD V6 ;
    5. décision bancaire ACCEPT / RETRY / REJECT.

    Cette version ajoute un runtime_profile pour diagnostiquer
    le temps réel de chaque étape avant Docker.
    """

    api_t0 = time.perf_counter()
    runtime_profile_api = {}

    try:
        # =====================================================
        # 1. Sauvegarde vidéo uploadée
        # =====================================================
        t0 = time.perf_counter()
        saved_video_path = _save_upload_file(video)
        runtime_profile_api["save_upload_sec"] = round(time.perf_counter() - t0, 4)

        logger.info(
            "Vidéo reçue | filename=%s | saved_path=%s | challenge_id=%s | "
            "enable_liveness=%s | require_challenge_validation=%s",
            video.filename,
            saved_video_path,
            challenge_id,
            enable_liveness,
            require_challenge_validation,
        )

        # =====================================================
        # 2. Analyse PAD complète : challenge + liveness + modèle
        # =====================================================
        t0 = time.perf_counter()
        result = pad_router.analyze(
            file_path=str(saved_video_path),
            enable_liveness=enable_liveness,
            challenge_id=challenge_id,
            require_challenge_validation=require_challenge_validation,
        )
        runtime_profile_api["pad_router_analyze_sec"] = round(time.perf_counter() - t0, 4)

        # =====================================================
        # 3. Sauvegarde résultat JSON
        # =====================================================
        result["saved_video_path"] = str(saved_video_path)

        t0 = time.perf_counter()
        saved_result_path = _save_result_json(result, saved_video_path)
        runtime_profile_api["save_result_json_sec"] = round(time.perf_counter() - t0, 4)

        result["saved_result_path"] = str(saved_result_path)

        runtime_profile_api["api_total_sec"] = round(time.perf_counter() - api_t0, 4)

        # On garde aussi le profiling déjà éventuel du router
        result["runtime_profile_api"] = runtime_profile_api

        logger.info(
            "Analyse terminée | decision=%s | score=%s | label=%s | "
            "api_total=%ss | router=%ss | result=%s",
            result.get("decision"),
            result.get("score"),
            result.get("label"),
            runtime_profile_api.get("api_total_sec"),
            runtime_profile_api.get("pad_router_analyze_sec"),
            saved_result_path,
        )

        return result

    except ValueError as exc:
        runtime_profile_api["api_total_sec"] = round(time.perf_counter() - api_t0, 4)
        logger.warning("Requête invalide: %s | profile=%s", str(exc), runtime_profile_api)
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        runtime_profile_api["api_total_sec"] = round(time.perf_counter() - api_t0, 4)
        logger.exception("Erreur interne pendant l'analyse vidéo | profile=%s", runtime_profile_api)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne analyse vidéo: {str(exc)}",
        )