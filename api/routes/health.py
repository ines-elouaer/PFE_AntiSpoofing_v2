from fastapi import APIRouter

from api.config import settings
from api.schemas import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Vérifie que l'API est disponible.
    """
    return HealthResponse(
        status="ok",
        service=settings.API_NAME,
        version=settings.API_VERSION,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """
    Retourne les informations du modèle final utilisé par l'API.
    """
    return ModelInfoResponse(
        model_name=settings.MODEL_NAME,
        api_version=settings.API_VERSION,
        video_checkpoint=str(settings.VIDEO_CHECKPOINT),
        behavior_pose_model=str(settings.BEHAVIOR_POSE_MODEL),
        face_landmarker_model=str(settings.FACE_LANDMARKER_MODEL),
        fusion_formula=settings.FUSION_FORMULA,
        decisions=["ACCEPT", "RETRY", "REJECT"],
    )