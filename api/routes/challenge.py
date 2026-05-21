from fastapi import APIRouter, Depends, HTTPException

from src.pad_system.router import PADRouter
from api.dependencies import get_pad_router
from api.schemas import ChallengeStartRequest
from api.logging_config import logger

router = APIRouter(prefix="/pad/challenge", tags=["Challenge"])


@router.post("/start")
def start_challenge(
    request: ChallengeStartRequest,
    pad_router: PADRouter = Depends(get_pad_router),
):
    """
    Crée un nouveau challenge vidéo.

    Exemple de challenge :
    - BLINK
    - SMILE
    - TURN_LEFT
    """
    try:
        result = pad_router.create_video_challenge(
            source="api",
            challenge_type=request.challenge_type,
        )
        logger.info(
            "Challenge créé | challenge_id=%s | type=%s",
            result.get("challenge_id"),
            result.get("challenge_type"),
        )
        return result

    except Exception as exc:
        logger.exception("Erreur lors de la création du challenge.")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur création challenge: {str(exc)}",
        )


@router.get("/{challenge_id}")
def get_challenge_status(
    challenge_id: str,
    pad_router: PADRouter = Depends(get_pad_router),
):
    """
    Vérifie l'état d'un challenge :
    - valide ;
    - expiré ;
    - déjà utilisé ;
    - inexistant.
    """
    try:
        result = pad_router.validate_video_challenge(challenge_id)
        return result

    except Exception as exc:
        logger.exception("Erreur lors de la validation du challenge.")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur validation challenge: {str(exc)}",
        )