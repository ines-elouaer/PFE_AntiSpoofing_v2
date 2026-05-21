from functools import lru_cache

from src.pad_system.router import PADRouter
from api.logging_config import logger


@lru_cache(maxsize=1)
def get_pad_router() -> PADRouter:
    """
    Charge une seule instance du PADRouter.

    Important :
    - le modèle V6 est lourd ;
    - il ne faut pas le recharger à chaque requête ;
    - lru_cache garantit une instance unique pendant le runtime API.
    """
    logger.info("Initialisation du PADRouter et chargement du modèle V6...")
    router = PADRouter(
        enable_liveness=True,
        load_video_model=True,
        challenge_ttl_seconds=60,
        allow_dataset_video_id=False,
        challenge_storage_mode="sqlite",
    )
    logger.info("PADRouter chargé avec succès.")
    return router