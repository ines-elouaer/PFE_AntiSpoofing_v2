from functools import lru_cache

from src.pad_system.router import PADRouter
from api.logging_config import logger


@lru_cache(maxsize=1)
def get_pad_router() -> PADRouter:
    
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