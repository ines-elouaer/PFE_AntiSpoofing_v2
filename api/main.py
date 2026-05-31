from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings, ensure_runtime_dirs
from api.logging_config import logger
from api.routes import health, challenge, pad
from api.dependencies import get_pad_router


def create_app() -> FastAPI:
    """
    Crée l'application FastAPI principale.
    """
    ensure_runtime_dirs()

    app = FastAPI(
        title=settings.API_NAME,
        version=settings.API_VERSION,
        description=settings.API_DESCRIPTION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # à restreindre en production réelle
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(challenge.router)
    app.include_router(pad.router)

    @app.on_event("startup")
    def startup_event():
        """
        Démarrage de l'API.

        Amélioration professionnelle :
        - création des répertoires runtime ;
        - chargement immédiat du PADRouter ;
        - chargement du modèle FOMA V6 au démarrage ;
        - évite une latence forte lors de la première vérification.
        """
        logger.info("Démarrage de PAD Banking API.")
        logger.info("Version: %s", settings.API_VERSION)
        logger.info("Modèle: %s", settings.MODEL_NAME)

        logger.info("Warmup PAD: chargement du PADRouter et du modèle FOMA V6 au démarrage...")

        try:
            get_pad_router()
            logger.info("Warmup PAD terminé: PADRouter et modèle FOMA V6 prêts.")

        except Exception as exc:
            logger.exception(
                "Warmup PAD échoué: impossible de charger le modèle FOMA V6 au démarrage."
            )
            raise exc

    @app.on_event("shutdown")
    def shutdown_event():
        logger.info("Arrêt de PAD Banking API.")

    return app


app = create_app()