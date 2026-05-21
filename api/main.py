from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings, ensure_runtime_dirs
from api.logging_config import logger
from api.routes import health, challenge, pad


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
        logger.info("Démarrage de PAD Banking API.")
        logger.info("Version: %s", settings.API_VERSION)
        logger.info("Modèle: %s", settings.MODEL_NAME)

    @app.on_event("shutdown")
    def shutdown_event():
        logger.info("Arrêt de PAD Banking API.")

    return app


app = create_app()