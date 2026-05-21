from pathlib import Path


class Settings:
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    API_NAME = "PAD Banking API"
    API_VERSION = "v6.0"
    API_DESCRIPTION = (
        "API  pour la détection d'attaques de présentation faciale "
        "dans un contexte bancaire."
    )

    RUNTIME_UPLOAD_DIR = PROJECT_ROOT / "data" / "runtime_uploads"
    RUNTIME_RESULTS_DIR = PROJECT_ROOT / "data" / "runtime_results"
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_FILE = LOG_DIR / "pad_api.log"

    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    MAX_UPLOAD_SIZE_MB = 100

    MODEL_NAME = "V6 final — gated rPPG + behavior-pose"

    VIDEO_CHECKPOINT = (
        PROJECT_ROOT
        / "experiments"
        / "03_final_models"
        / "video_v6_behavior_pose"
        / "mixed_casia_axon_local_msu_gated_hard_balanced_rppg_v3"
        / "seed42"
        / "best_model.pth"
    )

    BEHAVIOR_POSE_MODEL = (
        PROJECT_ROOT
        / "experiments"
        / "03_final_models"
        / "banking_demo_models"
        / "final_models"
        / "behavior_pose_v6"
        / "behavior_pose_clf.pkl"
    )

    FACE_LANDMARKER_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"

    FUSION_FORMULA = (
        "score_final_v6 = 0.85 * score_v3_multimodal "
        "+ 0.15 * score_behavior_pose"
    )


settings = Settings()


def ensure_runtime_dirs() -> None:
   
    settings.RUNTIME_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.RUNTIME_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)