from .utils import detect_input_type
from .image_model import ImagePADModel
from .video_model import VideoPADModel
from .active_liveness import ActiveLivenessChallenge
from .video_decision import VideoDecisionSystem
from .banking_adapter import banking_decision, label_from_decision


class PADRouter:
    """
    Routeur principal du système PAD bancaire.

    Architecture :
    - si entrée image :
        → branche image statique
        → modèle CelebA-Spoof
        → score image
        → décision bancaire

    - si entrée vidéo :
        → système de décision vidéo
        → challenge actif / liveness
        → si FAIL : RETRY
        → si PASS : modèle vidéo CNN+LSTM
        → score vidéo
        → décision bancaire
    """

    def __init__(self, enable_liveness: bool = False):
        self.enable_liveness = enable_liveness

        # Branche image
        self.image_model = ImagePADModel()

        # Branche vidéo
        self.video_model = VideoPADModel()
        self.liveness = ActiveLivenessChallenge()

        # Système de décision vidéo
        self.video_decision = VideoDecisionSystem(
            video_model=self.video_model,
            liveness_module=self.liveness,
        )

    def analyze(self, file_path, challenge=None, enable_liveness=None):
        """
        Analyse une entrée image ou vidéo.

        Paramètres :
        - file_path :
            chemin image, chemin vidéo, ou video_id CASIA.

        - challenge :
            TURN_LEFT / TURN_RIGHT / BLINK / SMILE.
            Utilisé seulement pour la branche vidéo.

        - enable_liveness :
            None  → utilise self.enable_liveness
            True  → force le liveness
            False → désactive le liveness
        """

        input_type = detect_input_type(file_path)

        if enable_liveness is None:
            use_liveness = self.enable_liveness
        else:
            use_liveness = bool(enable_liveness)

        # ======================================================
        # Branche image : CelebA uniquement
        # ======================================================
        if input_type == "image":
            score = self.image_model.predict(file_path)
            score = float(score)

            decision = banking_decision(score)
            label = label_from_decision(decision)

            return {
                "type": "image",
                "branch": "image_static_celeba",
                "score": round(score, 4),
                "label": label,
                "decision": decision,
                "liveness": {
                    "status": "NOT_APPLICABLE",
                    "passed": True,
                    "reason": "image_branch_no_active_liveness",
                },
                "model_called": True,
                "message": "Image analysée par le modèle CelebA-Spoof.",
            }

        # ======================================================
        # Branche vidéo : système de décision vidéo
        # ======================================================
        elif input_type == "video":
            video_result = self.video_decision.analyze(
                video_path_or_id=file_path,
                challenge=challenge,
                enable_liveness=use_liveness,
            )

            return {
                "type": "video",
                "branch": "video_dynamic_decision_system",
                "score": video_result["score"],
                "label": video_result["label"],
                "decision": video_result["decision"],
                "liveness": video_result["liveness"],
                "model_called": video_result["model_called"],
                "message": video_result["message"],
            }

        # ======================================================
        # Type non supporté
        # ======================================================
        else:
            return {
                "type": input_type,
                "branch": "unknown",
                "score": None,
                "label": "UNKNOWN",
                "decision": "RETRY",
                "liveness": None,
                "model_called": False,
                "message": "Type d'entrée non supporté.",
            }