from typing import Optional, Dict, Any

from .banking_adapter import banking_decision, label_from_decision


class VideoDecisionSystem:
    """
    Système de décision pour la branche vidéo.

    Rôle :
    - appliquer le challenge actif si activé
    - si le challenge échoue : retourner RETRY sans appeler le modèle vidéo
    - si le challenge réussit : appeler le modèle vidéo CNN+LSTM
    - appliquer la décision bancaire finale
    """

    def __init__(self, video_model, liveness_module):
        self.video_model = video_model
        self.liveness_module = liveness_module

    def analyze(
        self,
        video_path_or_id: str,
        challenge: Optional[str] = None,
        enable_liveness: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyse une entrée vidéo.

        Paramètres :
        - video_path_or_id :
            chemin vidéo réel, exemple :
            E:/PFE_AntiSpoofing_v2/data/demo/webcam_challenge.mp4

            ou video_id CASIA, exemple :
            13_1

        - challenge :
            TURN_LEFT / TURN_RIGHT / BLINK / SMILE
            Si None, le module liveness peut choisir automatiquement.

        - enable_liveness :
            True  → appliquer le challenge actif avant modèle vidéo
            False → aller directement au modèle vidéo
        """

        # ======================================================
        # Étape 1 — Challenge actif / liveness
        # ======================================================
        if enable_liveness:
            liveness_result = self.liveness_module.check(
                video_path_or_id,
                challenge=challenge,
            )

            # Si le challenge échoue, on ne lance PAS le modèle vidéo.
            if not liveness_result.get("passed", False):
                return {
                    "score": None,
                    "label": "UNCERTAIN",
                    "decision": "RETRY",
                    "liveness": liveness_result,
                    "message": "Challenge actif échoué. Nouvelle tentative requise.",
                    "model_called": False,
                }

        else:
            liveness_result = {
                "status": "DISABLED",
                "passed": True,
                "challenge": challenge,
                "reason": "liveness_disabled_for_this_call",
                "metrics": {},
            }

        # ======================================================
        # Étape 2 — Modèle vidéo CNN+LSTM
        # ======================================================
        score = self.video_model.predict(video_path_or_id)
        score = float(score)

        decision = banking_decision(score)
        label = label_from_decision(decision)

        return {
            "score": round(score, 4),
            "label": label,
            "decision": decision,
            "liveness": liveness_result,
            "message": "Analyse vidéo terminée.",
            "model_called": True,
        }