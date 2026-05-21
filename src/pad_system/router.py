import time
from pathlib import Path
from typing import Optional, Dict, Any

from src.pad_system.video_model import VideoPADModel
from src.pad_system.input_validator import PADInputValidator
from src.pad_system.challenge_manager import ChallengeManager
from src.pad_system.active_liveness_system import ActiveLivenessSystem

try:
    from src.pad_system.banking_adapter import banking_decision, label_from_decision
except Exception:
    def banking_decision(
        score: float,
        profile: str = "video",
        video_quality_score: float = 0.70,
    ) -> str:
        """
        Fallback local si banking_adapter.py n'est pas disponible.

        Convention :
        - score proche de 0 => REAL
        - score proche de 1 => SPOOF
        """

        score = float(score)

        if score < 0.30:
            return "ACCEPT"

        if score < 0.60:
            return "RETRY"

        return "REJECT"

    def label_from_decision(decision: str) -> str:
        if decision == "ACCEPT":
            return "REAL"

        if decision == "REJECT":
            return "SPOOF"

        return "UNCERTAIN"


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class PADRouter:
    """
    Routeur principal du système PAD bancaire.

    Flux final :
        challenge/start
        -> capture vidéo
        -> validation vidéo
        -> validation challenge_id
        -> liveness actif
        -> modèle PAD vidéo V6
        -> quality-aware policy
        -> règle runtime robuste
        -> décision ACCEPT / RETRY / REJECT

    La branche image est volontairement désactivée.

    Cette version ajoute un runtime_profile pour diagnostiquer
    le temps de chaque étape avant Docker.
    """

    def __init__(
        self,
        enable_liveness: bool = True,
        load_video_model: bool = True,
        challenge_ttl_seconds: int = 60,
        allow_dataset_video_id: bool = True,
        challenge_storage_mode: str = "sqlite",
    ):
        self.default_enable_liveness = enable_liveness
        self.allow_dataset_video_id = allow_dataset_video_id

        self.project_root = Path(__file__).resolve().parents[2]

        self.input_validator = PADInputValidator(
            allow_images=False,
            min_video_frames=16,
            min_video_duration_sec=1.0,
            min_width=64,
            min_height=64,
        )

        self.challenge_manager = ChallengeManager(
            ttl_seconds=challenge_ttl_seconds,
            storage_mode=challenge_storage_mode,
        )

        self.video_model = VideoPADModel() if load_video_model else None

        self.liveness_system = self._try_init_liveness_system()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def analyze(
        self,
        file_path: str,
        enable_liveness: Optional[bool] = None,
        challenge: Optional[str] = None,
        challenge_type: Optional[str] = None,
        session_id: Optional[str] = None,
        challenge_id: Optional[str] = None,
        require_challenge_validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyse une vidéo.

        Cas 1 — vidéo fichier :
            router.analyze(
                "data/demo/webcam_challenge.mp4",
                challenge_id="...",
                require_challenge_validation=True,
                enable_liveness=True
            )

        Cas 2 — video_id dataset CASIA/Axon pour debug :
            router.analyze("13_1", enable_liveness=False)

        Le flux image est volontairement désactivé.
        """

        if enable_liveness is None:
            enable_liveness = self.default_enable_liveness

        selected_challenge = challenge_type or challenge

        # Cas debug : identifiant vidéo dataset, ex: "13_1" ou "13_1.avi"
        if self.allow_dataset_video_id and self._looks_like_dataset_video_id(file_path):
            video_id = self._normalize_dataset_video_id(file_path)

            return self._analyze_video(
                input_value=video_id,
                original_input=file_path,
                enable_liveness=False,
                challenge=selected_challenge,
                validation={
                    "is_valid": True,
                    "input_type": "video",
                    "reason": "dataset_video_id",
                    "details": {
                        "video_id": video_id,
                        "original_input": file_path,
                    },
                },
                session_id=session_id,
                challenge_id=challenge_id,
                require_challenge_validation=False,
            )

        validation = self.input_validator.validate(file_path)

        if not validation.is_valid:
            return self._invalid_input_response(
                validation=validation,
                session_id=session_id,
                challenge_id=challenge_id,
            )

        if validation.input_type != "video":
            return {
                "type": validation.input_type,
                "branch": "input_validation",
                "score": None,
                "label": "UNCERTAIN",
                "decision": "INVALID_INPUT",
                "next_action": "USE_VIDEO_CHALLENGE",
                "liveness": {
                    "status": "NOT_APPLICABLE",
                    "passed": False,
                    "challenge": selected_challenge,
                    "reason": "only_video_input_is_supported",
                    "metrics": {},
                },
                "model_called": False,
                "message": "Le flux image est désactivé. Utiliser une vidéo challenge.",
                "validation": validation.to_dict(),
                "session_id": session_id,
                "challenge_id": challenge_id,
                "challenge": None,
                "original_input": file_path,
            }

        return self._analyze_video(
            input_value=file_path,
            original_input=file_path,
            enable_liveness=enable_liveness,
            challenge=selected_challenge,
            validation=validation.to_dict(),
            session_id=session_id,
            challenge_id=challenge_id,
            require_challenge_validation=require_challenge_validation,
        )

    # ==========================================================
    # VIDEO BRANCH
    # ==========================================================

    def _analyze_video(
        self,
        input_value: str,
        original_input: str,
        enable_liveness: bool,
        challenge: Optional[str],
        validation: Dict[str, Any],
        session_id: Optional[str] = None,
        challenge_id: Optional[str] = None,
        require_challenge_validation: bool = False,
    ) -> Dict[str, Any]:

        print(f"[INFO] Analyse vidéo: {input_value}")

        router_t0 = time.perf_counter()
        runtime_profile: Dict[str, Any] = {
            "profiling_enabled": True,
            "input_value": str(input_value),
            "enable_liveness": bool(enable_liveness),
            "require_challenge_validation": bool(require_challenge_validation),
        }

        def finish_profile() -> Dict[str, Any]:
            runtime_profile["router_total_sec"] = round(
                time.perf_counter() - router_t0,
                4,
            )
            return runtime_profile

        if self.video_model is None:
            return {
                "type": "video",
                "branch": "video_dynamic_decision_system",
                "score": None,
                "label": "UNCERTAIN",
                "decision": "MODEL_NOT_LOADED",
                "next_action": "RETRY_LATER",
                "liveness": {
                    "status": "DISABLED",
                    "passed": False,
                    "challenge": challenge,
                    "reason": "video_model_not_loaded",
                    "metrics": {},
                },
                "model_called": False,
                "runtime_profile": finish_profile(),
                "message": "Modèle vidéo non chargé.",
                "validation": validation,
                "session_id": session_id,
                "challenge_id": challenge_id,
                "challenge": None,
                "original_input": original_input,
            }

        challenge_validation = None
        expected_challenge = challenge

        # ======================================================
        # 1. Validation challenge_id
        # ======================================================

        if require_challenge_validation:
            if not challenge_id:
                return {
                    "type": "video",
                    "branch": "challenge_validation",
                    "score": None,
                    "label": "UNCERTAIN",
                    "decision": "INVALID_CHALLENGE",
                    "next_action": "RESTART_CHALLENGE",
                    "liveness": {
                        "status": "NOT_APPLICABLE",
                        "passed": False,
                        "challenge": challenge,
                        "reason": "missing_challenge_id",
                        "metrics": {},
                    },
                    "model_called": False,
                    "runtime_profile": finish_profile(),
                    "message": (
                        "challenge_id manquant. Il faut démarrer un challenge "
                        "avant d'envoyer la vidéo."
                    ),
                    "validation": validation,
                    "session_id": session_id,
                    "challenge_id": challenge_id,
                    "challenge": None,
                    "original_input": original_input,
                }

            t0 = time.perf_counter()
            challenge_validation = self.challenge_manager.validate_challenge(challenge_id)
            runtime_profile["challenge_validation_sec"] = round(
                time.perf_counter() - t0,
                4,
            )

            if not challenge_validation.get("is_valid", False):
                return {
                    "type": "video",
                    "branch": "challenge_validation",
                    "score": None,
                    "label": "UNCERTAIN",
                    "decision": "INVALID_CHALLENGE",
                    "next_action": "RESTART_CHALLENGE",
                    "liveness": {
                        "status": "NOT_APPLICABLE",
                        "passed": False,
                        "challenge": challenge,
                        "reason": challenge_validation.get("reason", "invalid_challenge"),
                        "metrics": {},
                    },
                    "model_called": False,
                    "runtime_profile": finish_profile(),
                    "message": f"Challenge invalide: {challenge_validation.get('reason')}",
                    "validation": validation,
                    "session_id": session_id,
                    "challenge_id": challenge_id,
                    "challenge": challenge_validation.get("challenge"),
                    "original_input": original_input,
                }

            challenge_data = challenge_validation["challenge"]
            expected_challenge = challenge_data.get("challenge_type", challenge)
            session_id = challenge_data.get("session_id", session_id)

        else:
            runtime_profile["challenge_validation_sec"] = 0.0

        is_real_video_file = Path(str(input_value)).exists()

        # ======================================================
        # 2. Liveness actif
        # ======================================================

        if enable_liveness and is_real_video_file and expected_challenge is not None:
            t0 = time.perf_counter()
            liveness_result = self._run_liveness(
                video_path=input_value,
                challenge=expected_challenge,
            )
            runtime_profile["active_liveness_sec"] = round(
                time.perf_counter() - t0,
                4,
            )

            if not liveness_result.get("passed", False):
                t0 = time.perf_counter()

                if require_challenge_validation and challenge_id:
                    self.challenge_manager.mark_used(challenge_id)

                runtime_profile["mark_challenge_used_sec"] = round(
                    time.perf_counter() - t0,
                    4,
                )

                return {
                    "type": "video",
                    "branch": "video_dynamic_decision_system",
                    "score": None,
                    "label": "UNCERTAIN",
                    "decision": "RETRY",
                    "next_action": "RETRY_VIDEO_CHALLENGE",
                    "liveness": liveness_result,
                    "model_called": False,
                    "runtime_profile": finish_profile(),
                    "message": "Liveness échoué. Nouvelle capture vidéo requise.",
                    "validation": validation,
                    "session_id": session_id,
                    "challenge_id": challenge_id,
                    "challenge": (
                        challenge_validation.get("challenge")
                        if challenge_validation
                        else None
                    ),
                    "original_input": original_input,
                }

        elif enable_liveness and is_real_video_file and expected_challenge is None:
            runtime_profile["active_liveness_sec"] = 0.0
            liveness_result = {
                "status": "DISABLED",
                "passed": True,
                "challenge": None,
                "reason": "liveness_enabled_but_no_challenge_provided",
                "metrics": {},
            }

        else:
            runtime_profile["active_liveness_sec"] = 0.0
            liveness_result = {
                "status": "DISABLED",
                "passed": True,
                "challenge": expected_challenge,
                "reason": "liveness_disabled_for_this_call",
                "metrics": {},
            }

        # ======================================================
        # 3. Modèle PAD vidéo V6
        # ======================================================

        t0 = time.perf_counter()
        model_details = self.video_model.predict_with_details(input_value)
        runtime_profile["video_model_predict_sec"] = round(
            time.perf_counter() - t0,
            4,
        )

        score = float(model_details["score_final_v6"])
        video_quality_score = float(model_details.get("video_quality_score", 0.70))
        behavior_pose_status = model_details.get("behavior_pose_status", "")

        # ======================================================
        # 4. Décision bancaire quality-aware
        # ======================================================

        t0 = time.perf_counter()
        decision = banking_decision(
            score,
            profile="video",
            video_quality_score=video_quality_score,
        )
        runtime_profile["banking_decision_sec"] = round(
            time.perf_counter() - t0,
            4,
        )

        label = label_from_decision(decision)

        # ======================================================
        # 5. Runtime robust guard
        # ======================================================

        t0 = time.perf_counter()

        instability_detected = (
            isinstance(behavior_pose_status, str)
            and (
                behavior_pose_status.startswith("low_pose_valid_rate")
                or behavior_pose_status.startswith("high_skipped_rate")
                or behavior_pose_status.startswith("unstable_pose")
            )
        )

        if decision == "REJECT" and instability_detected and score >= 0.90:
            decision = "REJECT"
            label = "SPOOF"
            next_action = "NONE"
            message = (
                "Analyse vidéo terminée. Vidéo rejetée comme attaque probable "
                "malgré une instabilité des features comportementales."
            )

        elif decision == "REJECT" and instability_detected and score < 0.90:
            decision = "RETRY"
            label = "UNCERTAIN"
            next_action = "RETRY_VIDEO_CAPTURE"
            message = (
                "Vidéo instable avec score élevé. "
                "Nouvelle capture recommandée."
            )

        elif decision == "ACCEPT":
            next_action = "NONE"
            message = "Analyse video terminee. Video acceptee."

        elif decision == "REJECT":
            next_action = "NONE"
            message = "Analyse video terminee. Video rejetee comme attaque probable."

        elif decision == "RETRY":
            next_action = "RETRY_VIDEO_CAPTURE"
            message = "Analyse video ambiguë. Nouvelle capture recommandée."

        else:
            next_action = "RETRY_VIDEO_CAPTURE"
            message = "Décision video incertaine."

        runtime_profile["runtime_guard_sec"] = round(
            time.perf_counter() - t0,
            4,
        )

        # ======================================================
        # 6. Marquer le challenge comme utilisé
        # ======================================================

        used_result = None

        t0 = time.perf_counter()

        if require_challenge_validation and challenge_id:
            used_result = self.challenge_manager.mark_used(challenge_id)

        runtime_profile["mark_challenge_used_sec"] = round(
            time.perf_counter() - t0,
            4,
        )

        return {
            "type": "video",
            "branch": "video_dynamic_decision_system",
            "score": round(score, 4),
            "label": label,
            "decision": decision,
            "next_action": next_action,
            "liveness": liveness_result,
            "model_called": True,
            "model_details": model_details,
            "video_quality_score": round(video_quality_score, 4),
            "behavior_pose_status": behavior_pose_status,
            "runtime_instability_guard": {
                "enabled": True,
                "triggered": bool(instability_detected),
                "reason": behavior_pose_status if instability_detected else None,
                "score_extreme_reject_kept": bool(
                    decision == "REJECT"
                    and instability_detected
                    and score >= 0.90
                ),
            },
            "runtime_profile": finish_profile(),
            "message": message,
            "validation": validation,
            "session_id": session_id,
            "challenge_id": challenge_id,
            "challenge": (
                used_result.get("challenge")
                if used_result and used_result.get("success")
                else challenge_validation.get("challenge")
                if challenge_validation
                else None
            ),
            "original_input": original_input,
        }

    # ==========================================================
    # CHALLENGE PUBLIC HELPERS
    # ==========================================================

    def create_video_challenge(
        self,
        source: str = "direct_video_challenge",
        challenge_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Crée un challenge vidéo direct.

        Utilisé par :
        - API /pad/challenge/start
        - démo webcam
        - tests.
        """

        return self.challenge_manager.create_challenge(
            source=source,
            challenge_type=challenge_type,
        )

    def validate_video_challenge(self, challenge_id: str) -> Dict[str, Any]:
        return self.challenge_manager.validate_challenge(challenge_id)

    # ==========================================================
    # LIVENESS
    # ==========================================================

    def _try_init_liveness_system(self):
        """
        Charge le système de liveness actif basé sur MediaPipe FaceLandmarker.
        """

        model_path = self.project_root / "models" / "face_landmarker.task"

        try:
            liveness_system = ActiveLivenessSystem(
                model_path=str(model_path)
            )
            print("[LIVENESS] ActiveLivenessSystem chargé avec succès.")
            return liveness_system

        except Exception as e:
            print(f"[LIVENESS] Erreur chargement ActiveLivenessSystem: {e}")
            print("[LIVENESS] Liveness désactivé.")
            return None

    def _run_liveness(
        self,
        video_path: str,
        challenge: Optional[str],
    ) -> Dict[str, Any]:

        if self.liveness_system is None:
            return {
                "status": "FAIL",
                "passed": False,
                "challenge": challenge,
                "reason": "no_liveness_system_available",
                "metrics": {},
            }

        if challenge is None:
            return {
                "status": "FAIL",
                "passed": False,
                "challenge": challenge,
                "reason": "missing_challenge",
                "metrics": {},
            }

        try:
            result = self.liveness_system.analyze(
                video_path=video_path,
                challenge=challenge,
            )
            return self._normalize_liveness_result(result, challenge)

        except Exception as e:
            return {
                "status": "ERROR",
                "passed": False,
                "challenge": challenge,
                "reason": f"liveness_exception: {str(e)}",
                "metrics": {},
            }

    def _normalize_liveness_result(
        self,
        result: Any,
        challenge: Optional[str],
    ) -> Dict[str, Any]:

        if isinstance(result, dict):
            passed = bool(result.get("passed", result.get("status") == "PASS"))
            status = result.get("status", "PASS" if passed else "FAIL")
            reason = result.get("reason", "liveness_checked")
            metrics = result.get("metrics", {})

            return {
                "status": status,
                "passed": passed,
                "challenge": result.get("challenge", challenge),
                "reason": reason,
                "metrics": metrics,
            }

        if isinstance(result, bool):
            return {
                "status": "PASS" if result else "FAIL",
                "passed": result,
                "challenge": challenge,
                "reason": "boolean_liveness_result",
                "metrics": {},
            }

        return {
            "status": "UNKNOWN",
            "passed": False,
            "challenge": challenge,
            "reason": "unsupported_liveness_result_format",
            "metrics": {
                "raw_result": str(result),
            },
        }

    # ==========================================================
    # VALIDATION / UTILS
    # ==========================================================

    def _invalid_input_response(
        self,
        validation,
        session_id: Optional[str],
        challenge_id: Optional[str],
    ) -> Dict[str, Any]:

        next_action = "RETRY_VIDEO_CAPTURE"

        if validation.reason == "image_input_disabled":
            next_action = "USE_VIDEO_CHALLENGE"

        return {
            "type": validation.input_type,
            "branch": "input_validation",
            "score": None,
            "label": "UNCERTAIN",
            "decision": "INVALID_INPUT",
            "next_action": next_action,
            "liveness": {
                "status": "NOT_APPLICABLE",
                "passed": False,
                "challenge": None,
                "reason": validation.reason,
                "metrics": validation.details,
            },
            "model_called": False,
            "message": f"Entrée invalide: {validation.reason}",
            "validation": validation.to_dict(),
            "session_id": session_id,
            "challenge_id": challenge_id,
            "challenge": None,
        }

    def _looks_like_dataset_video_id(self, value: str) -> bool:
        """
        Reconnaît un identifiant vidéo dataset, par exemple :
        - 13_1
        - 13_1.avi

        Cette option est utile pour les tests sur CASIA/Axon préparés.
        Elle n'est pas destinée à l'API publique.
        """

        p = Path(str(value))

        if p.exists():
            return False

        value_str = str(value).strip()

        if "\\" in value_str or "/" in value_str:
            return False

        suffix = p.suffix.lower()

        if suffix == "":
            return True

        if suffix in VIDEO_EXTS:
            return True

        return False

    def _normalize_dataset_video_id(self, value: str) -> str:
        p = Path(str(value))

        if p.suffix.lower() in VIDEO_EXTS:
            return p.stem

        return str(value).strip()