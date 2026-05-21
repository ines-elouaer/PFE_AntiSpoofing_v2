from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Any
import uuid
import random

try:
    from src.pad_system.challenge_store import SQLiteChallengeStore
except Exception:
    SQLiteChallengeStore = None


CHALLENGE_TYPES = [
    "BLINK",
    "SMILE",
    "TURN_LEFT",
]

CHALLENGE_INSTRUCTIONS = {
    "BLINK": "Clignez les yeux.",
    "SMILE": "Souriez pendant quelques secondes.",
    "TURN_LEFT": "Tournez legerement la tete vers la gauche.",
}


@dataclass
class ChallengeSession:
    session_id: str
    challenge_id: str
    challenge_type: str
    instruction: str
    created_at: str
    expires_at: str
    used: bool = False
    source: str = "direct_video_challenge"


class ChallengeManager:
    """
    Gestion des challenges vidéo.

    Version stabilisée :
    - stockage mémoire pour compatibilité ;
    - stockage SQLite optionnel pour API / Docker ;
    - session_id unique ;
    - challenge_id unique ;
    - expiration courte ;
    - usage unique du challenge.
    """

    def __init__(
        self,
        ttl_seconds: int = 60,
        storage_mode: str = "sqlite",
        db_path: Optional[str] = None,
    ):
        self.ttl_seconds = int(ttl_seconds)
        self.sessions: Dict[str, ChallengeSession] = {}

        self.storage_mode = str(storage_mode).lower().strip()
        self.store = None

        if self.storage_mode == "sqlite":
            if SQLiteChallengeStore is None:
                print("[CHALLENGE][WARN] SQLiteChallengeStore indisponible. Fallback mémoire.")
                self.storage_mode = "memory"
            else:
                if db_path is None:
                    project_root = Path(__file__).resolve().parents[2]
                    db_path = (
                        project_root
                        / "data"
                        / "runtime_challenges"
                        / "challenges.sqlite"
                    )

                self.store = SQLiteChallengeStore(db_path)
                print(f"[CHALLENGE] Stockage SQLite activé: {db_path}")

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================

    def _session_to_dict(self, session: ChallengeSession) -> Dict[str, Any]:
        return asdict(session)

    def _dict_to_session(self, data: Dict[str, Any]) -> ChallengeSession:
        return ChallengeSession(
            session_id=str(data["session_id"]),
            challenge_id=str(data["challenge_id"]),
            challenge_type=str(data["challenge_type"]),
            instruction=str(data["instruction"]),
            created_at=str(data["created_at"]),
            expires_at=str(data["expires_at"]),
            used=bool(data.get("used", False)),
            source=str(data.get("source", "api")),
        )

    def _save_session(self, session: ChallengeSession) -> None:
        self.sessions[session.challenge_id] = session

        if self.store is not None:
            self.store.save(self._session_to_dict(session))

    def _load_session(self, challenge_id: str) -> Optional[ChallengeSession]:
        session = self.sessions.get(challenge_id)

        if session is not None:
            return session

        if self.store is not None:
            data = self.store.get(challenge_id)

            if data is not None:
                session = self._dict_to_session(data)
                self.sessions[challenge_id] = session
                return session

        return None

    # ==========================================================
    # PUBLIC METHODS
    # ==========================================================

    def create_challenge(
        self,
        source: str = "direct_video_challenge",
        challenge_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self.ttl_seconds)

        if challenge_type is None:
            challenge_type = random.choice(CHALLENGE_TYPES)

        challenge_type = str(challenge_type).upper().strip()

        if challenge_type not in CHALLENGE_TYPES:
            raise ValueError(f"Challenge non supporté: {challenge_type}")

        session_id = str(uuid.uuid4())
        challenge_id = str(uuid.uuid4())

        session = ChallengeSession(
            session_id=session_id,
            challenge_id=challenge_id,
            challenge_type=challenge_type,
            instruction=CHALLENGE_INSTRUCTIONS[challenge_type],
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            used=False,
            source=source,
        )

        self._save_session(session)

        return {
            "status": "CREATED",
            "session_id": session_id,
            "challenge_id": challenge_id,
            "challenge_type": challenge_type,
            "instruction": session.instruction,
            "expires_in_seconds": self.ttl_seconds,
            "expires_at": session.expires_at,
            "source": source,
        }

    def get_challenge(self, challenge_id: str) -> Optional[ChallengeSession]:
        return self._load_session(challenge_id)

    def validate_challenge(self, challenge_id: str) -> Dict[str, Any]:
        session = self._load_session(challenge_id)

        if session is None:
            return {
                "is_valid": False,
                "reason": "challenge_not_found",
                "challenge": None,
            }

        if session.used:
            return {
                "is_valid": False,
                "reason": "challenge_already_used",
                "challenge": self._session_to_dict(session),
            }

        now = datetime.now(timezone.utc)
        expires_at = datetime.fromisoformat(session.expires_at)

        if now > expires_at:
            return {
                "is_valid": False,
                "reason": "challenge_expired",
                "challenge": self._session_to_dict(session),
            }

        return {
            "is_valid": True,
            "reason": "challenge_valid",
            "challenge": self._session_to_dict(session),
        }

    def mark_used(self, challenge_id: str) -> Dict[str, Any]:
        session = self._load_session(challenge_id)

        if session is None:
            return {
                "success": False,
                "reason": "challenge_not_found",
            }

        session.used = True
        self.sessions[challenge_id] = session

        if self.store is not None:
            self.store.mark_used(challenge_id)

        return {
            "success": True,
            "reason": "challenge_marked_as_used",
            "challenge": self._session_to_dict(session),
        }

    def cleanup_expired(self) -> int:
        now = datetime.now(timezone.utc)

        to_delete = []

        for challenge_id, session in self.sessions.items():
            expires_at = datetime.fromisoformat(session.expires_at)

            if now > expires_at:
                to_delete.append(challenge_id)

        for challenge_id in to_delete:
            del self.sessions[challenge_id]

        deleted_sqlite = 0

        if self.store is not None:
            deleted_sqlite = self.store.cleanup_expired_before(now.isoformat())

        return len(to_delete) + int(deleted_sqlite)