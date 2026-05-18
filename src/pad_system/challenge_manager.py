from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any
import uuid
import random


CHALLENGE_TYPES = [
    "BLINK",
    "SMILE",
    "TURN_LEFT",
    "EYEBROW_RAISE",
]


CHALLENGE_INSTRUCTIONS = {
    "BLINK": "Clignez les yeux.",
    "SMILE": "Souriez pendant quelques secondes.",
    "TURN_LEFT": "Tournez legerement la tete vers la gauche.",
    "EYEBROW_RAISE": "Levez les sourcils pendant une seconde.",
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

    Version adaptée au flux final :
    - challenge vidéo direct ;
    - session_id unique ;
    - challenge_id unique ;
    - expiration courte ;
    - usage unique du challenge.

    Plus tard, pour une version production :
    - stockage Redis ou base de données ;
    - association avec user_id / transaction_id ;
    - logs sécurité.
    """

    def __init__(self, ttl_seconds: int = 60):
        self.ttl_seconds = int(ttl_seconds)
        self.sessions: Dict[str, ChallengeSession] = {}

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

        self.sessions[challenge_id] = session

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
        return self.sessions.get(challenge_id)

    def validate_challenge(self, challenge_id: str) -> Dict[str, Any]:
        session = self.sessions.get(challenge_id)

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
                "challenge": asdict(session),
            }

        now = datetime.now(timezone.utc)
        expires_at = datetime.fromisoformat(session.expires_at)

        if now > expires_at:
            return {
                "is_valid": False,
                "reason": "challenge_expired",
                "challenge": asdict(session),
            }

        return {
            "is_valid": True,
            "reason": "challenge_valid",
            "challenge": asdict(session),
        }

    def mark_used(self, challenge_id: str) -> Dict[str, Any]:
        session = self.sessions.get(challenge_id)

        if session is None:
            return {
                "success": False,
                "reason": "challenge_not_found",
            }

        session.used = True

        return {
            "success": True,
            "reason": "challenge_marked_as_used",
            "challenge": asdict(session),
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

        return len(to_delete)