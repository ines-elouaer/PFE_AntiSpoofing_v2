from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any


class SQLiteChallengeStore:
    """
    Stockage SQLite simple pour les challenges PAD.

    Objectif :
    - éviter la perte des challenges si le serveur redémarre ;
    - garder une trace locale des challenges créés ;
    - rendre le prototype plus stable pour Docker et démonstration société.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS challenges (
                    challenge_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    challenge_type TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, challenge: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO challenges (
                    challenge_id,
                    session_id,
                    challenge_type,
                    instruction,
                    created_at,
                    expires_at,
                    used,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    challenge["challenge_id"],
                    challenge["session_id"],
                    challenge["challenge_type"],
                    challenge["instruction"],
                    challenge["created_at"],
                    challenge["expires_at"],
                    int(bool(challenge.get("used", False))),
                    challenge.get("source", "api"),
                ),
            )
            conn.commit()

    def get(self, challenge_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT *
                FROM challenges
                WHERE challenge_id = ?
                """,
                (challenge_id,),
            ).fetchone()

        if row is None:
            return None

        data = dict(row)
        data["used"] = bool(data["used"])
        return data

    def mark_used(self, challenge_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE challenges
                SET used = 1
                WHERE challenge_id = ?
                """,
                (challenge_id,),
            )
            conn.commit()

        return cursor.rowcount > 0

    def delete(self, challenge_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM challenges
                WHERE challenge_id = ?
                """,
                (challenge_id,),
            )
            conn.commit()

        return cursor.rowcount > 0

    def cleanup_expired_before(self, now_iso: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM challenges
                WHERE expires_at < ?
                """,
                (now_iso,),
            )
            conn.commit()

        return cursor.rowcount